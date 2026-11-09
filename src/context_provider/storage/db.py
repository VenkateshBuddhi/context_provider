"""
Async SQLite Storage Engine for the Local Context Handoff System.

Manages connection lifecycle, schema initialization, and high-performance
asynchronous reads/writes using aiosqlite in WAL mode.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import aiosqlite

from context_provider.config import default_config
from context_provider.storage.models import (
    ProjectRecord,
    SessionRecord,
    EpisodeRecord,
    EventRecord,
    EventType,
    EpisodeStatus,
)


CREATE_TABLES_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    root_path TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    metadata TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    confidence_score REAL NOT NULL DEFAULT 0.5,
    summary TEXT,
    files_touched TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    episode_id TEXT,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    evidence_meta TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY(episode_id) REFERENCES episodes(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_episode ON events(episode_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
"""


class DatabaseManager:
    """Manages asynchronous database interactions with the local SQLite store."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or default_config.db_path
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensures the parent directory for SQLite DB exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Initializes tables and pragmas."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(CREATE_TABLES_SQL)
            await db.commit()

    # -------------------------------------------------------------------------
    # Project Operations
    # -------------------------------------------------------------------------
    async def get_or_create_project(self, root_path: str, name: Optional[str] = None) -> ProjectRecord:
        """Retrieves an existing project by normalized path or registers a new one."""
        norm_path = str(Path(root_path).resolve())
        project_name = name or Path(norm_path).name

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM projects WHERE root_path = ?", (norm_path,)
            )
            row = await cursor.fetchone()
            if row:
                return ProjectRecord(
                    id=row["id"],
                    name=row["name"],
                    root_path=row["root_path"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )

            # Insert new project
            new_project = ProjectRecord(name=project_name, root_path=norm_path)
            await db.execute(
                """
                INSERT INTO projects (id, name, root_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    new_project.id,
                    new_project.name,
                    new_project.root_path,
                    new_project.created_at,
                    new_project.updated_at,
                ),
            )
            await db.commit()
            return new_project

    # -------------------------------------------------------------------------
    # Session Operations
    # -------------------------------------------------------------------------
    async def create_session(self, project_id: str, metadata: Optional[Dict[str, Any]] = None) -> SessionRecord:
        """Starts a new active session for the project."""
        session = SessionRecord(project_id=project_id, metadata=metadata or {})
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO sessions (id, project_id, started_at, is_active, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.project_id,
                    session.started_at,
                    1 if session.is_active else 0,
                    json.dumps(session.metadata),
                ),
            )
            await db.commit()
        return session

    async def get_active_session(self, project_id: str) -> Optional[SessionRecord]:
        """Gets the most recently started active session for a project."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM sessions
                WHERE project_id = ? AND is_active = 1
                ORDER BY started_at DESC LIMIT 1
                """,
                (project_id,),
            )
            row = await cursor.fetchone()
            if not row:
                return None
            return SessionRecord(
                id=row["id"],
                project_id=row["project_id"],
                started_at=row["started_at"],
                ended_at=row["ended_at"],
                is_active=bool(row["is_active"]),
                metadata=json.loads(row["metadata"]),
            )

    async def close_session(self, session_id: str) -> None:
        """Marks a session as inactive and records completion timestamp."""
        async with aiosqlite.connect(self.db_path) as db:
            from context_provider.storage.models import utc_now_iso
            await db.execute(
                """
                UPDATE sessions
                SET is_active = 0, ended_at = ?
                WHERE id = ?
                """,
                (utc_now_iso(), session_id),
            )
            await db.commit()

    # -------------------------------------------------------------------------
    # Episode Operations
    # -------------------------------------------------------------------------
    async def create_episode(self, episode: EpisodeRecord) -> EpisodeRecord:
        """Stores a new episode record."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO episodes (
                    id, session_id, title, status, started_at, ended_at,
                    confidence_score, summary, files_touched, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.id,
                    episode.session_id,
                    episode.title,
                    episode.status.value if isinstance(episode.status, EpisodeStatus) else episode.status,
                    episode.started_at,
                    episode.ended_at,
                    episode.confidence_score,
                    episode.summary,
                    json.dumps(episode.files_touched),
                    json.dumps(episode.metadata),
                ),
            )
            await db.commit()
        return episode

    async def update_episode_status(
        self,
        episode_id: str,
        status: EpisodeStatus,
        summary: Optional[str] = None,
        confidence_score: Optional[float] = None,
    ) -> None:
        """Updates status, summary, and confidence score for an episode."""
        from context_provider.storage.models import utc_now_iso
        async with aiosqlite.connect(self.db_path) as db:
            query = "UPDATE episodes SET status = ?, ended_at = ?"
            params: List[Any] = [
                status.value if isinstance(status, EpisodeStatus) else status,
                utc_now_iso(),
            ]
            if summary is not None:
                query += ", summary = ?"
                params.append(summary)
            if confidence_score is not None:
                query += ", confidence_score = ?"
                params.append(confidence_score)
            query += " WHERE id = ?"
            params.append(episode_id)

            await db.execute(query, tuple(params))
            await db.commit()

    async def save_episodes(self, episodes: List[EpisodeRecord]) -> None:
        """Inserts or replaces multiple episode records in a single transaction."""
        async with aiosqlite.connect(self.db_path) as db:
            for ep in episodes:
                await db.execute(
                    """
                    INSERT INTO episodes (
                        id, session_id, title, status, started_at, ended_at,
                        confidence_score, summary, files_touched, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        title = excluded.title,
                        status = excluded.status,
                        started_at = excluded.started_at,
                        ended_at = excluded.ended_at,
                        confidence_score = excluded.confidence_score,
                        summary = excluded.summary,
                        files_touched = excluded.files_touched,
                        metadata = excluded.metadata
                    """,
                    (
                        ep.id,
                        ep.session_id,
                        ep.title,
                        ep.status.value if isinstance(ep.status, EpisodeStatus) else ep.status,
                        ep.started_at,
                        ep.ended_at,
                        ep.confidence_score,
                        ep.summary,
                        json.dumps(ep.files_touched),
                        json.dumps(ep.metadata),
                    ),
                )
            await db.commit()

    async def link_events_to_episode(self, episode_id: str, event_ids: List[str]) -> None:
        """Associates a list of event IDs with their parent episode ID."""
        if not event_ids:
            return
        async with aiosqlite.connect(self.db_path) as db:
            placeholders = ",".join("?" for _ in event_ids)
            await db.execute(
                f"UPDATE events SET episode_id = ? WHERE id IN ({placeholders})",
                [episode_id] + event_ids,
            )
            await db.commit()

    async def get_session_episodes(self, session_id: str) -> List[EpisodeRecord]:
        """Retrieves all episodes for a session ordered chronologically."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM episodes
                WHERE session_id = ?
                ORDER BY started_at ASC
                """,
                (session_id,),
            )
            rows = await cursor.fetchall()
            return [
                EpisodeRecord(
                    id=r["id"],
                    session_id=r["session_id"],
                    title=r["title"],
                    status=EpisodeStatus(r["status"]),
                    started_at=r["started_at"],
                    ended_at=r["ended_at"],
                    confidence_score=r["confidence_score"],
                    summary=r["summary"],
                    files_touched=json.loads(r["files_touched"]),
                    metadata=json.loads(r["metadata"]),
                )
                for r in rows
            ]

    # -------------------------------------------------------------------------
    # Event Operations
    # -------------------------------------------------------------------------
    async def log_event(self, event: EventRecord) -> EventRecord:
        """Appends a raw event to the chronological event log."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO events (
                    id, session_id, episode_id, event_type, timestamp, payload, evidence_meta
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.session_id,
                    event.episode_id,
                    event.event_type.value if isinstance(event.event_type, EventType) else event.event_type,
                    event.timestamp,
                    json.dumps(event.payload),
                    json.dumps(event.evidence_meta),
                ),
            )
            await db.commit()
        return event

    async def get_session_events(
        self, session_id: str, limit: int = 1000
    ) -> List[EventRecord]:
        """Retrieves events for a specific session ordered chronologically."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM events
                WHERE session_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (session_id, limit),
            )
            rows = await cursor.fetchall()
            return [
                EventRecord(
                    id=r["id"],
                    session_id=r["session_id"],
                    episode_id=r["episode_id"],
                    event_type=EventType(r["event_type"]),
                    timestamp=r["timestamp"],
                    payload=json.loads(r["payload"]),
                    evidence_meta=json.loads(r["evidence_meta"]),
                )
                for r in rows
            ]

