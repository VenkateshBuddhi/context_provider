"""
File System Watcher and Live Capture Service for real-time workspace tracking.

Uses Watchdog with .gitignore filtering, event debouncing, and a thread-safe
event queue to stream file changes (created, modified, deleted, moved) and
Git state shifts directly into SQLite.
"""

import os
import time
import fnmatch
import threading
import queue
import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional, Set, List, Dict, Any
from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileModifiedEvent,
    FileDeletedEvent,
    FileMovedEvent,
)

from context_provider.config import default_config
from context_provider.storage.models import EventRecord, EventType
from context_provider.storage.db import DatabaseManager
from context_provider.capture.git_tracker import GitTracker


logger = logging.getLogger(__name__)

# Hardcoded exclusion list for context system artifacts to prevent infinite feedback loops
CONTEXT_SYSTEM_ARTIFACT_NAMES: Set[str] = {
    "context.md",
    "handoff.md",
    "context.json",
    "evaluation.json",
    "evaluation.md",
    "differential_experiment.json",
    "differential_experiment.md",
}

CONTEXT_SYSTEM_ARTIFACT_PATTERNS: List[str] = [
    "*.db",
    "*.db-wal",
    "*.db-shm",
    "*.db-journal",
]


def is_context_system_artifact(path: Path) -> bool:
    """
    Checks if a given path matches hardcoded context provider artifacts or SQLite storage files.
    This check drops events immediately to prevent infinite observation loops.
    """
    filename = path.name.lower()

    if filename in CONTEXT_SYSTEM_ARTIFACT_NAMES:
        return True

    for pattern in CONTEXT_SYSTEM_ARTIFACT_PATTERNS:
        if fnmatch.fnmatch(filename, pattern):
            return True

    return False


class GitIgnoreFilter:
    """Filters file paths using .gitignore patterns and default exclusion rules."""

    def __init__(self, workspace_root: Path, additional_patterns: Optional[List[str]] = None):
        self.workspace_root = workspace_root.resolve()
        self.patterns = list(default_config.default_ignore_patterns)
        if additional_patterns:
            self.patterns.extend(additional_patterns)
        self._load_gitignore()

    def _load_gitignore(self) -> None:
        """Parses .gitignore file from the project root if it exists."""
        gitignore_path = self.workspace_root / ".gitignore"
        if gitignore_path.is_file():
            try:
                with open(gitignore_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            self.patterns.append(line)
            except Exception:
                pass

    def is_ignored(self, path: Path) -> bool:
        """Determines whether a file or directory path matches ignore patterns."""
        if is_context_system_artifact(path):
            return True

        try:
            rel_path = path.resolve().relative_to(self.workspace_root).as_posix()
        except ValueError:
            return True  # Outside workspace root

        parts = rel_path.split("/")
        # Filter hidden directories and files (except explicit .gitignore)
        if any(p.startswith(".") and p not in [".", "..", ".gitignore"] for p in parts):
            return True
        # Filter standard build & dependency folders
        if any(p in ["node_modules", "__pycache__", "venv", ".venv", "dist", "build", ".qodo"] for p in parts):
            return True

        for pattern in self.patterns:
            clean_pat = pattern.rstrip("/").replace("\\", "/")
            if fnmatch.fnmatch(rel_path, clean_pat) or fnmatch.fnmatch(path.name, clean_pat):
                return True
            if pattern.endswith("/**") and rel_path.startswith(pattern[:-3]):
                return True
        return False


class DebouncedFileSystemHandler(FileSystemEventHandler):
    """
    Handles filesystem events, filters ignored files, and debounces rapid file modifications.
    Normalizes relative paths consistently across Windows and POSIX.
    """

    def __init__(
        self,
        workspace_root: Path,
        on_event_callback: Callable[[EventRecord], None],
        session_id: str,
        debounce_seconds: float = 1.0,
    ):
        super().__init__()
        self.workspace_root = workspace_root.resolve()
        self.on_event_callback = on_event_callback
        self.session_id = session_id
        self.debounce_seconds = debounce_seconds
        self.filter = GitIgnoreFilter(workspace_root)
        self._last_modified_times: Dict[str, float] = {}

    def _is_excluded(self, path: Path) -> bool:
        """
        Fast-path check: completely drops context system artifacts and
        evaluates gitignore filter rules before queueing events.
        """
        if is_context_system_artifact(path):
            return True
        return self.filter.is_ignored(path)

    def _should_debounce(self, file_path_str: str) -> bool:
        now = time.time()
        last_time = self._last_modified_times.get(file_path_str, 0.0)
        if now - last_time < self.debounce_seconds:
            return True
        self._last_modified_times[file_path_str] = now
        return False

    def _get_rel_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.workspace_root).as_posix())
        except Exception:
            return str(path.name)

    def _get_file_size(self, path: Path) -> int:
        try:
            return path.stat().st_size if path.exists() else 0
        except Exception:
            return 0

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._is_excluded(path):
            return

        rel_path = self._get_rel_path(path)
        size = self._get_file_size(path)

        record = EventRecord(
            session_id=self.session_id,
            event_type=EventType.FILE,
            payload={
                "action": "created",
                "file_path": str(path.resolve()),
                "relative_path": rel_path,
                "size_bytes": size,
            },
            evidence_meta={"source": "watchdog"},
        )
        self.on_event_callback(record)

    def on_modified(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._is_excluded(path):
            return
        if self._should_debounce(str(path)):
            return

        rel_path = self._get_rel_path(path)
        size = self._get_file_size(path)

        record = EventRecord(
            session_id=self.session_id,
            event_type=EventType.FILE,
            payload={
                "action": "modified",
                "file_path": str(path.resolve()),
                "relative_path": rel_path,
                "size_bytes": size,
            },
            evidence_meta={"source": "watchdog"},
        )
        self.on_event_callback(record)

    def on_deleted(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._is_excluded(path):
            return

        rel_path = self._get_rel_path(path)
        record = EventRecord(
            session_id=self.session_id,
            event_type=EventType.FILE,
            payload={
                "action": "deleted",
                "file_path": str(path.resolve()),
                "relative_path": rel_path,
            },
            evidence_meta={"source": "watchdog"},
        )
        self.on_event_callback(record)

    def on_moved(self, event):
        if event.is_directory:
            return
        src_path = Path(event.src_path)
        dest_path = Path(event.dest_path)
        if self._is_excluded(src_path) or self._is_excluded(dest_path):
            return

        record = EventRecord(
            session_id=self.session_id,
            event_type=EventType.FILE,
            payload={
                "action": "moved",
                "src_path": str(src_path.resolve()),
                "dest_path": str(dest_path.resolve()),
                "src_rel": self._get_rel_path(src_path),
                "dest_rel": self._get_rel_path(dest_path),
            },
            evidence_meta={"source": "watchdog"},
        )
        self.on_event_callback(record)


class FileWatcher:
    """Manages the background filesystem observer thread."""

    def __init__(
        self,
        workspace_root: Path,
        session_id: str,
        on_event_callback: Callable[[EventRecord], None],
        debounce_seconds: float = 1.0,
    ):
        self.workspace_root = workspace_root.resolve()
        self.session_id = session_id
        self.on_event_callback = on_event_callback
        self.debounce_seconds = debounce_seconds
        self.observer = Observer()
        self.handler = DebouncedFileSystemHandler(
            workspace_root=self.workspace_root,
            on_event_callback=self.on_event_callback,
            session_id=self.session_id,
            debounce_seconds=self.debounce_seconds,
        )

    def start(self) -> None:
        """Starts watching the workspace directory recursively."""
        self.observer.schedule(self.handler, path=str(self.workspace_root), recursive=True)
        self.observer.start()

    def stop(self) -> None:
        """Stops the observer thread."""
        self.observer.stop()
        self.observer.join()


class LiveCaptureService:
    """
    Continuous capture engine coordinating FileWatcher and GitTracker.
    Buffers events in a thread-safe queue and asynchronously persists them to SQLite.
    """

    def __init__(
        self,
        workspace_root: Path,
        session_id: str,
        db: DatabaseManager,
        git_poll_interval_seconds: float = 3.0,
        on_event_callback: Optional[Callable[[EventRecord], None]] = None,
    ):
        self.workspace_root = workspace_root.resolve()
        self.session_id = session_id
        self.db = db
        self.git_poll_interval_seconds = git_poll_interval_seconds
        self.on_event_callback = on_event_callback

        self.queue: queue.Queue = queue.Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._git_thread: Optional[threading.Thread] = None
        self._db_thread: Optional[threading.Thread] = None

        self.git_tracker = GitTracker(self.workspace_root, self.session_id)
        self.watcher = FileWatcher(
            workspace_root=self.workspace_root,
            session_id=self.session_id,
            on_event_callback=self._enqueue_event,
            debounce_seconds=default_config.watcher_debounce_seconds,
        )

    def _enqueue_event(self, record: EventRecord) -> None:
        """Pushes an observed event to the internal persistence queue."""
        self.queue.put(record)
        if self.on_event_callback:
            try:
                self.on_event_callback(record)
            except Exception:
                pass

    def _persistence_worker(self) -> None:
        """Background thread consuming queued events and writing to SQLite."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self._running or not self.queue.empty():
            try:
                event = self.queue.get(timeout=0.5)
                loop.run_until_complete(self.db.log_event(event))
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Failed to persist event in background worker: %s", e)

        loop.close()

    def _git_poll_worker(self) -> None:
        """Background thread periodically checking for new Git commits or status changes."""
        if not self.git_tracker.is_git_repo():
            return

        while self._running:
            time.sleep(self.git_poll_interval_seconds)
            try:
                commit_event = self.git_tracker.capture_commit_event()
                if commit_event:
                    self._enqueue_event(commit_event)
            except Exception:
                pass

    def _db_poll_worker(self) -> None:
        """Background thread polling SQLite for new external events (commands, tests, notes)."""
        import sqlite3
        import json
        from context_provider.storage.models import EventRecord, EventType

        seen_event_ids = set()
        db_path = str(self.db.db_path)

        # Pre-seed with existing events so we only stream new events
        try:
            conn = sqlite3.connect(db_path, timeout=3.0)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM events WHERE session_id = ?;", (self.session_id,))
            for r in cursor.fetchall():
                seen_event_ids.add(r[0])
            conn.close()
        except Exception:
            pass

        while self._running:
            time.sleep(0.5)
            try:
                conn = sqlite3.connect(db_path, timeout=3.0)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, session_id, event_type, timestamp, payload, evidence_meta
                    FROM events
                    WHERE session_id = ? AND event_type NOT IN ('file', 'git')
                    ORDER BY rowid ASC;
                    """,
                    (self.session_id,),
                )
                rows = cursor.fetchall()
                conn.close()

                for row in rows:
                    ev_id, s_id, ev_type_val, ts, payload_str, meta_str = row
                    if ev_id in seen_event_ids:
                        continue
                    seen_event_ids.add(ev_id)

                    rec = EventRecord(
                        id=ev_id,
                        session_id=s_id,
                        event_type=EventType(ev_type_val),
                        timestamp=ts,
                        payload=json.loads(payload_str) if payload_str else {},
                        evidence_meta=json.loads(meta_str) if meta_str else {},
                    )
                    if self.on_event_callback:
                        try:
                            self.on_event_callback(rec)
                        except Exception:
                            pass
            except Exception:
                pass

    def start(self) -> None:
        """Starts the filesystem observer and background persistence threads."""
        self._running = True
        self.watcher.start()

        self._worker_thread = threading.Thread(
            target=self._persistence_worker, daemon=True, name="CapturePersistenceWorker"
        )
        self._worker_thread.start()

        self._git_thread = threading.Thread(
            target=self._git_poll_worker, daemon=True, name="GitPollWorker"
        )
        self._git_thread.start()

        self._db_thread = threading.Thread(
            target=self._db_poll_worker, daemon=True, name="DbPollWorker"
        )
        self._db_thread.start()

    def stop(self) -> None:
        """Stops live monitoring and drains queued events to disk."""
        self._running = False
        self.watcher.stop()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
        if self._git_thread and self._git_thread.is_alive():
            self._git_thread.join(timeout=3.0)
        if self._db_thread and self._db_thread.is_alive():
            self._db_thread.join(timeout=3.0)
