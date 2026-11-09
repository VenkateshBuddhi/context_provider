"""
Context Provider Python SDK Client.

High-level async (and sync) interface for integrating the Local Context Handoff
System into autonomous AI agent loops, LangChain pipelines, and AutoGen workflows.
"""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List

from context_provider.storage.db import DatabaseManager
from context_provider.storage.models import EventRecord, EventType
from context_provider.export.exporter import ExportEngine


class ContextProvider:
    """
    High-level Python SDK client for local context management and agent handoffs.

    Usage (async)::

        from context_provider import ContextProvider

        async def main():
            client = ContextProvider(workspace_path=".")
            await client.initialize()
            await client.record_note("Switched to async generator pattern", category="decision")
            artifacts = await client.export()
            print(artifacts)

    Usage (sync)::

        client = ContextProvider(workspace_path=".")
        client.initialize_sync()
        ctx = client.get_active_context_sync()
    """

    def __init__(
        self,
        workspace_path: "str | Path" = ".",
        db_path: "Optional[str | Path]" = None,
    ) -> None:
        self.workspace = Path(workspace_path).resolve()
        self.db = DatabaseManager(db_path) if db_path else DatabaseManager()
        self.exporter = ExportEngine(self.db, self.workspace)
        self._project = None
        self._session = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize database and get or create project/session."""
        await self.db.initialize()
        self._project = await self.db.get_or_create_project(str(self.workspace))
        self._session = await self.db.get_active_session(self._project.id)
        if not self._session:
            self._session = await self.db.create_session(self._project.id)
        self._initialized = True

    async def record_note(self, text: str, category: str = "decision") -> EventRecord:
        """
        Record an architectural decision, bug finding, hypothesis, or goal.

        Args:
            text: The note content.
            category: One of 'goal', 'decision', 'bug', 'hypothesis', 'observation'.

        Returns:
            The persisted EventRecord.
        """
        self._require_initialized()
        import datetime
        record = EventRecord(
            session_id=self._session.id,
            event_type=EventType.NOTE,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            payload={"note": text, "category": category},
        )
        await self.db.log_event(record)
        return record

    async def log_command(
        self,
        command: str,
        exit_code: int,
        stderr: str = "",
        stdout: str = "",
        duration_ms: Optional[int] = None,
    ) -> EventRecord:
        """
        Log the outcome of a shell command execution.

        Args:
            command: The command string that was run.
            exit_code: Process exit code (0 = success).
            stderr: stderr tail output (last ~400 chars recommended).
            stdout: stdout tail output.
            duration_ms: Execution duration in milliseconds.

        Returns:
            The persisted EventRecord.
        """
        self._require_initialized()
        import datetime
        payload: Dict[str, Any] = {
            "command": command,
            "exit_code": exit_code,
            "stderr_tail": stderr[-400:] if stderr else "",
            "stdout_tail": stdout[-400:] if stdout else "",
        }
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        record = EventRecord(
            session_id=self._session.id,
            event_type=EventType.COMMAND,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            payload=payload,
        )
        await self.db.log_event(record)
        return record

    async def get_active_context(self) -> Dict[str, Any]:
        """
        Retrieve the current workspace status suitable for agent prompts.

        Returns:
            Dict with keys: 'project', 'session_id', 'started_at', 'recent_notes',
            'total_events', 'status'.
        """
        self._require_initialized()
        events = await self.db.get_session_events(self._session.id)
        notes = [
            e.payload.get("note", "")
            for e in events
            if e.event_type == EventType.NOTE
        ]
        return {
            "status": "active",
            "project": self._project.name,
            "session_id": self._session.id,
            "started_at": self._session.started_at,
            "total_events": len(events),
            "recent_notes": notes[-10:],
        }

    async def export(self) -> Dict[str, Path]:
        """
        Synthesize CONTEXT.md, HANDOFF.md, and context.json in the workspace.

        Runs the full pipeline: episode clustering -> LAYA contradiction scoring
        -> llama3.1 synthesis -> markdown and JSON file generation.

        Returns:
            Dict mapping artifact names to their absolute file paths.
        """
        self._require_initialized()
        return await self.exporter.export_all(self._session.id)

    # ------------------------------------------------------------------
    # Synchronous convenience wrappers
    # ------------------------------------------------------------------

    def initialize_sync(self) -> None:
        """Synchronous wrapper around :meth:`initialize`."""
        asyncio.run(self.initialize())

    def record_note_sync(self, text: str, category: str = "decision") -> EventRecord:
        """Synchronous wrapper around :meth:`record_note`."""
        return asyncio.run(self.record_note(text, category))

    def log_command_sync(
        self,
        command: str,
        exit_code: int,
        stderr: str = "",
        stdout: str = "",
        duration_ms: Optional[int] = None,
    ) -> EventRecord:
        """Synchronous wrapper around :meth:`log_command`."""
        return asyncio.run(self.log_command(command, exit_code, stderr, stdout, duration_ms))

    def get_active_context_sync(self) -> Dict[str, Any]:
        """Synchronous wrapper around :meth:`get_active_context`."""
        return asyncio.run(self.get_active_context())

    def export_sync(self) -> Dict[str, Path]:
        """Synchronous wrapper around :meth:`export`."""
        return asyncio.run(self.export())

    # ------------------------------------------------------------------
    # Context Manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "ContextProvider":
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass  # Reserved for future connection teardown

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "ContextProvider is not initialized. "
                "Call 'await client.initialize()' or use 'async with ContextProvider(...) as client:'."
            )

    @property
    def project(self):
        """The active ProjectRecord (available after initialize())."""
        return self._project

    @property
    def session(self):
        """The active SessionRecord (available after initialize())."""
        return self._session

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"ContextProvider(workspace='{self.workspace}', status='{status}')"
