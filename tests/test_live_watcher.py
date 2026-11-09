"""Unit tests for Phase 2: LiveCaptureService and file watching."""

import pytest
import time
import tempfile
import shutil
from pathlib import Path

from context_provider.capture.watcher import LiveCaptureService
from context_provider.storage.db import DatabaseManager
from context_provider.storage.models import EventType


@pytest.mark.asyncio
async def test_live_capture_service_file_events():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        db = DatabaseManager(db_path=temp_dir / "test_watch.db")
        await db.initialize()
        project = await db.get_or_create_project(str(temp_dir))
        session = await db.create_session(project.id)

        service = LiveCaptureService(temp_dir, session.id, db, git_poll_interval_seconds=1.0)
        service.start()

        # Create a new file in watched directory
        target_file = temp_dir / "sample.py"
        target_file.write_text("print('Live Watcher Test')\n", encoding="utf-8")

        # Allow watchdog to catch and worker to flush to sqlite
        time.sleep(1.5)

        service.stop()

        events = await db.get_session_events(session.id)
        file_events = [e for e in events if e.event_type == EventType.FILE]
        assert len(file_events) >= 1
        assert any("sample.py" in e.payload.get("relative_path", "") for e in file_events)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_live_capture_service_db_polling():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        db = DatabaseManager(db_path=temp_dir / "test_poll.db")
        await db.initialize()
        project = await db.get_or_create_project(str(temp_dir))
        session = await db.create_session(project.id)

        received_events = []

        def callback(rec):
            received_events.append(rec)

        service = LiveCaptureService(temp_dir, session.id, db, on_event_callback=callback)
        service.start()

        # Simulate external process logging a command event into SQLite
        from context_provider.storage.models import EventRecord
        cmd_event = EventRecord(
            session_id=session.id,
            event_type=EventType.COMMAND,
            payload={"command": "npm test", "exit_code": 0},
        )
        await db.log_event(cmd_event)

        # Allow DbPollWorker to detect event
        time.sleep(1.5)

        service.stop()

        assert any(
            e.event_type == EventType.COMMAND and e.payload.get("command") == "npm test"
            for e in received_events
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_live_watcher_drops_internal_artifacts_before_queue():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        db = DatabaseManager(db_path=temp_dir / "test_drop.db")
        await db.initialize()
        project = await db.get_or_create_project(str(temp_dir))
        session = await db.create_session(project.id)

        received_events = []

        def callback(rec):
            received_events.append(rec)

        service = LiveCaptureService(temp_dir, session.id, db, on_event_callback=callback)
        service.start()

        # Create/modify system artifacts that must be ignored
        (temp_dir / "CONTEXT.md").write_text("# Active Context\n", encoding="utf-8")
        (temp_dir / "HANDOFF.md").write_text("# Handoff\n", encoding="utf-8")
        (temp_dir / "context.json").write_text("{}", encoding="utf-8")
        (temp_dir / "storage.db").write_text("db binary", encoding="utf-8")
        (temp_dir / "storage.db-wal").write_text("wal data", encoding="utf-8")
        (temp_dir / "storage.db-shm").write_text("shm data", encoding="utf-8")
        (temp_dir / "EVALUATION.md").write_text("# Eval\n", encoding="utf-8")
        (temp_dir / "evaluation.json").write_text("{}", encoding="utf-8")

        # Create a legitimate developer file that SHOULD be tracked
        (temp_dir / "legitimate_code.py").write_text("print('tracked')", encoding="utf-8")

        time.sleep(1.5)
        service.stop()

        # Verify that ONLY legitimate_code.py was captured
        captured_paths = [e.payload.get("relative_path", "") for e in received_events if e.event_type == EventType.FILE]
        assert any("legitimate_code.py" in p for p in captured_paths)

        # Assert none of the system artifacts were queued or persisted
        assert not any("CONTEXT.md" in p for p in captured_paths)
        assert not any("HANDOFF.md" in p for p in captured_paths)
        assert not any("context.json" in p for p in captured_paths)
        assert not any("storage.db" in p for p in captured_paths)
        assert not any("EVALUATION.md" in p for p in captured_paths)
        assert not any("evaluation.json" in p for p in captured_paths)

        # Check DB directly to ensure queue was never fed
        events = await db.get_session_events(session.id)
        db_paths = [e.payload.get("relative_path", "") for e in events if e.event_type == EventType.FILE]
        assert any("legitimate_code.py" in p for p in db_paths)
        assert not any("CONTEXT.md" in p for p in db_paths)
        assert not any("HANDOFF.md" in p for p in db_paths)
        assert not any("context.json" in p for p in db_paths)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

