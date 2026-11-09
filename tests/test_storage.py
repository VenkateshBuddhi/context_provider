"""Unit tests for the SQLite Database Engine and data models."""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from context_provider.storage.db import DatabaseManager
from context_provider.storage.models import EventRecord, EventType, EpisodeRecord, EpisodeStatus


@pytest.fixture
def temp_db():
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_store.db"
    manager = DatabaseManager(db_path=db_path)
    yield manager
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_db_initialization_and_project(temp_db):
    await temp_db.initialize()
    project = await temp_db.get_or_create_project("/dummy/path", name="DummyProject")
    assert project.id is not None
    assert project.name == "DummyProject"

    # Fetch again should return identical record
    same_proj = await temp_db.get_or_create_project("/dummy/path")
    assert same_proj.id == project.id


@pytest.mark.asyncio
async def test_session_and_events(temp_db):
    await temp_db.initialize()
    project = await temp_db.get_or_create_project("/dummy/path", name="DummyProject")
    session = await temp_db.create_session(project.id)

    assert session.is_active is True
    active_s = await temp_db.get_active_session(project.id)
    assert active_s is not None
    assert active_s.id == session.id

    # Log an event
    event = EventRecord(
        session_id=session.id,
        event_type=EventType.NOTE,
        payload={"note": "Initial test note"},
    )
    await temp_db.log_event(event)

    events = await temp_db.get_session_events(session.id)
    assert len(events) == 1
    assert events[0].payload["note"] == "Initial test note"

    # Close session
    await temp_db.close_session(session.id)
    closed_active = await temp_db.get_active_session(project.id)
    assert closed_active is None

