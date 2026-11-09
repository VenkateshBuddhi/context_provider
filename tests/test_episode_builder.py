"""Unit tests for Phase 2: Episode Builder and Event Clustering."""

import pytest
from datetime import datetime, timezone, timedelta
from context_provider.pipeline.episode_builder import EpisodeBuilder
from context_provider.storage.models import EventRecord, EventType, EpisodeStatus
from context_provider.storage.db import DatabaseManager
import tempfile
import shutil
from pathlib import Path


def _make_event(session_id: str, event_type: EventType, timestamp: str, payload: dict) -> EventRecord:
    return EventRecord(
        session_id=session_id,
        event_type=event_type,
        timestamp=timestamp,
        payload=payload,
    )


def test_jaccard_similarity():
    builder = EpisodeBuilder()
    set_a = {"backend/main.py", "backend/rag.py"}
    set_b = {"backend/main.py", "backend/persistence.py"}
    set_c = {"frontend/App.jsx", "frontend/index.html"}

    # Overlap: 1 / 3 = 0.333
    assert abs(builder.compute_jaccard_similarity(set_a, set_b) - 0.333) < 0.01
    # Disjoint: 0.0
    assert builder.compute_jaccard_similarity(set_a, set_c) == 0.0
    # Empty: 1.0
    assert builder.compute_jaccard_similarity(set(), set()) == 1.0


def test_temporal_clustering():
    builder = EpisodeBuilder(idle_gap_seconds=300)
    base_time = datetime(2026, 11, 3, 10, 0, 0, tzinfo=timezone.utc)

    t0 = base_time.isoformat()
    t1 = (base_time + timedelta(seconds=60)).isoformat()
    t2 = (base_time + timedelta(seconds=120)).isoformat()
    t3 = (base_time + timedelta(seconds=500)).isoformat()  # Gap > 300s
    t4 = (base_time + timedelta(seconds=560)).isoformat()
    t5 = (base_time + timedelta(seconds=620)).isoformat()

    events = [
        _make_event("s1", EventType.FILE, t0, {"relative_path": "a.py", "action": "modified"}),
        _make_event("s1", EventType.FILE, t1, {"relative_path": "a.py", "action": "modified"}),
        _make_event("s1", EventType.FILE, t2, {"relative_path": "a.py", "action": "modified"}),
        _make_event("s1", EventType.FILE, t3, {"relative_path": "b.py", "action": "created"}),
        _make_event("s1", EventType.FILE, t4, {"relative_path": "b.py", "action": "modified"}),
        _make_event("s1", EventType.FILE, t5, {"relative_path": "b.py", "action": "modified"}),
    ]

    episodes = builder.build_episodes_from_events("s1", events, session_is_active=False)
    assert len(episodes) == 2
    assert "TEMPORAL_IDLE_GAP" in episodes[1].metadata["boundary_reason"]
    assert "a.py" in episodes[0].files_touched
    assert "b.py" in episodes[1].files_touched


def test_hard_boundary_git_commit():
    builder = EpisodeBuilder()
    base_time = datetime(2026, 11, 3, 10, 0, 0, tzinfo=timezone.utc)

    t0 = base_time.isoformat()
    t1 = (base_time + timedelta(seconds=30)).isoformat()
    t2 = (base_time + timedelta(seconds=45)).isoformat()
    t3 = (base_time + timedelta(seconds=50)).isoformat()
    t4 = (base_time + timedelta(seconds=55)).isoformat()

    events = [
        _make_event("s1", EventType.FILE, t0, {"relative_path": "backend/rag.py", "action": "modified"}),
        _make_event("s1", EventType.GIT, t1, {"action": "commit", "subject": "feat: add chunking"}),
        _make_event("s1", EventType.FILE, t2, {"relative_path": "backend/main.py", "action": "modified"}),
        _make_event("s1", EventType.FILE, t3, {"relative_path": "backend/main.py", "action": "modified"}),
        _make_event("s1", EventType.FILE, t4, {"relative_path": "backend/main.py", "action": "modified"}),
    ]

    episodes = builder.build_episodes_from_events("s1", events, session_is_active=False)
    assert len(episodes) == 2
    assert episodes[0].status == EpisodeStatus.COMPLETED
    assert "GIT_COMMIT_CHECKPOINT" in episodes[0].metadata["boundary_reason"]


def test_module_switch_jaccard_segmentation():
    builder = EpisodeBuilder(subsystem_pause_seconds=30.0, file_overlap_threshold=0.15)
    base_time = datetime(2026, 11, 3, 10, 0, 0, tzinfo=timezone.utc)

    t0 = base_time.isoformat()
    t1 = (base_time + timedelta(seconds=5)).isoformat()
    t2 = (base_time + timedelta(seconds=10)).isoformat()
    t3 = (base_time + timedelta(seconds=50)).isoformat()  # pause > 30s on completely different files
    t4 = (base_time + timedelta(seconds=55)).isoformat()
    t5 = (base_time + timedelta(seconds=60)).isoformat()

    events = [
        _make_event("s1", EventType.FILE, t0, {"relative_path": "backend/rag.py", "action": "modified"}),
        _make_event("s1", EventType.FILE, t1, {"relative_path": "backend/rag.py", "action": "modified"}),
        _make_event("s1", EventType.FILE, t2, {"relative_path": "backend/rag.py", "action": "modified"}),
        _make_event("s1", EventType.FILE, t3, {"relative_path": "frontend/src/App.jsx", "action": "modified"}),
        _make_event("s1", EventType.FILE, t4, {"relative_path": "frontend/src/App.jsx", "action": "modified"}),
        _make_event("s1", EventType.FILE, t5, {"relative_path": "frontend/src/App.jsx", "action": "modified"}),
    ]

    episodes = builder.build_episodes_from_events("s1", events, session_is_active=False)
    assert len(episodes) == 2
    assert "MODULE_SWITCH" in episodes[1].metadata["boundary_reason"]


def test_status_abandoned_on_revert():
    builder = EpisodeBuilder()
    base_time = datetime(2026, 11, 3, 10, 0, 0, tzinfo=timezone.utc)

    t0 = base_time.isoformat()
    t1 = (base_time + timedelta(seconds=20)).isoformat()

    events = [
        _make_event("s1", EventType.FILE, t0, {"relative_path": "exp.py", "action": "created"}),
        _make_event("s1", EventType.GIT, t1, {"action": "commit", "is_revert": True, "subject": "Revert experiment"}),
    ]

    episodes = builder.build_episodes_from_events("s1", events, session_is_active=False)
    assert len(episodes) == 1
    assert episodes[0].status == EpisodeStatus.ABANDONED


@pytest.mark.asyncio
async def test_episode_database_persistence():
    temp_dir = tempfile.mkdtemp()
    try:
        db = DatabaseManager(db_path=Path(temp_dir) / "store.db")
        await db.initialize()
        project = await db.get_or_create_project(temp_dir)
        session = await db.create_session(project.id)

        ev1 = EventRecord(session_id=session.id, event_type=EventType.FILE, payload={"relative_path": "test.py"})
        ev2 = EventRecord(session_id=session.id, event_type=EventType.GIT, payload={"action": "commit", "subject": "initial commit"})
        await db.log_event(ev1)
        await db.log_event(ev2)

        builder = EpisodeBuilder()
        episodes = builder.build_episodes_from_events(session.id, [ev1, ev2])

        # Persist and link
        await db.save_episodes(episodes)
        for ep in episodes:
            await db.link_events_to_episode(ep.id, ep.metadata["event_ids"])

        # Query back
        stored = await db.get_session_episodes(session.id)
        assert len(stored) == len(episodes)
        assert stored[0].status == EpisodeStatus.COMPLETED

        # Verify event linkage
        updated_events = await db.get_session_events(session.id)
        assert updated_events[0].episode_id == stored[0].id
        assert updated_events[1].episode_id == stored[0].id
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_trivial_action_lone_git_status_not_finalized():
    """Single trivial command (e.g. git status exit 0) must not spawn an episode."""
    builder = EpisodeBuilder()
    ev = _make_event("s1", EventType.COMMAND, "2026-11-03T10:00:00Z", {"command": "git status", "exit_code": 0})

    episodes = builder.build_episodes_from_events("s1", [ev])
    assert len(episodes) == 0
    assert len(builder.active_event_buffer) == 1
    assert builder.active_event_buffer[0].id == ev.id


def test_unfinalized_events_merge_on_subsequent_activity():
    """Buffered events must merge into the next episode once threshold is reached."""
    builder = EpisodeBuilder()
    ev1 = _make_event("s1", EventType.COMMAND, "2026-11-03T10:00:00Z", {"command": "git status", "exit_code": 0})
    episodes1 = builder.build_episodes_from_events("s1", [ev1])
    assert len(episodes1) == 0
    assert len(builder.active_event_buffer) == 1

    # More activity arrives
    ev2 = _make_event("s1", EventType.FILE, "2026-11-03T10:01:00Z", {"relative_path": "a.py", "action": "modified"})
    ev3 = _make_event("s1", EventType.FILE, "2026-11-03T10:02:00Z", {"relative_path": "b.py", "action": "modified"})

    episodes2 = builder.build_episodes_from_events("s1", [ev2, ev3])
    assert len(episodes2) == 1
    assert episodes2[0].metadata["event_count"] == 3
    assert set(episodes2[0].metadata["event_ids"]) == {ev1.id, ev2.id, ev3.id}
    assert len(builder.active_event_buffer) == 0


def test_single_high_signal_developer_note():
    """A developer note is high-signal and should finalize an episode even if lone."""
    builder = EpisodeBuilder()
    ev = _make_event("s1", EventType.NOTE, "2026-11-03T10:00:00Z", {"note": "Decided to refactor auth", "category": "decision"})

    episodes = builder.build_episodes_from_events("s1", [ev])
    assert len(episodes) == 1
    assert episodes[0].metadata["event_count"] == 1


def test_single_high_signal_command_failure():
    """A command failure with non-zero exit code is high-signal and must finalize."""
    builder = EpisodeBuilder()
    ev = _make_event("s1", EventType.COMMAND, "2026-11-03T10:00:00Z", {"command": "pytest", "exit_code": 1, "failed": True})

    episodes = builder.build_episodes_from_events("s1", [ev])
    assert len(episodes) == 1
    assert episodes[0].metadata["event_count"] == 1


def test_single_high_signal_git_commit():
    """A git commit is high-signal and must finalize."""
    builder = EpisodeBuilder()
    ev = _make_event("s1", EventType.GIT, "2026-11-03T10:00:00Z", {"action": "commit", "subject": "fix: resolve deadlock"})

    episodes = builder.build_episodes_from_events("s1", [ev])
    assert len(episodes) == 1
    assert episodes[0].metadata["event_count"] == 1


def test_sub_threshold_events_merged_across_temporal_gap():
    """2 sub-threshold events before an idle gap merge with the 3rd event into one episode."""
    builder = EpisodeBuilder(idle_gap_seconds=300)
    ev1 = _make_event("s1", EventType.FILE, "2026-11-03T10:00:00Z", {"relative_path": "a.py", "action": "modified"})
    ev2 = _make_event("s1", EventType.FILE, "2026-11-03T10:01:00Z", {"relative_path": "a.py", "action": "modified"})
    ev3 = _make_event("s1", EventType.FILE, "2026-11-03T10:10:00Z", {"relative_path": "b.py", "action": "modified"})  # gap > 300s

    episodes = builder.build_episodes_from_events("s1", [ev1, ev2, ev3])
    # Cluster 1 had 2 events (sub-threshold, no high signal) -> did not split into a separate episode
    # Merged with ev3 to form 1 episode of 3 events
    assert len(episodes) == 1
    assert episodes[0].metadata["event_count"] == 3


