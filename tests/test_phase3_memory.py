"""Unit tests for Phase 3: Rules Engine, LAYA BERT Engine, and Synthesis Gateway."""

import pytest
from datetime import datetime, timezone, timedelta
from context_provider.pipeline.rules_engine import RulesEngine
from context_provider.analysis.laya_engine import LayaEngine
from context_provider.analysis.detector import ConflictDetector
from context_provider.synthesis.gateway import SynthesisGateway
from context_provider.storage.models import EventRecord, EventType, EpisodeRecord, EpisodeStatus


def _make_event(session_id: str, event_type: EventType, timestamp: str, payload: dict) -> EventRecord:
    return EventRecord(
        session_id=session_id,
        event_type=event_type,
        timestamp=timestamp,
        payload=payload,
    )


def test_rules_engine_thrashing_and_pivots():
    engine = RulesEngine()
    base_t = datetime(2026, 11, 3, 12, 0, 0, tzinfo=timezone.utc)

    events = [
        # Command failing twice
        _make_event("s1", EventType.COMMAND, (base_t + timedelta(seconds=10)).isoformat(), {"command": "pytest backend/", "exit_code": 1, "stderr_tail": "AssertionError"}),
        _make_event("s1", EventType.COMMAND, (base_t + timedelta(seconds=20)).isoformat(), {"command": "pytest backend/", "exit_code": 1, "stderr_tail": "AssertionError"}),
        # Note declaring pivot
        _make_event("s1", EventType.NOTE, (base_t + timedelta(seconds=25)).isoformat(), {"note": "Revert: streaming approach failed", "category": "decision"}),
    ]

    findings = engine.analyze_patterns(events)
    patterns = [f["pattern"] for f in findings]

    assert "COMMAND_THRASHING" in patterns
    assert "FAILED_ATTEMPT_PIVOT" in patterns


def test_rules_engine_test_regression():
    engine = RulesEngine()
    base_t = datetime(2026, 11, 3, 12, 0, 0, tzinfo=timezone.utc)

    events = [
        _make_event("s1", EventType.TEST, (base_t + timedelta(seconds=10)).isoformat(), {"command": "pytest", "exit_code": 0}),
        _make_event("s1", EventType.TEST, (base_t + timedelta(seconds=30)).isoformat(), {"command": "pytest", "exit_code": 1}),
    ]

    findings = engine.analyze_patterns(events)
    patterns = [f["pattern"] for f in findings]
    assert "TEST_REGRESSION" in patterns


def test_laya_engine_schema_safety():
    engine = LayaEngine()

    premise = "Fixed streaming response endpoint with AsyncGenerator"
    evidence = "Command failed with TypeError: StreamingResponse object is not subscriptable"

    result = engine.evaluate_pair(premise, evidence)
    assert isinstance(result, dict)
    assert "is_contradiction" in result
    assert isinstance(result["is_contradiction"], bool)
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["is_contradiction"] is True


def test_conflict_detector_staleness():
    detector = ConflictDetector(stale_time_threshold_hours=12.0)
    old_time = (datetime.now(timezone.utc) - timedelta(hours=20)).isoformat()

    stale_eval = detector.check_staleness(old_time)
    assert stale_eval["is_stale"] is True
    assert stale_eval["freshness_score"] < 0.5


def test_synthesis_gateway_event_compression():
    gateway = SynthesisGateway()
    base_t = datetime(2026, 11, 3, 12, 0, 0, tzinfo=timezone.utc)

    events = [
        _make_event("s1", EventType.FILE, (base_t + timedelta(seconds=1)).isoformat(), {"action": "modified", "relative_path": "backend/main.py", "size_bytes": 1024}),
        _make_event("s1", EventType.COMMAND, (base_t + timedelta(seconds=2)).isoformat(), {"command": "pytest", "exit_code": 1, "stderr_tail": "AssertionError: Expected 200 but got 500"}),
        _make_event("s1", EventType.NOTE, (base_t + timedelta(seconds=3)).isoformat(), {"note": "Decided to rollback", "category": "decision"}),
    ]

    compressed = gateway.compress_events(events)
    assert "FILE MODIFIED: backend/main.py" in compressed
    assert "CMD (exit 1): pytest" in compressed
    assert "NOTE [DECISION]: Decided to rollback" in compressed


def test_synthesis_gateway_traceback_bottom_lines_preserved():
    """Verifies that long tracebacks preserve the bottom fatal exception lines instead of cutting off at the top."""
    gateway = SynthesisGateway()
    base_t = datetime(2026, 11, 3, 12, 0, 0, tzinfo=timezone.utc)

    long_traceback = (
        "Traceback (most recent call last):\n"
        "  File \"C:\\very\\long\\path\\to\\project\\backend\\vector\\search\\indexer.py\", line 125, in load_index\n"
        "    from faiss import IndexFlatL2\n"
        "  File \"C:\\very\\long\\path\\to\\project\\.venv\\lib\\site-packages\\faiss\\__init__.py\", line 14, in <module>\n"
        "    from .loader import *\n"
        "ImportError: DLL load failed while importing _swigfaiss: The specified module could not be found."
    )

    ev = _make_event("s1", EventType.COMMAND, base_t.isoformat(), {
        "command": "python -m backend.indexer",
        "exit_code": 1,
        "stderr_tail": long_traceback,
    })

    compressed = gateway.compress_events([ev])
    # The fatal error must NOT be cut off
    assert "ImportError: DLL load failed while importing _swigfaiss" in compressed


def test_synthesis_gateway_stdout_fallback_for_error():
    """Verifies that if stderr_tail is empty on a crash (e.g. pytest), stdout_tail error is used."""
    gateway = SynthesisGateway()
    base_t = datetime(2026, 11, 3, 12, 0, 0, tzinfo=timezone.utc)

    ev = _make_event("s1", EventType.TEST, base_t.isoformat(), {
        "command": "pytest tests/test_rag.py",
        "exit_code": 1,
        "stderr_tail": "",
        "stdout_tail": "FAILED tests/test_rag.py::test_chroma - ModuleNotFoundError: No module named 'chromadb'",
    })

    compressed = gateway.compress_events([ev])
    assert "ModuleNotFoundError: No module named 'chromadb'" in compressed



def test_synthesis_gateway_sanitize_summary_boilerplate():
    """Verifies that conversational filler is stripped preceding the first markdown bullet."""
    gateway = SynthesisGateway()

    raw_response = (
        "Here are 3 concise markdown bullet points summarizing the episode:\n\n"
        "- **Attempt**: Migrated database from ChromaDB to FAISS.\n"
        "- **Evidence/Outcome**: Failed with compiler error.\n"
        "- **Status**: Abandoned."
    )
    sanitized = gateway.sanitize_summary(raw_response)

    assert "Here are 3 concise markdown bullet points" not in sanitized
    assert sanitized.startswith("- **Attempt**: Migrated database from ChromaDB to FAISS.")
    assert "- **Evidence/Outcome**: Failed with compiler error." in sanitized
    assert "- **Status**: Abandoned." in sanitized


def test_synthesis_gateway_sanitize_summary_asterisk_bullets():
    """Verifies that conversational boilerplate before asterisk bullets (* ) is stripped."""
    gateway = SynthesisGateway()

    raw_response = (
        "Certainly! Below is the summary you requested:\n"
        "* **Attempt**: Refactored API routes\n"
        "* **Evidence/Outcome**: Tests passed cleanly\n"
        "* **Status**: Completed"
    )
    sanitized = gateway.sanitize_summary(raw_response)

    assert "Certainly!" not in sanitized
    assert "Below is the summary" not in sanitized
    assert sanitized.startswith("* **Attempt**: Refactored API routes")


def test_synthesis_gateway_sanitize_summary_fallback_no_bullets():
    """Verifies that if no bullet points exist, the raw string is preserved stripped of whitespace."""
    gateway = SynthesisGateway()

    raw_response = "   Raw paragraph summary without any bullets or list formatting.   \n\n"
    sanitized = gateway.sanitize_summary(raw_response)

    assert sanitized == "Raw paragraph summary without any bullets or list formatting."


def test_synthesis_prompt_strict_directive():
    """Verifies that prompts contain the strict negative-constraint directives."""
    from context_provider.synthesis.prompts import EPISODE_SUMMARY_PROMPT, CONTEXT_FOCUS_PROMPT, TRAPS_AVOIDED_PROMPT
    directive = "Output ONLY the markdown bullet points. Do not include any introductory phrases, conversational filler, or acknowledging text."
    assert directive in EPISODE_SUMMARY_PROMPT
    assert directive in CONTEXT_FOCUS_PROMPT
    assert directive in TRAPS_AVOIDED_PROMPT
    assert "ReferenceError" in EPISODE_SUMMARY_PROMPT


def test_synthesize_active_focus_strips_boilerplate(monkeypatch):
    """Verifies that synthesize_active_focus sanitizes conversational filler."""
    gateway = SynthesisGateway()

    raw_boilerplate = (
        "Based on the provided information, here are 2-4 concise markdown bullet points:\n\n"
        "- Implementing user authentication routes in backend\n"
        "- Debugging medicine controller endpoint"
    )
    monkeypatch.setattr(gateway, "_call_ollama", lambda prompt, max_tokens: raw_boilerplate)

    result = gateway.synthesize_active_focus(notes=["note1"], recent_episodes=[])
    assert result is not None
    assert "Based on the provided information" not in result
    assert result.startswith("- Implementing user authentication routes in backend")



@pytest.mark.asyncio
async def test_export_engine_episode_filtering_for_handoff(tmp_path):
    """
    Verifies that HANDOFF.md strictly filters out low-signal inconclusive episodes
    (status == INCONCLUSIVE and confidence_score < 0.5), while keeping active,
    abandoned, completed, and high-confidence inconclusive episodes.
    Also verifies that the full unfiltered record is preserved in SQLite and context.json.
    """
    import json
    from unittest.mock import MagicMock
    from context_provider.storage.db import DatabaseManager
    from context_provider.export.exporter import ExportEngine

    db_path = tmp_path / "test_filter.db"
    db = DatabaseManager(db_path)
    await db.initialize()

    project = await db.get_or_create_project(str(tmp_path), name="test-filter-proj")
    session = await db.create_session(project.id)

    # 5 test episodes covering the condition matrix
    ep_low_inconclusive = EpisodeRecord(
        session_id=session.id,
        title="Low Confidence Inconclusive Task",
        status=EpisodeStatus.INCONCLUSIVE,
        confidence_score=0.25,
        started_at="2026-11-06T10:00:00Z",
    )
    ep_high_inconclusive = EpisodeRecord(
        session_id=session.id,
        title="High Confidence Inconclusive Task",
        status=EpisodeStatus.INCONCLUSIVE,
        confidence_score=0.75,
        started_at="2026-11-06T10:10:00Z",
    )
    ep_active = EpisodeRecord(
        session_id=session.id,
        title="Active Work Task",
        status=EpisodeStatus.ACTIVE,
        confidence_score=0.1,
        started_at="2026-11-06T10:20:00Z",
    )
    ep_abandoned = EpisodeRecord(
        session_id=session.id,
        title="Abandoned Sandbox Attempt",
        status=EpisodeStatus.ABANDONED,
        confidence_score=0.2,
        started_at="2026-11-06T10:30:00Z",
    )
    ep_completed = EpisodeRecord(
        session_id=session.id,
        title="Completed Core Refactor",
        status=EpisodeStatus.COMPLETED,
        confidence_score=0.4,
        started_at="2026-11-06T10:40:00Z",
    )

    all_episodes = [
        ep_low_inconclusive,
        ep_high_inconclusive,
        ep_active,
        ep_abandoned,
        ep_completed,
    ]

    engine = ExportEngine(db, tmp_path)
    # Mock clustering and confidence scoring to return our exact configured episodes
    engine.episode_builder.build_episodes_from_events = MagicMock(return_value=all_episodes)
    # Sequentially return the pre-configured confidence score for each episode
    engine.confidence_scorer.compute_episode_confidence = MagicMock(
        side_effect=[ep.confidence_score for ep in all_episodes]
    )
    engine.synthesis_gateway.summarize_episode = MagicMock(return_value="Synthesized episode summary")
    engine.synthesis_gateway.synthesize_active_focus = MagicMock(return_value="Focus area")
    engine.synthesis_gateway.synthesize_traps_avoided = MagicMock(return_value="Traps avoided")

    # Run export
    artifacts = await engine.export_all(session.id)

    handoff_text = artifacts["HANDOFF.md"].read_text(encoding="utf-8")
    context_text = artifacts["CONTEXT.md"].read_text(encoding="utf-8")
    json_data = json.loads(artifacts["context.json"].read_text(encoding="utf-8"))

    # 1. Verification of HANDOFF.md filtering
    # Low-confidence inconclusive MUST be dropped
    assert "Low Confidence Inconclusive Task" not in handoff_text

    # High-confidence inconclusive MUST be present
    assert "High Confidence Inconclusive Task" in handoff_text
    # Active, Abandoned, and Completed MUST be present regardless of confidence < 0.5
    assert "Active Work Task" in handoff_text
    assert "Abandoned Sandbox Attempt" in handoff_text
    assert "Completed Core Refactor" in handoff_text

    # 2. Verification that CONTEXT.md applies the exact same filtering (no low-confidence inconclusive spam)
    assert "Low Confidence Inconclusive Task" not in context_text
    assert "High Confidence Inconclusive Task" in context_text
    assert "Active Work Task" in context_text

    # 3. Verification of complete, unfiltered preservation in context.json
    json_titles = [ep["title"] for ep in json_data["episodes"]]
    assert len(json_titles) == 5
    assert "Low Confidence Inconclusive Task" in json_titles
    assert "High Confidence Inconclusive Task" in json_titles
    assert "Active Work Task" in json_titles
    assert "Abandoned Sandbox Attempt" in json_titles
    assert "Completed Core Refactor" in json_titles

    # 4. Verification of complete, unfiltered preservation in SQLite database
    db_episodes = await db.get_session_episodes(session.id)
    db_titles = [ep.title for ep in db_episodes]
    assert len(db_titles) == 5
    assert "Low Confidence Inconclusive Task" in db_titles


def test_markdown_exporter_empty_episodes_fallback(tmp_path):
    """Verifies that MarkdownExporter renders an informative fallback when episodes list is empty."""
    from context_provider.export.markdown import MarkdownExporter
    from context_provider.storage.models import ProjectRecord, SessionRecord

    exporter = MarkdownExporter()
    project = ProjectRecord(id="p1", name="demo-proj", root_path=str(tmp_path))
    session = SessionRecord(id="s1", project_id="p1")

    handoff_md = exporter.render_handoff_md(
        project=project,
        session=session,
        episodes=[],
        failed_attempts=[],
        traps_avoided=[],
        contradictions=[],
    )

    assert "## 2. Chronological Episode Record" in handoff_md
    assert "- *No high-signal episodes recorded.*" in handoff_md


def test_markdown_exporter_format_files_modified_unit():
    """Unit test for format_files_modified helper method in MarkdownExporter."""
    from context_provider.export.markdown import MarkdownExporter

    exporter = MarkdownExporter()

    # 1. Zero files -> 'None'
    assert exporter.format_files_modified([]) == "None"

    # 2. 5 or fewer files -> full comma-separated list
    files_3 = ["src/a.py", "src/b.py", "src/c.py"]
    assert exporter.format_files_modified(files_3) == "src/a.py, src/b.py, src/c.py"

    files_5 = [f"src/f{i}.py" for i in range(1, 6)]
    assert exporter.format_files_modified(files_5) == "src/f1.py, src/f2.py, src/f3.py, src/f4.py, src/f5.py"

    # 3. More than 5 files -> first 5 + '...and X other files'
    files_8 = [f"src/f{i}.py" for i in range(1, 9)]
    expected_8 = "src/f1.py, src/f2.py, src/f3.py, src/f4.py, src/f5.py, ...and 3 other files"
    assert exporter.format_files_modified(files_8) == expected_8


def test_markdown_and_json_files_modified_contrast(tmp_path):
    """
    Verifies that HANDOFF.md truncates files modified to 5 + '...and X other files',
    while context.json preserves the full untruncated list.
    """
    import json
    from context_provider.export.markdown import MarkdownExporter
    from context_provider.export.json_export import JsonExporter
    from context_provider.storage.models import ProjectRecord, SessionRecord, EpisodeRecord, EpisodeStatus

    files_10 = [f"module/sub{i}.py" for i in range(1, 11)]  # 10 files
    episode = EpisodeRecord(
        id="ep-10",
        session_id="s1",
        title="Large Batch Refactor",
        status=EpisodeStatus.COMPLETED,
        files_touched=files_10,
    )

    project = ProjectRecord(id="p1", name="test-trunc-proj", root_path=str(tmp_path))
    session = SessionRecord(id="s1", project_id="p1")

    md_exporter = MarkdownExporter()
    handoff_md = md_exporter.render_handoff_md(
        project=project,
        session=session,
        episodes=[episode],
        failed_attempts=[],
        traps_avoided=[],
        contradictions=[],
    )

    # 1. Check Markdown truncation
    expected_md_files = (
        "- **Files Modified**: module/sub1.py, module/sub2.py, module/sub3.py, "
        "module/sub4.py, module/sub5.py, ...and 5 other files"
    )
    assert expected_md_files in handoff_md
    assert "module/sub6.py" not in handoff_md

    # 2. Check JSON export preserves all 10 untruncated files
    json_exporter = JsonExporter()
    json_raw = json_exporter.render_context_json(
        project=project,
        session=session,
        episodes=[episode],
        events=[],
        rules_findings=[],
        contradictions=[],
    )
    json_data = json.loads(json_raw)

    exported_files = json_data["episodes"][0]["files_touched"]
    assert len(exported_files) == 10
    assert exported_files == files_10



