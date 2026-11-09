"""Unit and integration tests for Phase 4: MCP Server and Evaluation Harness."""

import json
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta
from context_provider.storage.db import DatabaseManager
from context_provider.storage.models import EventRecord, EventType
from context_provider.mcp.server import ContextMCPServer
from context_provider.evaluation.harness import EvaluationHarness
from context_provider.evaluation.models import EvaluationCondition


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_mcp.db"
    return DatabaseManager(db_file)


@pytest.mark.asyncio
async def test_mcp_server_tools_and_rpc(tmp_path, temp_db):
    await temp_db.initialize()
    project = await temp_db.get_or_create_project(str(tmp_path), name="test-mcp-proj")
    session = await temp_db.create_session(project.id)

    server = ContextMCPServer(temp_db, tmp_path)

    # 1. Check tool definitions
    tools = server.get_tool_definitions()
    assert len(tools) == 3
    tool_names = [t["name"] for t in tools]
    assert "get_active_context" in tool_names
    assert "get_handoff_dossier" in tool_names
    assert "record_agent_note" in tool_names

    # 2. Record agent note via tool call
    note_resp = await server.handle_tool_call(
        "record_agent_note",
        {"note": "Agent observed 404 error on auth callback", "category": "bug"},
    )
    assert note_resp["success"] is True
    assert note_resp["recorded_note"] == "Agent observed 404 error on auth callback"

    # 3. Retrieve active context via tool call
    ctx_resp = await server.handle_tool_call("get_active_context", {})
    assert ctx_resp["status"] == "active"
    assert ctx_resp["project"] == "test-mcp-proj"
    assert any("404 error" in n for n in ctx_resp["recent_notes"])

    # 4. JSON-RPC Protocol Handshake
    init_req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    init_resp = await server._handle_jsonrpc(init_req)
    assert init_resp["result"]["serverInfo"]["name"] == "context-provider-mcp"

    # 5. JSON-RPC Tools List
    list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    list_resp = await server._handle_jsonrpc(list_req)
    assert len(list_resp["result"]["tools"]) == 3

    # 6. JSON-RPC Tool Call
    call_req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "record_agent_note",
            "arguments": {"note": "Decided to use bearer token authentication", "category": "decision"},
        },
    }
    call_resp = await server._handle_jsonrpc(call_req)
    assert "content" in call_resp["result"]
    parsed_content = json.loads(call_resp["result"]["content"][0]["text"])
    assert parsed_content["success"] is True


@pytest.mark.asyncio
async def test_evaluation_harness_benchmarks(tmp_path, temp_db):
    await temp_db.initialize()
    project = await temp_db.get_or_create_project(str(tmp_path), name="eval-test-proj")
    session = await temp_db.create_session(project.id)

    base_t = datetime.now(timezone.utc)

    # Log events creating known ground-truth traps & contradictions:
    # 1. Note claiming feature works
    await temp_db.log_event(EventRecord(
        session_id=session.id,
        event_type=EventType.NOTE,
        timestamp=(base_t + timedelta(seconds=1)).isoformat(),
        payload={"note": "Fixed websocket streaming endpoint", "category": "decision"},
    ))

    # 2. Command failing twice (COMMAND_THRASHING trap)
    await temp_db.log_event(EventRecord(
        session_id=session.id,
        event_type=EventType.COMMAND,
        timestamp=(base_t + timedelta(seconds=5)).isoformat(),
        payload={"command": "pytest backend/test_stream.py", "exit_code": 1, "stderr_tail": "TypeError"},
    ))
    await temp_db.log_event(EventRecord(
        session_id=session.id,
        event_type=EventType.COMMAND,
        timestamp=(base_t + timedelta(seconds=10)).isoformat(),
        payload={"command": "pytest backend/test_stream.py", "exit_code": 1, "stderr_tail": "TypeError: StreamingResponse"},
    ))

    # 3. Note abandoning approach
    await temp_db.log_event(EventRecord(
        session_id=session.id,
        event_type=EventType.NOTE,
        timestamp=(base_t + timedelta(seconds=15)).isoformat(),
        payload={"note": "Revert: streaming approach failed due to library bug", "category": "decision"},
    ))

    # Run Evaluation
    harness = EvaluationHarness(temp_db, tmp_path)
    report = await harness.run_evaluation(session.id)

    assert report.project_name == "eval-test-proj"
    assert report.total_known_traps >= 1
    assert report.total_known_contradictions >= 1

    # Check comparative metrics
    m_repo = report.results[EvaluationCondition.REPO_ONLY.value]
    m_manual = report.results[EvaluationCondition.MANUAL_CONTEXT.value]
    m_auto = report.results[EvaluationCondition.AUTO_GENERATED.value]

    # Auto-generated context must achieve superior trap avoidance and composite continuation score
    assert m_auto.trap_avoidance_rate > m_repo.trap_avoidance_rate
    assert m_auto.contradiction_resolution_rate > m_manual.contradiction_resolution_rate
    assert m_auto.composite_continuation_score > m_repo.composite_continuation_score
    assert m_auto.composite_continuation_score > m_manual.composite_continuation_score

    # Check Markdown report generation
    md_output = harness.render_markdown_report(report)
    assert "# Context Provider — Task Continuation Benchmark Report" in md_output
    assert "Comparative Benchmark Matrix" in md_output
    assert "Trap Avoidance Rate" in md_output

