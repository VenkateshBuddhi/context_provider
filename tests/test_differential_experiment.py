"""Unit tests for TaskSandbox, HeadlessAgentRunner, and DifferentialExperiment."""

import pytest
import shutil
import tempfile
from pathlib import Path
from context_provider.evaluation.sandbox import TaskSandbox, FAISS_BINARY_ERROR
from context_provider.evaluation.agent_runner import HeadlessAgentRunner
from context_provider.evaluation.differential import DifferentialExperiment


@pytest.fixture
def mock_workspace(tmp_path):
    """Creates a miniature mock workspace resembling pdf-rag-assistant."""
    backend_dir = tmp_path / "backend"
    backend_dir.mkdir()
    rag_file = backend_dir / "rag.py"
    rag_file.write_text("import faiss\nimport numpy as np\n# Vector retrieval logic\n", encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text("# PDF RAG Assistant\n", encoding="utf-8")
    return tmp_path


def test_task_sandbox_isolation_and_traps(mock_workspace):
    sandbox = TaskSandbox(mock_workspace, scenario_name="test_faiss_trap")

    # 1. Condition A (Blind): Handoff files must NOT be present
    sandbox_dir_a = sandbox.setup_sandbox(include_context_provider_files=False)
    assert (sandbox_dir_a / "backend" / "rag.py").exists()
    assert not (sandbox_dir_a / "HANDOFF.md").exists()
    assert not (sandbox_dir_a / "CONTEXT.md").exists()

    # 2. Trap execution
    code, out = sandbox.execute_sandbox_command("python -c \"import faiss\"")
    assert code == 1
    assert "Could not load FAISS C++ library" in out

    # Safe command execution
    code_safe, out_safe = sandbox.execute_sandbox_command("python -c \"print('safe')\"")
    assert code_safe == 0
    assert "safe" in out_safe

    sandbox.cleanup()
    assert not sandbox_dir_a.exists()

    # 3. Condition C: Handoff files MUST be present and contain trap
    sandbox_dir_c = sandbox.setup_sandbox(include_context_provider_files=True)
    handoff = sandbox_dir_c / "HANDOFF.md"
    assert handoff.exists()
    text = handoff.read_text(encoding="utf-8")
    assert "CRITICAL TRAPS TO AVOID" in text
    assert "DO NOT attempt to install or import FAISS" in text

    sandbox.cleanup()
    assert not sandbox_dir_c.exists()


def test_differential_agent_experiment(mock_workspace, monkeypatch):
    # Route _call_llm to fast deterministic agent action for instantaneous, reproducible unit testing
    monkeypatch.setattr(
        HeadlessAgentRunner,
        "_call_llm",
        lambda self, prompt: self._heuristic_agent_turn(prompt),
    )
    experiment = DifferentialExperiment(mock_workspace)
    results = experiment.run_experiment(max_turns=3)

    assert "run_a_blind" in results
    assert "run_c_context_provider" in results
    assert "differential_summary" in results

    diff = results["differential_summary"]
    # In Run A, the blind agent attempted FAISS
    assert diff["trap_repeated_in_run_a"] is True
    # In Run C, the agent reading HANDOFF.md avoided the trap
    assert diff["trap_avoided_in_run_c"] is True

    # Check Markdown report rendering
    md = experiment.render_markdown_report(results)
    assert "# Empirical Agent Differential Experiment: Blind vs Context Provider" in md
    assert "100% Trap Avoidance" in md
    assert "Turn-by-Turn Differential Trace" in md

