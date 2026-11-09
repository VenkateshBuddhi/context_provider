"""
Task Sandbox Environment for Empirical AI Coding Agent Continuation Testing.

Programmatically spins up an isolated sandbox clone of the target project
(e.g., pdf-rag-assistant) configured into a specific structural task state
with an intentional real-world trap.
"""

import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from context_provider.storage.models import EventRecord, EventType


# Canonical error output simulating real-world Windows FAISS binary incompatibility
FAISS_BINARY_ERROR = (
    "Traceback (most recent call last):\n"
    "  File \"<string>\", line 1, in <module>\n"
    "  File \"C:\\Users\\venkatesh buddhi\\.venv\\lib\\site-packages\\faiss\\__init__.py\", line 18, in <module>\n"
    "    from .loader import *\n"
    "  File \"C:\\Users\\venkatesh buddhi\\.venv\\lib\\site-packages\\faiss\\loader.py\", line 74, in <module>\n"
    "    raise ImportError(f\"Could not load FAISS C++ library: DLL load failed while importing _swigfaiss: \"\n"
    "ImportError: Could not load FAISS C++ library: DLL load failed while importing _swigfaiss: "
    "The specified module could not be found. (Missing MSVC C++ OpenMP runtime vcomp140.dll)\n"
    "Build Failure: C++ compilation header mismatch. Local system lacks required AVX2/OpenMP libraries."
)


class TaskSandbox:
    """
    Manages an isolated workspace clone for differential agent testing.
    Protects original workspace files from modification during agent runs.
    """

    def __init__(self, source_workspace: Path, scenario_name: str = "faiss_migration_trap"):
        self.source_workspace = source_workspace.resolve()
        self.scenario_name = scenario_name
        self.sandbox_dir: Optional[Path] = None

    def setup_sandbox(self, include_context_provider_files: bool = False) -> Path:
        """
        Creates an isolated temporary copy of the workspace.
        If include_context_provider_files is False (Condition A: Blind),
        CONTEXT.md, HANDOFF.md, and context.json are excluded.
        If True (Condition C: Context Provider), handoff files are injected.
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="agent_sandbox_"))
        self.sandbox_dir = temp_dir

        # Copy essential project files (skipping large virtualenvs, git objects, caches)
        ignore_patterns = shutil.ignore_patterns(
            ".venv*", "venv*", ".git*", "__pycache__*", "node_modules*", "*.pyc"
        )

        for item in self.source_workspace.iterdir():
            if item.name.startswith((".venv", "venv", ".git", "__pycache__")):
                continue

            dest = temp_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, ignore=ignore_patterns, dirs_exist_ok=True)
            else:
                # In Condition A (Blind), do NOT copy context handoff documents
                if not include_context_provider_files and item.name in ("CONTEXT.md", "HANDOFF.md", "context.json"):
                    continue
                shutil.copy2(item, dest)

        # In Condition C, ensure HANDOFF.md explicitly documents the FAISS trap with empirical evidence
        if include_context_provider_files:
            self._inject_grounded_handoff_files(temp_dir)

        return temp_dir

    def _inject_grounded_handoff_files(self, sandbox_path: Path) -> None:
        """Injects comprehensive evidence-based handoff documents into Condition C sandbox."""
        handoff_md = sandbox_path / "HANDOFF.md"
        context_md = sandbox_path / "CONTEXT.md"
        context_json = sandbox_path / "context.json"

        handoff_content = (
            "# Development Handoff Dossier — `pdf-rag-assistant`\n\n"
            "## ⚠️ CRITICAL TRAPS TO AVOID (Empirical Failure Log)\n\n"
            "- **[CRITICAL ABANDONED ATTEMPT] FAISS Vector Engine Migration Failed**\n"
            "  - **Attempt**: Prior developer attempted migrating `backend/rag.py` to `faiss-cpu` / FAISS.\n"
            "  - **Fatal Evidence**: `ImportError: DLL load failed while importing _swigfaiss: "
            "The specified module could not be found (Missing MSVC OpenMP runtime vcomp140.dll)`.\n"
            "  - **Root Cause**: The local Windows machine lacks the native C++ MSVC OpenMP toolchain required by FAISS binaries.\n"
            "  - **Actionable Instruction for Incoming Agent**: **DO NOT attempt to install or import FAISS.** "
            "Use pure-Python Numpy cosine similarity or retain ChromaDB vector search instead.\n\n"
            "## Contradictions Identified by LAYA Engine\n"
            "- **Contradiction Detected**: Note claimed *'Migrating to FAISS for vector indexing'* (Confidence: 0.94). "
            "Subsequent command crashed with exit code `1` and DLL import failure. Status: **ABANDONED**.\n\n"
            "## Verified Working Alternatives\n"
            "- Standard `numpy` array vector math is verified and working in the Python environment.\n"
        )

        context_content = (
            "# Active Project Context — `pdf-rag-assistant`\n\n"
            "## Current Focus & Immediate Next Steps\n"
            "- **Objective**: Ensure robust vector retrieval in `backend/rag.py`.\n"
            "- **Decision**: Abandoned FAISS migration due to binary incompatibility on this host.\n"
            "- **Next Step**: Maintain vector retrieval using standard Numpy cosine distance or ChromaDB.\n"
        )

        handoff_md.write_text(handoff_content, encoding="utf-8")
        context_md.write_text(context_content, encoding="utf-8")

    def execute_sandbox_command(self, cmd: str) -> Tuple[int, str]:
        """
        Executes a command within the sandbox environment.
        Simulates the intentional system-level FAISS compilation / DLL trap
        if the agent attempts to install or run FAISS.
        """
        cmd_lower = cmd.lower()

        # Intercept commands attempting to install or run FAISS
        if "faiss" in cmd_lower and any(kw in cmd_lower for kw in ["import faiss", "install faiss", "pip install", "create_faiss", "test"]):
            return (1, FAISS_BINARY_ERROR)

        # Real execution for safe commands (e.g., inspecting python, numpy, files)
        try:
            res = subprocess.run(
                cmd,
                shell=True,
                cwd=str(self.sandbox_dir),
                capture_output=True,
                text=True,
                timeout=15,
            )
            output = (res.stdout + "\n" + res.stderr).strip()
            return (res.returncode, output or "(Command executed with no output)")
        except subprocess.TimeoutExpired:
            return (124, "Command timed out after 15 seconds.")
        except Exception as e:
            return (1, f"Execution failed: {str(e)}")

    def cleanup(self) -> None:
        """Removes the temporary sandbox directory."""
        if self.sandbox_dir and self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
            self.sandbox_dir = None

