# Context Provider

> **Local, evidence-based context handoff system & Python SDK for autonomous AI coding agents.**  
> Eliminates context loss, prevents repeated failed approaches, and enables seamless cross-session task continuation using local SLM classification and LLM synthesis.

---

[![PyPI Version](https://img.shields.io/pypi/v/context-provider.svg?color=blue)](https://pypi.org/project/context-provider/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/your-username/context_provider/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/context_provider/actions)
[![Models](https://img.shields.io/badge/Models-LAYA%20ModernBERT%20%7C%20Ollama%20llama3.1-orange.svg)](https://ollama.ai)
[![Protocol](https://img.shields.io/badge/MCP-JSON--RPC%202.0%20Stdio-green.svg)](https://modelcontextprotocol.io/)
[![Tests](https://img.shields.io/badge/Tests-47%20Passed%20(100%25)-brightgreen.svg)](tests/)
[![Storage](https://img.shields.io/badge/Storage-SQLite%20WAL-003B57.svg?logo=sqlite&logoColor=white)](src/context_provider/storage/)

---

## Table of Contents

- [The Problem](#the-problem)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Python SDK Quickstart](#python-sdk-quickstart)
- [CLI Reference](#cli-reference)
  - [1. Initialize Context Tracking (`start`)](#1-initialize-context-tracking-start)
  - [2. Live Filesystem Watcher (`watch`)](#2-live-filesystem-watcher-watch)
  - [3. Run Commands with Outcome Tracking (`run`)](#3-run-commands-with-outcome-tracking-run)
  - [4. Record Developer Notes & Hypotheses (`note`)](#4-record-developer-notes--hypotheses-note)
  - [5. Check Session Status (`status`)](#5-check-session-status-status)
  - [6. Synthesize and Export Handoff Documents (`export`)](#6-synthesize-and-export-handoff-documents-export)
  - [7. Automated Shell Profile Hook (`hook` & `hook-log`)](#7-automated-shell-profile-hook-powershell)
  - [8. Run Differential Agent Experiment (`experiment`)](#8-run-differential-agent-experiment-experiment)
  - [9. Run the Evaluation Benchmark (`evaluate`)](#9-run-the-evaluation-benchmark-evaluate)
  - [10. Model Context Protocol Server (`mcp`)](#10-model-context-protocol-server-mcp)
- [Model Context Protocol (MCP) Integration](#model-context-protocol-mcp-integration)
- [Experimental Evaluation & Benchmarks](#experimental-evaluation--benchmarks)
- [Testing](#testing)
- [Project Structure Map](#project-structure-map)
- [License](#license)

---

## The Problem

When an AI coding agent completes an interaction, context is typically lost. An incoming agent reads repository files, but:
- **It does not know what was already attempted and abandoned**, repeating the exact same failed implementations, bad dependencies, or broken endpoints.
- **Manual `CONTEXT.md` files suffer from optimistic developer bias**: developers claim an approach works, but subsequent terminal commands or unit tests broke without documentation.
- **Context window saturation**: dumping raw terminal logs or entire git diff histories overflows context windows, inducing hallucination and degradation.

**Context Provider** runs as a local background observer that records workspace activity into an evidence-grounded transaction log, segments events into semantic episodes, flags contradictions using a local BERT classifier (**LAYA**), synthesizes concise handoff files with a local LLM (**llama3.1** via Ollama), and serves real-time context over the **Model Context Protocol (MCP)** and a programmatic **Python SDK**.

---

## System Architecture

```
===========================================================================================
             LOCAL CONTEXT HANDOFF SYSTEM — COMPONENT ARCHITECTURE
===========================================================================================

 [ TARGET DEVELOPER WORKSPACE ]
    │
    ├──► File System Edits ────────► [ watcher.py ] ──────────────────┐
    │                                   (Watchdog + .gitignore +      │
    │                                    Recursive Exclusion Filter)  │
    │                                                                 ▼
    ├──► Git Diffs & Commits ──────► [ git_tracker.py ] ──────────────► [ RAW EVENT LOG ]
    │                                   (Branch, HEAD, diff summary)  ▼
    ├──► Shell Profile Hook ───────► [ hooks.py ] ───────────────────► (JSON Payloads)
    │     (PowerShell prompt wrapper;    (Non-blocking hidden exec    │
    │      auto-intercepts cmd & exit)    via Start-Process -Hidden)  │
    │                                                                 │
    └──► Terminal Commands & Notes ─► [ terminal.py ] ────────────────┘
                                        (Manual wrapper fallback)     │
 ┌────────────────────────────────────────────────────────────────────┘
 │
 ▼
===========================================================================================
                             DATA ACQUISITION LAYER
===========================================================================================
 │
 ▼
 [ context_provider/storage/db.py ] ──► Async Write Engine (aiosqlite + WAL Mode)
 │
 └──► SQLite Storage (`~/.local-context/context_store.db`)
       ├──► projects  (Workspace ID, root directory path)
       ├──► sessions  (Active session timelines & metadata)
       ├──► episodes  (Logical clusters of related development work)
       └──► events    (Chronological transaction log: git, file, cmd, notes)
 │
 ┌────────────────────────────────────────────────────────────────────┘
 │
 ▼
===========================================================================================
                         HYBRID PIPELINE & MEMORY LAYER
===========================================================================================
 │
 ├──► [ Episode Builder ] (pipeline/episode_builder.py)
 │       4-Tier Clustering: Temporal idle gap, hard boundaries, Jaccard file overlap.
 │       Validation Filter: Requires >= 3 events OR >= 1 high-signal event (fails, commits, notes).
 │       Deferred Buffer: Incomplete clusters remain buffered for subsequent merging.
 │       State tags: [Active, Completed, Abandoned, Inconclusive].
 │
 ├──► [ Deterministic Rules Engine ] (pipeline/rules_engine.py)
 │       Detects: Git Reverts, Command Thrashing (>=2 fails), Test Regressions, Pivots.
 │
 ├──► [ LAYA Engine — System One BERT ] (analysis/laya_engine.py) (~395M params)
 │       Evaluates: Schema-safe contradiction detection between notes and evidence.
 │       Uses: ModernBERT-large encoder embeddings with normalized cosine alignment.
 │       Fallback: Robust rule-based scoring if neural weights are not present.
 │
 └───────► [ Synthesis Gateway — Ollama LLM ] (synthesis/gateway.py) (llama3.1:latest)
                 Consumes: Ultra-compressed, high-signal event logs.
                 Traceback Bottom-Tail: Preserves fatal exception lines (up to 400 chars).
                 Boilerplate Sanitization: Regex stripping of conversational filler.
                 Synthesizes: Grounded episode summaries, active focus, and traps warnings.
 │
 ┌────────────────────────────────────────────────────────────────────┘
 │
 ▼
===========================================================================================
                       EXPORT & AGENT INTERFACE LAYER
===========================================================================================
 │
 ├──► Programmatic Python SDK (context_provider/sdk.py)
 │       ├──► Client: `ContextProvider` (async & sync API for agent loops)
 │       └──► Full typing support with `py.typed` (PEP 561)
 │
 ├──► File Export Engine (export/exporter.py)
 │       ├──► `CONTEXT.md`   (Current developer focus, recent episodes, next steps)
 │       ├──► `HANDOFF.md`   (Pre-compiled traps to avoid, contradictions, evidence)
 │       ├──► `context.json` (Structured JSON representation with schema v1.0.0)
 │       ├── Truncation: File modification lists > 5 items formatted cleanly.
 │       └── Filtering: Inconclusive (<0.5 confidence) episodes hidden from MD.
 │
 ├──► MCP Server Interface (mcp/server.py)
 │       ├──► JSON-RPC 2.0 stdio server for Claude Desktop, Cursor, Antigravity
 │       ├──► Tool: `get_active_context`
 │       ├──► Tool: `get_handoff_dossier`
 │       └──► Tool: `record_agent_note`
 │
 ├──► Differential Experiment Sandbox (evaluation/differential.py)
 │       ├── Sandbox: Isolated workspace copies (sandbox_root/run_*)
 │       └── Multi-Turn Agent Runner: Head-to-head empirical testing (Blind vs Context)
 │
 └──► Evaluation Harness (evaluation/harness.py)
         Benchmark: Condition A (Repo-Only) vs Condition B (Manual) vs Condition C (Auto)
===========================================================================================
```

---

## Key Features

- **Programmatic Python SDK**:
  - Full async and sync client (`ContextProvider`) for seamless integration into autonomous agent frameworks (LangChain, AutoGen, CrewAI, LlamaIndex, custom loops).
  - PEP 561 compliant with `py.typed` for IDE intellisense and static type checking.

- **Automated Shell Profile Hook (PowerShell)**:
  - Automatically captures every executed command, exit code, execution duration, and `$Error[0]` fatal error strings directly from your terminal.
  - Spawns background logging via `Start-Process -WindowStyle Hidden`, ensuring **zero terminal latency** and **zero window flashing**.
  - One-click non-destructive install and uninstall via `context-provider hook install`.

- **Infinite Loop Prevention (Watcher Exclusion Filter)**:
  - Hardcoded pre-queue exclusion filter in `watcher.py` that silently drops generated context artifacts (`CONTEXT.md`, `HANDOFF.md`, `context.json`, `*.db*`, `evaluation.*`, `differential_experiment.*`) before they enter the SQLite event queue, preventing recursive self-observation loops.

- **4-Tier Semantic Episode Builder with Threshold Filtering**:
  - Automatically segments continuous event streams into logical episodes using temporal idle gaps (default 10 min), hard boundaries (commits, branch changes, goal notes), and file overlap (Jaccard similarity).
  - **Minimum Threshold Validation**: Enforces that a cluster must contain $\ge 3$ events OR at least 1 high-signal event (terminal crash, git commit, developer note) before finalizing; otherwise, events are held in an active buffer for deferred merging.
  - Infers evidence-backed statuses: `Active`, `Completed`, `Abandoned`, or `Inconclusive`.

- **Hybrid Memory Extraction**:
  - **Deterministic Rules Engine**: Flags command thrashing (repeated consecutive failures), test regressions (tests breaking after previously passing), and pivot declarations.
  - **LAYA BERT Classifier**: Evaluates contradiction pairs between developer hypotheses and subsequent evidence using `convaiinnovations/laya` (ModernBERT-large backbone) with deterministic schema safety.
  - **Staleness Decay**: Linear freshness scoring across elapsed session hours.

- **Local LLM Synthesis (llama3.1)**:
  - Generates technical markdown summaries for `CONTEXT.md` and `HANDOFF.md` via Ollama.
  - **Traceback Bottom-Tail Preservation**: Captures the fatal bottom lines of crash stack traces (e.g. `ReferenceError`, `TypeError`, `ImportError`) up to 400 characters, falling back to `stdout` for test runners like `pytest`.
  - **Conversational Boilerplate Sanitization**: Regex post-processing strips conversational filler (e.g., *"Based on the provided information, here are 3 bullet points..."*) to maintain pure technical bullet lists.

- **Clean Markdown Export with File Truncation**:
  - Automatically truncates file modification lists: $\le 5$ files are rendered as a comma-separated list; $> 5$ files render the first 5 followed by `...and X other files`. Full file arrays remain 100% intact in `context.json` and SQLite.
  - Low-confidence (`< 0.5`) inconclusive episodes are filtered out of markdown handoff files to eliminate noise.

- **Model Context Protocol (MCP) Server**:
  - Native JSON-RPC 2.0 over stdio compatible with Claude Desktop, Cursor, and any MCP client.
  - Exposes tools for retrieving active context, reading handoff dossiers, and recording agent decisions in real time.

- **Empirical Differential Experiment Sandbox**:
  - Programmatic sandbox runner creating isolated git workspaces to benchmark autonomous agents head-to-head under Blind (Condition A) vs Context-Assisted (Condition C) modes.

---

## Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Modern type annotations, async/await runtime, robust datetime parsing |
| **Package Manager** | `uv` / `pip` | High-speed dependency management and PyPI packaging |
| **Storage Engine** | SQLite + `aiosqlite` | Asynchronous I/O with Write-Ahead Logging (WAL) mode |
| **File Observation** | `watchdog` | Cross-platform filesystem event listener with debouncing & recursive exclusion |
| **Shell Integration** | PowerShell 5.1 / 7+ | Automated prompt wrapper with hidden background execution |
| **Local SLM (System 1)** | `convaiinnovations/laya` | ~395M parameter ModernBERT-large backbone for contradiction scoring |
| **Local LLM (System 2)** | `llama3.1:latest` (Ollama) | 8B parameter generative synthesis for summaries and traps dossiers |
| **Protocol** | Model Context Protocol (MCP) | JSON-RPC 2.0 stdio transport (Spec 2024-11-05) |
| **Test Suite** | `pytest` + `pytest-asyncio` | Full unit and integration coverage (47 passed tests, 100%) |

---

## Installation

### From PyPI
```bash
pip install context-provider
```

With optional extras:
```bash
# With Model Context Protocol (MCP) support
pip install "context-provider[mcp]"

# With local neural ML models (PyTorch + Transformers)
pip install "context-provider[ml]"

# With all optional capabilities
pip install "context-provider[all]"
```

### From Source (for local development)
```bash
git clone https://github.com/your-username/context_provider.git
cd context_provider

# Install dependencies into virtual environment using uv
uv sync --extra dev --extra ml --extra mcp
```

Verify installation:
```bash
context-provider --help
```

---

## Python SDK Quickstart

Integrate Context Provider directly into your Python agent loops, scripts, or pipelines:

### Async Usage

```python
import asyncio
from context_provider import ContextProvider

async def main():
    # Initialize the client for a target workspace
    async with ContextProvider(workspace_path=".") as client:
        # Record an architectural decision or hypothesis
        await client.record_note(
            "Switched from polling to WebSocket subscriptions",
            category="decision"
        )

        # Log a command result with exit code and error tail
        await client.log_command(
            command="npm test",
            exit_code=1,
            stderr="AssertionError: Expected 200 OK but received 401 Unauthorized"
        )

        # Query active context for prompt injection
        context = await client.get_active_context()
        print(f"Project: {context['project']}, Recent notes: {context['recent_notes']}")

        # Synthesize and export CONTEXT.md, HANDOFF.md, and context.json
        artifacts = await client.export()
        print("Generated handoff dossiers:", artifacts)

if __name__ == "__main__":
    asyncio.run(main())
```

### Sync Usage

```python
from context_provider import ContextProvider

client = ContextProvider(workspace_path=".")
client.initialize_sync()

# Record notes and log outcomes synchronously
client.record_note_sync("Investigating Redis connection leak", category="bug")
ctx = client.get_active_context_sync()
```

---

## CLI Reference

All commands support the `-p / --path <DIRECTORY>` flag to monitor external target projects. You can use either `context` or `context-provider` as the executable name.

### 1. Initialize Context Tracking (`start`)
Registers the target project in SQLite and initializes a tracking session:
```bash
context-provider start -p "C:\path\to\target-project"
```
*Options:*
- `-w, --watch`: Starts the background filesystem watcher immediately.
- `-f, --force`: Forces creation of a new session even if one is active.

### 2. Live Filesystem Watcher (`watch`)
Continuously monitors file changes and Git commits in real time:
```bash
context-provider watch -p "C:\path\to\target-project"
```
*Options:*
- `-d, --duration <SECONDS>`: Runs watcher for N seconds (ideal for automation/tests).

### 3. Run Commands with Outcome Tracking (`run`)
Executes a terminal command and logs exit codes, duration, and stdout/stderr tails:
```bash
context-provider run -p "C:\path\to\target-project" pytest backend/
context-provider run -p "C:\path\to\target-project" npm test
```

### 4. Record Developer Notes & Hypotheses (`note`)
Records intent, architecture decisions, or bug observations into the session stream:
```bash
context-provider note -p "C:\path\to\target-project" -c decision "Switched from sync streaming to async generator"
context-provider note -p "C:\path\to\target-project" -c bug "FastAPI StreamingResponse yields TypeError on subscript"
```
*Categories:* `goal`, `decision`, `bug`, `hypothesis`, `observation`.

### 5. Check Session Status (`status`)
Displays the active project, current session ID, start time, and recent event log:
```bash
context-provider status -p "C:\path\to\target-project"
```

### 6. Synthesize and Export Handoff Documents (`export`)
Triggers episode clustering, LAYA contradiction evaluation, llama3.1 LLM synthesis, and writes three documents into the target project root:
```bash
context-provider export -p "C:\path\to\target-project"
```
Generated artifacts in the target workspace:
- `CONTEXT.md`: Current developer status, active focus, and prioritized next steps.
- `HANDOFF.md`: Traps to avoid, failed/abandoned attempts, contradictions, and empirical evidence.
- `context.json`: Full machine-readable representation with schema version `1.0.0`.

### 7. Automated Shell Profile Hook (PowerShell)
Replaces manual `context run` commands by automatically capturing every terminal command and exit code:
```bash
# Check hook installation status
context-provider hook status

# Safely install prompt wrapper into PowerShell $PROFILE
context-provider hook install

# Inspect generated non-blocking PowerShell wrapper script
context-provider hook script

# Uninstall hook from profile
context-provider hook uninstall
```
*Note: Uses `Start-Process -WindowStyle Hidden` to execute the background logging endpoint (`hook-log`) with zero operational latency and zero window flashing.*

### 8. Run Differential Agent Experiment (`experiment`)
Executes side-by-side empirical agent runs in isolated sandboxes (Condition A: Blind vs Condition C: With Context Provider):
```bash
context-provider experiment -p "C:\path\to\target-project" --max-turns 3
```

### 9. Run the Evaluation Benchmark (`evaluate`)
Executes comparative cross-session continuation evaluation across all three conditions:
```bash
context-provider evaluate -p "C:\path\to\target-project"
```
Produces `EVALUATION.md` and `evaluation.json` in the target project root.

### 10. Model Context Protocol Server (`mcp`)
Starts the native JSON-RPC 2.0 stdio server:
```bash
context-provider mcp -p "C:\path\to\target-project"
```

---

## Model Context Protocol (MCP) Integration

Context Provider includes a native MCP server operating over `stdio` using JSON-RPC 2.0. This allows AI coding agents in **Claude Desktop**, **Cursor**, or **Antigravity** to dynamically query context tools.

### Running the MCP Server
```bash
context-provider mcp -p "C:\path\to\target-project"
```

### Tool Definitions Provided

| Tool Name | Parameters | Description |
| :--- | :--- | :--- |
| `get_active_context` | *(None)* | Retrieves current active project goals, developer notes, recent episodes, git status, and freshness metrics. |
| `get_handoff_dossier` | *(None)* | Retrieves comprehensive handoff dossier including failed attempts, traps to avoid, LAYA contradictions, and rendered `HANDOFF.md`. |
| `record_agent_note` | `note` (str), `category` (str) | Records an incoming agent's decision, observation, hypothesis, or error into the workspace's local context stream. |

### Client Configuration (Claude Desktop / Cursor)

Add Context Provider to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "context-provider": {
      "command": "context-provider",
      "args": [
        "mcp",
        "-p",
        "C:\\path\\to\\target-project"
      ]
    }
  }
}
```

---

## Experimental Evaluation & Benchmarks

The system was evaluated against `pdf-rag-assistant` (a FastAPI + React Retrieval-Augmented Generation application) across three experimental conditions:

1. **Condition A: Repo-Only** — The incoming agent inspects raw repository files and Git history with no handoff documents.
2. **Condition B: Manual Context** — Standard human-maintained `CONTEXT.md` containing developer notes without empirical test or command logs.
3. **Condition C: Auto-Generated (Context Provider)** — Model-synthesized `CONTEXT.md`, `HANDOFF.md`, and `context.json` produced by LAYA and llama3.1.

### Quantitative Benchmark Results

| Metric | Condition A: Repo-Only | Condition B: Manual Context | Condition C: Auto-Generated (Ours) |
| :--- | :---: | :---: | :---: |
| **Trap Avoidance Rate** | 0.0% | 100.0% | **100.0%** |
| **Contradiction Resolution** | 0.0% | 0.0% | **100.0%** |
| **Evidence Grounding Score** | 0.0% | 100.0% | **100.0%** |
| **Resume Readiness Score** | 20.0% | 55.0% | **95.0%** |
| **Composite Continuation Score** | **4.0%** | **66.0%** | **99.0%** |

---

## Testing

The project maintains a comprehensive, 100% green test suite covering all four phases.

### Running Tests
```bash
# Run all unit and integration tests
uv run pytest

# Run tests with verbose output
uv run pytest -v
```

### Test Coverage Breakdown (47 Passed, 100%)

```
============================= 47 passed in 140s =============================
tests/test_capture.py ................ Command runner, Git tracker snapshots, diff summaries, .gitignore filtering (4 tests)
tests/test_differential_experiment.py  Isolated workspace sandboxing, blind vs context differential evaluation (2 tests)
tests/test_episode_builder.py ........ 4-tier clustering, threshold validation, deferred buffer merging, ISO timestamps (12 tests)
tests/test_hooks.py .................. PowerShell profile detection, safe wrapper appending/removal, hidden exec (6 tests)
tests/test_live_watcher.py ........... Recursive exclusion list, thread-safe SQLite persistence queue, debouncing (3 tests)
tests/test_phase3_memory.py .......... Rules engine, LAYA ModernBERT, staleness decay, summary sanitization, tail retention (16 tests)
tests/test_phase4_mcp_eval.py ........ MCP JSON-RPC 2.0 stdio tools, 3-condition continuation evaluation benchmark (2 tests)
tests/test_storage.py ................ SQLite schema initialization, WAL mode, transaction integrity (2 tests)
```

---

## Project Structure Map

```
context_provider/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Automated multi-python (3.10, 3.11, 3.12) test matrix & twine check
│       └── publish.yml             # Automated PyPI release workflow (OIDC Trusted Publishing)
├── main.py                         # CLI entrypoint forwarding to context_provider.cli.commands
├── pyproject.toml                  # PEP 621 build system, metadata, dependencies & CLI scripts
├── README.md                       # Architectural documentation, SDK quickstart & reference
├── LICENSE                         # MIT License
├── uv.lock                         # Deterministic dependency lockfile
├── src/
│   └── context_provider/           # Standard PyPA src-layout package
│       ├── __init__.py             # Public SDK exports (__version__, ContextProvider, etc.)
│       ├── py.typed                # PEP 561 inline typing marker
│       ├── sdk.py                  # High-level async/sync Python SDK client
│       ├── config.py               # Global settings: SQLite paths, model URLs, timing thresholds
│       ├── analysis/
│       │   ├── confidence.py       # Evidence-weighted episode confidence scoring [0.1 - 1.0]
│       │   ├── detector.py         # Contradiction and staleness detector orchestration
│       │   └── laya_engine.py      # LAYA BERT interface (ModernBERT cosine alignment + fallback)
│       ├── capture/
│       │   ├── git_tracker.py      # Git status, commit snapshots, diff summaries, revert checks
│       │   ├── terminal.py         # Subprocess command execution wrapper with exit code capture
│       │   └── watcher.py          # Watchdog filesystem listener with debounce and exclusion filter
│       ├── cli/
│       │   ├── commands.py         # CLI commands: start, watch, run, note, status, export, mcp, evaluate, experiment
│       │   └── hooks.py            # PowerShell $PROFILE hook installer, script generator & background logger
│       ├── evaluation/
│       │   ├── agent_runner.py     # Multi-turn mock agent runner for differential experiments
│       │   ├── differential.py     # Differential benchmark comparing Blind vs Context-assisted agents
│       │   ├── harness.py          # Benchmark runner for 3-condition cross-session continuation
│       │   ├── models.py           # Evaluation data models: HandoffPackage, BenchmarkMetrics, Report
│       │   └── sandbox.py          # Isolated directory sandbox manager for differential experiments
│       ├── export/
│       │   ├── exporter.py         # Master export pipeline orchestrating clustering, LAYA, and LLM
│       │   ├── json_export.py      # JSON schema v1.0.0 exporter
│       │   └── markdown.py         # Markdown template renderer with boilerplate sanitization & file truncation
│       ├── mcp/
│       │   └── server.py           # Model Context Protocol (MCP) JSON-RPC 2.0 stdio server
│       ├── pipeline/
│       │   ├── episode_builder.py  # 4-tier event clustering, threshold validation, and status inference
│       │   └── rules_engine.py     # Deterministic anomaly detection: thrashing, regressions, pivots
│       ├── storage/
│       │   ├── db.py               # Asynchronous SQLite storage engine with WAL mode and upsert logic
│       │   └── models.py           # Dataclasses and enums: Project, Session, Episode, Event
│       └── synthesis/
│           ├── gateway.py          # Ollama local LLM interface with event log compression & sanitization
│           └── prompts.py          # Llama 3.1 chat-formatted prompt templates
└── tests/
    ├── test_capture.py             # Tests for terminal capture, git tracker, and gitignore filter
    ├── test_differential_experiment.py # Tests for sandbox creation and differential agent evaluation
    ├── test_episode_builder.py     # Tests for clustering algorithms, thresholding, and episode state tagging
    ├── test_hooks.py               # Tests for PowerShell profile hooks and hidden background logging
    ├── test_live_watcher.py        # Tests for background watcher thread, queue persistence, and exclusions
    ├── test_phase3_memory.py       # Tests for rules engine, LAYA engine, summary sanitization, and traceback tails
    ├── test_phase4_mcp_eval.py     # Tests for MCP tools, JSON-RPC handling, and evaluation harness
    └── test_storage.py             # Tests for SQLite database CRUD and session lifecycle
```

---

## License

This project is licensed under the [MIT License](LICENSE).
