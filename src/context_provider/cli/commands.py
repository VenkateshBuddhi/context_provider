"""
CLI Interface for the Local Context Handoff System.

Provides commands:
- `start`: Initializes project registration and starts an active tracking session.
- `run`: Executes a shell command, measures duration, exit codes, and test outcomes.
- `note`: Records a developer decision, hypothesis, or failure reason directly into the log.
- `status`: Displays current project tracking information, active session, and recent events.
- `export`: Generates CONTEXT.md, HANDOFF.md, and context.json for AI agent consumption.
"""

import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import Optional

from context_provider.storage.db import DatabaseManager
from context_provider.storage.models import EventRecord, EventType
from context_provider.capture.terminal import CommandRunner
from context_provider.capture.git_tracker import GitTracker


async def _get_current_session(db: DatabaseManager, project_path: Path):
    project = await db.get_or_create_project(str(project_path))
    session = await db.get_active_session(project.id)
    return project, session


def _get_cwd(args: argparse.Namespace) -> Path:
    target = getattr(args, "path", None)
    return Path(target or Path.cwd()).resolve()


def cmd_start(args: argparse.Namespace) -> int:
    """Initializes project and creates a new active session."""
    cwd = _get_cwd(args)
    db = DatabaseManager()

    async def _run():
        await db.initialize()
        project = await db.get_or_create_project(str(cwd), name=cwd.name)
        existing_session = await db.get_active_session(project.id)
        if existing_session and not args.force:
            print(f"[context] Existing active session found: {existing_session.id}")
            print(f"[context] Started at: {existing_session.started_at}")
            return existing_session.id

        session = await db.create_session(
            project.id,
            metadata={"cli_initiated": True, "cwd": str(cwd)},
        )
        print(f"[context] New session started: {session.id}")
        print(f"[context] Project: {project.name} ({project.root_path})")

        # Capture initial git snapshot
        git = GitTracker(cwd, session.id)
        if git.is_git_repo():
            snapshot = git.capture_snapshot()
            await db.log_event(snapshot)
            print(f"[context] Captured initial Git snapshot (branch: {git.get_current_branch()})")

        return session.id

    session_id = asyncio.run(_run())

    # Detect Windows PowerShell and provide automated hook setup
    if sys.platform == "win32":
        from context_provider.cli.hooks import detect_powershell_profile, is_hook_installed
        prof = detect_powershell_profile()
        if prof and is_hook_installed(prof):
            print("[context] Automated terminal tracking hook: ACTIVE in PowerShell $PROFILE")
        else:
            print("\n--------------------------------------------------------------------------------")
            print("[context] PowerShell on Windows detected!")
            print("To automatically track all terminal commands & tests with ZERO manual typing:")
            print("   Run: uv run python main.py hook install")
            print("This non-blocking wrapper runs via Start-Process -WindowStyle Hidden (zero latency).")
            print("--------------------------------------------------------------------------------\n")

    if getattr(args, "watch", False):
        return cmd_watch(args)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Runs a developer command and logs execution outcome."""
    cwd = _get_cwd(args)
    db = DatabaseManager()
    full_cmd = " ".join(args.command)

    async def _run():
        await db.initialize()
        project, session = await _get_current_session(db, cwd)
        if not session:
            # Auto-create session if none active
            project = await db.get_or_create_project(str(cwd))
            session = await db.create_session(project.id, metadata={"auto_created": True})
            print(f"[context] Created new session {session.id}")

        runner = CommandRunner(cwd, session.id)
        exit_code, record = runner.run_command(full_cmd)
        await db.log_event(record)

        # Print output to terminal as expected
        if record.payload.get("stdout_tail"):
            print(record.payload["stdout_tail"], end="")
        if record.payload.get("stderr_tail"):
            print(record.payload["stderr_tail"], file=sys.stderr, end="")

        print(f"\n[context] Command executed with exit code {exit_code} (logged to session {session.id})")
        return exit_code

    return asyncio.run(_run())


def cmd_note(args: argparse.Namespace) -> int:
    """Records a developer note or decision into the active session."""
    cwd = _get_cwd(args)
    db = DatabaseManager()
    note_text = " ".join(args.text)

    async def _run():
        await db.initialize()
        project, session = await _get_current_session(db, cwd)
        if not session:
            project = await db.get_or_create_project(str(cwd))
            session = await db.create_session(project.id, metadata={"auto_created": True})

        event = EventRecord(
            session_id=session.id,
            event_type=EventType.NOTE,
            payload={
                "note": note_text,
                "category": args.category or "observation",
            },
            evidence_meta={"source": "cli_note", "author": "developer"},
        )
        await db.log_event(event)
        print(f"[context] Note recorded in session {session.id}: \"{note_text}\"")

    asyncio.run(_run())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Displays tracking status for the current directory."""
    cwd = _get_cwd(args)
    db = DatabaseManager()

    async def _run():
        await db.initialize()
        project, session = await _get_current_session(db, cwd)
        print("==================================================")
        print(" LOCAL CONTEXT HANDOFF SYSTEM - STATUS")
        print("==================================================")
        print(f"Database Path : {db.db_path}")
        print(f"Workspace     : {cwd}")
        if not project or not session:
            print("Status        : NO ACTIVE SESSION")
            print("Run 'context start' to begin tracking this directory.")
            return

        print(f"Project ID    : {project.id}")
        print(f"Session ID    : {session.id}")
        print(f"Session Start : {session.started_at}")

        events = await db.get_session_events(session.id)
        print(f"Total Events  : {len(events)}")
        print("--------------------------------------------------")
        print("Recent Events (Last 5):")
        for ev in events[-5:]:
            action = ev.payload.get("action") or ev.payload.get("command") or ev.payload.get("note", "")
            print(f" - [{ev.timestamp[11:19]}] {ev.event_type.value.upper()}: {str(action)[:50]}")
        print("==================================================")

    asyncio.run(_run())
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Runs the live capture service in the foreground, recording file edits and git events."""
    import time
    from context_provider.capture.watcher import LiveCaptureService

    cwd = _get_cwd(args)
    db = DatabaseManager()

    async def _init_session():
        await db.initialize()
        project, session = await _get_current_session(db, cwd)
        if not session:
            project = await db.get_or_create_project(str(cwd))
            session = await db.create_session(project.id, metadata={"cli_initiated": True, "cwd": str(cwd)})
        return project, session

    project, session = asyncio.run(_init_session())

    def _print_live_event(rec: EventRecord) -> None:
        action = rec.payload.get("action") or rec.payload.get("event") or ""
        path = rec.payload.get("relative_path") or rec.payload.get("path") or ""
        if rec.event_type == EventType.GIT:
            commit = rec.payload.get("commit_sha", "")[:7]
            msg = rec.payload.get("message", "")
            print(f"[{rec.timestamp[11:19]}] -> Captured GIT commit {commit}: {msg[:50]}")
        elif rec.event_type == EventType.FILE:
            print(f"[{rec.timestamp[11:19]}] -> Captured FILE {action}: {path}")
        elif rec.event_type in (EventType.COMMAND, EventType.TEST):
            cmd = rec.payload.get("command", "")
            exit_code = rec.payload.get("exit_code", 0)
            status_tag = f"(exit {exit_code})" if exit_code == 0 else f"(FAILED exit {exit_code})"
            print(f"[{rec.timestamp[11:19]}] -> Captured {rec.event_type.value.upper()} {status_tag}: {cmd[:60]}")
        else:
            print(f"[{rec.timestamp[11:19]}] -> Captured {rec.event_type.value.upper()}: {action}")

    service = LiveCaptureService(cwd, session.id, db, on_event_callback=_print_live_event)
    service.start()

    print("==================================================")
    print(" LOCAL CONTEXT HANDOFF SYSTEM - LIVE WATCHER")
    print("==================================================")
    print(f"Workspace     : {cwd}")
    print(f"Project       : {project.name}")
    print(f"Session ID    : {session.id}")
    print("Monitoring workspace file system & Git commits in real time...")
    print("Press Ctrl+C to stop watching.")
    print("==================================================")

    try:
        duration = getattr(args, "duration", None)
        start_t = time.time()
        while True:
            time.sleep(0.5)
            if duration and (time.time() - start_t) >= duration:
                break
    except KeyboardInterrupt:
        print("\n[context] Stopping live watcher...")
    finally:
        service.stop()
        print("[context] Live watcher stopped. All events flushed to SQLite.")

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Exports session context to Markdown and JSON handoff files."""
    from context_provider.export.exporter import ExportEngine

    cwd = _get_cwd(args)
    db = DatabaseManager()

    async def _run():
        await db.initialize()
        project, session = await _get_current_session(db, cwd)
        if not session:
            print("[context] No active session found to export.")
            return

        engine = ExportEngine(db, cwd)
        files = await engine.export_all(session.id)
        print("[context] Export completed:")
        for name, path in files.items():
            print(f" - {name}: {path}")

    asyncio.run(_run())
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    """Runs the MCP server over stdio for AI coding agent integration."""
    from context_provider.mcp.server import ContextMCPServer

    cwd = _get_cwd(args)
    db = DatabaseManager()

    async def _run():
        await db.initialize()
        server = ContextMCPServer(db, cwd)
        await server.run_stdio_server()

    asyncio.run(_run())
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Runs cross-session task continuation benchmarks across 3 handoff conditions."""
    from context_provider.evaluation.harness import EvaluationHarness

    cwd = _get_cwd(args)
    db = DatabaseManager()

    async def _run():
        await db.initialize()
        project, session = await _get_current_session(db, cwd)
        if not session:
            print("[context] No active session found to evaluate.")
            return

        harness = EvaluationHarness(db, cwd)
        report = await harness.run_evaluation(session.id)
        md_text = harness.render_markdown_report(report)

        report_md_path = cwd / "EVALUATION.md"
        report_json_path = cwd / "evaluation.json"

        report_md_path.write_text(md_text, encoding="utf-8")
        report_json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

        print(md_text)
        print(f"\n[context] Evaluation report saved:")
        print(f" - Markdown: {report_md_path}")
        print(f" - JSON: {report_json_path}")

    asyncio.run(_run())
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    """Runs a live differential agent experiment (Blind vs Context Provider) in an isolated sandbox."""
    from context_provider.evaluation.differential import DifferentialExperiment

    cwd = _get_cwd(args)
    print("==================================================")
    print(" EMPIRICAL AGENT DIFFERENTIAL EXPERIMENT")
    print("==================================================")
    print(f"Target Workspace: {cwd}")
    print("Scenario        : FAISS Binary Incompatibility & Vector Engine Migration Trap")
    print("Spinning up isolated sandboxes (Run A: Blind vs Run C: Context Provider)...")

    experiment = DifferentialExperiment(cwd)
    data = experiment.run_experiment(max_turns=args.max_turns)
    md_report = experiment.render_markdown_report(data)

    report_path = cwd / "DIFFERENTIAL_EXPERIMENT.md"
    json_path = cwd / "differential_experiment.json"

    report_path.write_text(md_report, encoding="utf-8")
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    try:
        print("\n" + md_report)
    except UnicodeEncodeError:
        print("\n" + md_report.encode("ascii", errors="replace").decode("ascii"))
    print(f"\n[context] Differential experiment reports saved:")
    print(f" - Markdown: {report_path}")
    print(f" - JSON: {json_path}")
    return 0


def cmd_hook_log(args: argparse.Namespace) -> int:
    """Fast background endpoint to log executed command and exit code into SQLite."""
    from context_provider.cli.hooks import log_shell_event
    import os

    cmd_str = getattr(args, "command", "")
    exit_code = getattr(args, "exit_code", 0)
    cwd_str = getattr(args, "cwd", os.getcwd())
    error_str = getattr(args, "error", "")
    log_shell_event(cmd_str, exit_code, cwd_str, error_message=error_str)
    return 0


def cmd_hook(args: argparse.Namespace) -> int:
    """Manages the automated shell profile hook (status, install, uninstall, script)."""
    from context_provider.cli.hooks import (
        detect_powershell_profile,
        is_hook_installed,
        install_powershell_hook,
        uninstall_powershell_hook,
        generate_powershell_hook_script,
    )
    action = getattr(args, "action", "status")
    prof = detect_powershell_profile()
    cli_dir = Path(__file__).resolve().parent.parent.parent

    if action == "status":
        print("==================================================")
        print(" SHELL PROFILE HOOK STATUS")
        print("==================================================")
        print(f"Platform          : {sys.platform}")
        print(f"PowerShell Profile: {prof or 'Not Found'}")
        if prof and prof.exists() and is_hook_installed(prof):
            print("Hook Status       : INSTALLED & ACTIVE")
        else:
            print("Hook Status       : NOT INSTALLED")
            print("\nTo install automatically, run:")
            print("   uv run python main.py hook install")
        print("==================================================")
        return 0

    elif action == "install":
        success, msg = install_powershell_hook(prof, cli_dir)
        print(f"[context] {msg}")
        if success:
            print("[context] Please restart your PowerShell session or run:")
            print(f"   . \"{prof}\"")
        return 0 if success else 1

    elif action == "uninstall":
        success, msg = uninstall_powershell_hook(prof)
        print(f"[context] {msg}")
        return 0 if success else 1

    elif action == "script":
        snippet = generate_powershell_hook_script(cli_dir)
        print(snippet)
        return 0

    return 0


def main():
    """Main CLI entry point for the `context` command."""
    parser = argparse.ArgumentParser(
        prog="context",
        description="Local Evidence-Based Context Handoff System for AI Coding Agents",
    )
    parser.add_argument("-p", "--path", type=str, default=None, help="Root path of target project (defaults to CWD)")
    subparsers = parser.add_subparsers(dest="subcommand", help="Available commands")

    # context start
    p_start = subparsers.add_parser("start", help="Start context tracking for the workspace")
    p_start.add_argument("-p", "--path", type=str, default=None, help="Root path of the project")
    p_start.add_argument("-f", "--force", action="store_true", help="Force starting a new session")
    p_start.add_argument("-w", "--watch", action="store_true", help="Start live filesystem watcher immediately")
    p_start.set_defaults(func=cmd_start)

    # context watch
    p_watch = subparsers.add_parser("watch", help="Continuously monitor file edits and Git commits in real time")
    p_watch.add_argument("-p", "--path", type=str, default=None, help="Root path of the project")
    p_watch.add_argument("-d", "--duration", type=float, default=None, help="Run watcher for N seconds (useful for tests)")
    p_watch.set_defaults(func=cmd_watch)

    # context run
    p_run = subparsers.add_parser("run", help="Run a command and track exit code/output")
    p_run.add_argument("-p", "--path", type=str, default=None, help="Root path of the project")
    p_run.add_argument("command", nargs=argparse.REMAINDER, help="Shell command to execute")
    p_run.set_defaults(func=cmd_run)

    # context note
    p_note = subparsers.add_parser("note", help="Record a developer note or hypothesis")
    p_note.add_argument("-p", "--path", type=str, default=None, help="Root path of the project")
    p_note.add_argument("-c", "--category", default="observation", help="Category (decision, bug, hypothesis)")
    p_note.add_argument("text", nargs="+", help="Note text to record")
    p_note.set_defaults(func=cmd_note)

    # context status
    p_status = subparsers.add_parser("status", help="Show current context tracking status")
    p_status.add_argument("-p", "--path", type=str, default=None, help="Root path of the project")
    p_status.set_defaults(func=cmd_status)

    # context export
    p_export = subparsers.add_parser("export", help="Generate CONTEXT.md, HANDOFF.md, and context.json")
    p_export.add_argument("-p", "--path", type=str, default=None, help="Root path of the project")
    p_export.set_defaults(func=cmd_export)

    # context mcp
    p_mcp = subparsers.add_parser("mcp", help="Run MCP (Model Context Protocol) server over stdio")
    p_mcp.add_argument("-p", "--path", type=str, default=None, help="Root path of the project")
    p_mcp.set_defaults(func=cmd_mcp)

    # context evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run cross-session continuation evaluation benchmark")
    p_eval.add_argument("-p", "--path", type=str, default=None, help="Root path of the project")
    p_eval.set_defaults(func=cmd_evaluate)

    # context experiment
    p_exp = subparsers.add_parser("experiment", help="Run differential agent experiment (Blind vs Context Provider)")
    p_exp.add_argument("-p", "--path", type=str, default=None, help="Root path of the project")
    p_exp.add_argument("--max-turns", type=int, default=5, help="Maximum number of turns per agent run (default: 5)")
    p_exp.set_defaults(func=cmd_experiment)

    # context hook-log (internal fast background logger)
    p_hlog = subparsers.add_parser("hook-log", help="Fast background endpoint to log command from shell hook")
    p_hlog.add_argument("--command", type=str, required=True, help="Command line executed")
    p_hlog.add_argument("--exit-code", type=int, default=0, help="Process return code")
    p_hlog.add_argument("--cwd", type=str, default=".", help="Working directory")
    p_hlog.add_argument("--error", type=str, default="", help="Error message or stderr tail if command failed")
    p_hlog.set_defaults(func=cmd_hook_log)

    # context hook
    p_hook = subparsers.add_parser("hook", help="Manage automated shell profile tracking hook")
    p_hook.add_argument("action", nargs="?", default="status", choices=["status", "install", "uninstall", "script"], help="Hook action")
    p_hook.set_defaults(func=cmd_hook)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()

