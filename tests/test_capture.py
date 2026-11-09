
"""Unit tests for the Capture layer (GitTracker, CommandRunner, FileWatcher filtering)."""

import pytest
from pathlib import Path
import tempfile
import shutil

from context_provider.capture.terminal import CommandRunner
from context_provider.capture.watcher import GitIgnoreFilter
from context_provider.capture.git_tracker import GitTracker
from context_provider.storage.models import EventType


def test_command_runner():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        runner = CommandRunner(temp_dir, session_id="test-session")
        exit_code, record = runner.run_command("python -c \"print('Hello stdout')\"")

        assert exit_code == 0
        assert record.event_type == EventType.COMMAND
        assert "Hello stdout" in record.payload["stdout_tail"]
        assert record.evidence_meta["success"] is True

        # Test failure with stderr
        exit_code_err, record_err = runner.run_command("python -c \"import sys; sys.stderr.write('Fatal crash'); sys.exit(1)\"")
        assert exit_code_err == 1
        assert "Fatal crash" in record_err.payload["stderr_tail"]

        # Test failure with stdout only (e.g. pytest pattern)
        exit_code_out, record_out = runner.run_command("python -c \"print('Pytest FAILED assertion'); import sys; sys.exit(1)\"")
        assert exit_code_out == 1
        # Effective stderr_tail must fall back to stdout to prevent losing the crash report
        assert "Pytest FAILED assertion" in record_out.payload["stderr_tail"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_gitignore_filter():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        (temp_dir / ".gitignore").write_text("*.secret\nbuild/\n", encoding="utf-8")
        filt = GitIgnoreFilter(temp_dir)

        assert filt.is_ignored(temp_dir / "test.secret") is True
        assert filt.is_ignored(temp_dir / ".git" / "HEAD") is True
        assert filt.is_ignored(temp_dir / ".venv" / "bin" / "python") is True
        assert filt.is_ignored(temp_dir / "src" / "main.py") is False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_system_artifact_exclusions():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        (temp_dir / ".gitignore").write_text("*.custom_ignore\n", encoding="utf-8")
        filt = GitIgnoreFilter(temp_dir)

        # 1. Core generated artifacts
        assert filt.is_ignored(temp_dir / "CONTEXT.md") is True
        assert filt.is_ignored(temp_dir / "HANDOFF.md") is True
        assert filt.is_ignored(temp_dir / "context.json") is True

        # Case-insensitivity
        assert filt.is_ignored(temp_dir / "context.md") is True
        assert filt.is_ignored(temp_dir / "Handoff.MD") is True

        # 2. SQLite database files
        assert filt.is_ignored(temp_dir / "context_store.db") is True
        assert filt.is_ignored(temp_dir / "context_store.db-wal") is True
        assert filt.is_ignored(temp_dir / "context_store.db-shm") is True
        assert filt.is_ignored(temp_dir / "app.db") is True

        # 3. Evaluation artifacts
        assert filt.is_ignored(temp_dir / "EVALUATION.md") is True
        assert filt.is_ignored(temp_dir / "evaluation.json") is True
        assert filt.is_ignored(temp_dir / "DIFFERENTIAL_EXPERIMENT.md") is True
        assert filt.is_ignored(temp_dir / "differential_experiment.json") is True

        # 4. Existing .gitignore patterns work alongside artifact exclusions
        assert filt.is_ignored(temp_dir / "my_secret.custom_ignore") is True

        # 5. Normal source code files are NOT ignored
        assert filt.is_ignored(temp_dir / "src" / "index.js") is False
        assert filt.is_ignored(temp_dir / "backend" / "app.py") is False
        assert filt.is_ignored(temp_dir / "notes.md") is False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_git_tracker_snapshot():
    # Run in current workspace which is a git repo or directory
    tracker = GitTracker(Path.cwd(), session_id="test-session")
    snapshot = tracker.capture_snapshot()
    assert snapshot.event_type == EventType.GIT
    assert "branch" in snapshot.payload

