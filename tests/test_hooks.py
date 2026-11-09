"""Unit tests for the Automated Shell Profile Hook Framework (src/cli/hooks.py)."""

import pytest
import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from context_provider.cli.hooks import (
    log_shell_event,
    generate_powershell_hook_script,
    install_powershell_hook,
    uninstall_powershell_hook,
    is_hook_installed,
    HOOK_START_MARKER,
    HOOK_END_MARKER,
)
from context_provider.storage.db import DatabaseManager
from context_provider.storage.models import EventType


@pytest.fixture
def test_env(tmp_path):
    """Sets up a temporary SQLite database and project workspace."""
    db_file = tmp_path / "test_hook.db"
    project_dir = tmp_path / "sample_project"
    project_dir.mkdir()

    # Initialize tables
    conn = sqlite3.connect(str(db_file))
    conn.execute("""
        CREATE TABLE projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            root_path TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    conn.execute("""
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            metadata TEXT NOT NULL DEFAULT '{}'
        );
    """)
    conn.execute("""
        CREATE TABLE events (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            episode_id TEXT,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            payload TEXT NOT NULL DEFAULT '{}',
            evidence_meta TEXT NOT NULL DEFAULT '{}'
        );
    """)

    # Register project & active session
    proj_id = str(uuid.uuid4())
    sess_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()

    conn.execute(
        "INSERT INTO projects VALUES (?, ?, ?, ?, ?)",
        (proj_id, "sample_project", str(project_dir), now_iso, now_iso),
    )
    conn.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, NULL, 1, '{}')",
        (sess_id, proj_id, now_iso),
    )
    conn.commit()
    conn.close()

    return {"db_file": db_file, "project_dir": project_dir, "session_id": sess_id}


def test_log_shell_event_command(test_env):
    db_file = test_env["db_file"]
    project_dir = test_env["project_dir"]

    # 1. Log normal command
    success = log_shell_event("python app.py", exit_code=0, cwd_str=str(project_dir), db_path=db_file)
    assert success is True

    # Verify event stored in SQLite
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("SELECT event_type, payload, evidence_meta FROM events;")
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 1
    ev_type, payload_str, evidence_str = rows[0]
    assert ev_type == EventType.COMMAND.value
    payload = json.loads(payload_str)
    assert payload["command"] == "python app.py"
    assert payload["exit_code"] == 0
    assert payload["hook_source"] == "powershell_profile"


def test_log_shell_event_test_framework(test_env):
    db_file = test_env["db_file"]
    project_dir = test_env["project_dir"]

    # Log test command failure
    success = log_shell_event("pytest tests/test_rag.py", exit_code=1, cwd_str=str(project_dir), db_path=db_file)
    assert success is True

    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("SELECT event_type, payload, evidence_meta FROM events;")
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 1
    ev_type, payload_str, evidence_str = rows[0]
    # Automatically classified as TEST
    assert ev_type == EventType.TEST.value
    payload = json.loads(payload_str)
    assert payload["command"] == "pytest tests/test_rag.py"
    assert payload["exit_code"] == 1


def test_log_shell_event_with_error(test_env):
    db_file = test_env["db_file"]
    project_dir = test_env["project_dir"]

    # Log command with error message
    success = log_shell_event(
        "node server.js",
        exit_code=1,
        cwd_str=str(project_dir),
        db_path=db_file,
        error_message="ReferenceError: medicineModel is not defined",
    )
    assert success is True

    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("SELECT payload FROM events ORDER BY timestamp DESC LIMIT 1;")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    payload = json.loads(row[0])
    assert payload["command"] == "node server.js"
    assert payload["exit_code"] == 1
    assert payload["stderr_tail"] == "ReferenceError: medicineModel is not defined"


def test_log_shell_event_unmonitored_dir(test_env, tmp_path):
    db_file = test_env["db_file"]
    unmonitored = tmp_path / "outside_project"
    unmonitored.mkdir()

    # Commands outside registered projects must be ignored silently
    success = log_shell_event("dir", exit_code=0, cwd_str=str(unmonitored), db_path=db_file)
    assert success is False

    # Internal hook commands must be ignored
    success_internal = log_shell_event("python main.py hook-log --command test", exit_code=0, cwd_str=str(test_env["project_dir"]), db_path=db_file)
    assert success_internal is False


def test_powershell_hook_script_generation(tmp_path):
    script = generate_powershell_hook_script(tmp_path)
    assert HOOK_START_MARKER in script
    assert HOOK_END_MARKER in script
    assert "Start-Process" in script
    assert "-WindowStyle Hidden" in script
    assert "hook-log" in script


def test_powershell_hook_installation_lifecycle(tmp_path):
    fake_profile = tmp_path / "profile.ps1"
    fake_profile.write_text("# Existing profile\n$Host.UI.RawUI.WindowTitle = 'Dev'\n", encoding="utf-8")

    assert not is_hook_installed(fake_profile)

    # 1. Install hook
    success, msg = install_powershell_hook(fake_profile, tmp_path)
    assert success is True
    assert is_hook_installed(fake_profile)

    content = fake_profile.read_text(encoding="utf-8")
    assert "Existing profile" in content
    assert HOOK_START_MARKER in content
    assert "-WindowStyle Hidden" in content

    # 2. Idempotency test (installing again does not duplicate)
    success_dup, msg_dup = install_powershell_hook(fake_profile, tmp_path)
    assert success_dup is True
    assert "already installed" in msg_dup
    content_dup = fake_profile.read_text(encoding="utf-8")
    assert content_dup.count(HOOK_START_MARKER) == 1

    # 3. Uninstall hook
    success_un, msg_un = uninstall_powershell_hook(fake_profile)
    assert success_un is True
    assert not is_hook_installed(fake_profile)

    content_un = fake_profile.read_text(encoding="utf-8")
    assert "Existing profile" in content_un
    assert HOOK_START_MARKER not in content_un
