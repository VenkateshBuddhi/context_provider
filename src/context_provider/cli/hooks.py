"""
Automated Shell Profile Hook Framework for Context Provider.

Provides:
1. Fast background endpoint (`hook-log`) to write command strings and exit codes
   directly into SQLite without terminal lag or blocking.
2. PowerShell profile installation and detection utilities for Windows.
3. Safe prompt wrapper generation using Start-Process -WindowStyle Hidden.
"""

import os
import sys
import uuid
import sqlite3
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

from context_provider.config import default_config
from context_provider.storage.models import EventType


HOOK_START_MARKER = "# >>> context-provider-hook >>>"
HOOK_END_MARKER = "# <<< context-provider-hook <<<"


def log_shell_event(
    command: str,
    exit_code: int,
    cwd_str: str,
    db_path: Optional[Path] = None,
    error_message: str = "",
) -> bool:
    """
    Synchronous, ultra-fast SQLite logging for background shell hooks.
    Returns True if logged to a monitored project session, False otherwise.
    """
    clean_cmd = (command or "").strip()
    if not clean_cmd:
        return False

    # Ignore internal hook commands to prevent infinite feedback loops
    if "hook-log" in clean_cmd or "context-provider-hook" in clean_cmd:
        return False

    cwd = Path(cwd_str).resolve()
    target_db = db_path or default_config.db_path

    if not target_db.exists():
        return False

    conn = sqlite3.connect(str(target_db), timeout=5.0)
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        cursor = conn.cursor()

        # Find registered project matching or containing this working directory
        cursor.execute("SELECT id, root_path FROM projects;")
        projects = cursor.fetchall()
        matching_project_id = None

        for proj_id, root_path_str in projects:
            root_p = Path(root_path_str).resolve()
            if cwd == root_p or (str(cwd).lower().startswith(str(root_p).lower())):
                matching_project_id = proj_id
                break

        # If user executed command outside any tracked project, ignore silently
        if not matching_project_id:
            return False

        # Find active session for this project
        cursor.execute(
            "SELECT id FROM sessions WHERE project_id = ? AND is_active = 1 ORDER BY started_at DESC LIMIT 1;",
            (matching_project_id,),
        )
        row = cursor.fetchone()

        if row:
            session_id = row[0]
        else:
            # Auto-create active session if none exists for this registered project
            session_id = str(uuid.uuid4())
            now_iso = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                """
                INSERT INTO sessions (id, project_id, started_at, is_active, metadata)
                VALUES (?, ?, ?, 1, ?);
                """,
                (session_id, matching_project_id, now_iso, json.dumps({"auto_created_by_hook": True})),
            )

        # Detect test vs command
        cmd_lower = clean_cmd.lower()
        is_test = any(kw in cmd_lower for kw in ["pytest", "unittest", "npm test", "jest", "cargo test", "go test"])
        event_type = EventType.TEST.value if is_test else EventType.COMMAND.value

        event_id = str(uuid.uuid4())
        now_ts = datetime.now(timezone.utc).isoformat()
        payload = {
            "command": clean_cmd,
            "exit_code": int(exit_code),
            "cwd": str(cwd),
            "hook_source": "powershell_profile",
            "is_test": is_test,
            "stderr_tail": error_message.strip() if error_message else "",
        }
        evidence_meta = {
            "success": int(exit_code) == 0,
            "working_dir": str(cwd),
        }

        cursor.execute(
            """
            INSERT INTO events (id, session_id, event_type, timestamp, payload, evidence_meta)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                event_id,
                session_id,
                event_type,
                now_ts,
                json.dumps(payload),
                json.dumps(evidence_meta),
            ),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def detect_powershell_profile() -> Optional[Path]:
    """Detects active PowerShell $PROFILE location on Windows."""
    # 1. Try querying PowerShell directly
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Write-Output $PROFILE"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0:
            out_path = res.stdout.strip()
            if out_path:
                return Path(out_path)
    except Exception:
        pass

    # 2. Fallback to standard Windows PowerShell profile paths
    user_docs = Path.home() / "Documents"
    candidate_1 = user_docs / "WindowsPowerShell" / "Microsoft.PowerShell_profile.ps1"
    candidate_2 = user_docs / "PowerShell" / "Microsoft.PowerShell_profile.ps1"

    if candidate_1.parent.exists():
        return candidate_1
    if candidate_2.parent.exists():
        return candidate_2
    return candidate_1


def generate_powershell_hook_script(context_provider_root: Path) -> str:
    """
    Generates non-blocking PowerShell tracking snippet using
    Start-Process -WindowStyle Hidden for zero operational latency or window flashing.
    """
    root_str = str(context_provider_root.resolve()).replace("\\", "\\\\")

    return f"""{HOOK_START_MARKER}
# Context Provider — Automated Command Tracking Hook (PowerShell)
# Spawns background logger with zero operational latency and zero window flashing.
if (-not (Test-Path variable:global:__ctx_hook_active)) {{
    $global:__ctx_hook_active = $true
    $global:__ctx_last_cmd = ""
    $global:__ctx_root = "{root_str}"

    if (Test-Path function:prompt) {{
        $global:__ctx_orig_prompt = $function:prompt
    }} else {{
        $global:__ctx_orig_prompt = {{ "PS $($executionContext.SessionState.Path.CurrentLocation)$('>' * ($nestedPromptLevel + 1)) " }}
    }}

    function prompt {{
        $lastExit = $LASTEXITCODE
        $lastHist = Get-History -Count 1 -ErrorAction SilentlyContinue
        if ($lastHist -and ($lastHist.CommandLine -ne $global:__ctx_last_cmd)) {{
            $global:__ctx_last_cmd = $lastHist.CommandLine
            $cText = $lastHist.CommandLine.Trim()
            if ($cText -and -not ($cText.Contains("hook-log")) -and -not ($cText.Contains("context-provider-hook"))) {{
                $errSnippet = ""
                if ($lastExit -ne 0 -and $Error.Count -gt 0) {{
                    $errSnippet = ($Error[0].ToString() -replace '["\r\n`]', ' ').Trim()
                    if ($errSnippet.Length -gt 300) {{ $errSnippet = $errSnippet.Substring(0, 300) }}
                }}
                $argStr = "run --directory `"$($global:__ctx_root)`" python main.py hook-log --command `"$cText`" --exit-code $lastExit --cwd `"$PWD`" --error `"$errSnippet`""
                Start-Process -FilePath "uv" -ArgumentList $argStr -WindowStyle Hidden
            }}
        }}
        & $global:__ctx_orig_prompt
    }}
}}
{HOOK_END_MARKER}"""


def is_hook_installed(profile_path: Path) -> bool:
    """Checks whether the hook snippet is already present in the profile script."""
    if not profile_path.exists():
        return False
    try:
        content = profile_path.read_text(encoding="utf-8")
        return HOOK_START_MARKER in content
    except Exception:
        return False


def install_powershell_hook(profile_path: Optional[Path] = None, context_provider_root: Optional[Path] = None) -> Tuple[bool, str]:
    """Safely appends our custom tracking wrapper to the PowerShell $PROFILE script."""
    prof = profile_path or detect_powershell_profile()
    if not prof:
        return False, "Could not determine PowerShell $PROFILE path."

    root_dir = context_provider_root or Path(__file__).resolve().parent.parent.parent

    # Ensure parent directory exists
    prof.parent.mkdir(parents=True, exist_ok=True)

    current_content = ""
    if prof.exists():
        try:
            current_content = prof.read_text(encoding="utf-8")
        except Exception:
            current_content = ""

    if HOOK_START_MARKER in current_content:
        return True, f"Hook is already installed in {prof}"

    snippet = generate_powershell_hook_script(root_dir)
    separator = "\n\n" if current_content and not current_content.endswith("\n") else "\n"
    new_content = current_content + separator + snippet + "\n"

    prof.write_text(new_content, encoding="utf-8")
    return True, f"Hook successfully installed into {prof}"


def uninstall_powershell_hook(profile_path: Optional[Path] = None) -> Tuple[bool, str]:
    """Removes the tracking wrapper from the PowerShell $PROFILE script."""
    prof = profile_path or detect_powershell_profile()
    if not prof or not prof.exists():
        return False, "Profile does not exist."

    content = prof.read_text(encoding="utf-8")
    if HOOK_START_MARKER not in content:
        return True, f"Hook was not found in {prof}"

    s_idx = content.find(HOOK_START_MARKER)
    e_idx = content.find(HOOK_END_MARKER)
    if s_idx != -1 and e_idx != -1:
        e_idx += len(HOOK_END_MARKER)
        updated = content[:s_idx].rstrip() + "\n" + content[e_idx:].lstrip()
        prof.write_text(updated.strip() + "\n", encoding="utf-8")
        return True, f"Hook removed from {prof}"

    return False, "Malformed hook markers in profile."
