"""
Terminal command execution wrapper and test output parser.

Executes arbitrary developer commands while intercepting return codes,
standard output, standard error, execution time, and test result signatures.
"""

import subprocess
import time
import re
from pathlib import Path
from typing import Dict, Any, Tuple
from context_provider.storage.models import EventRecord, EventType


class CommandRunner:
    """Runs shell/terminal commands and formats outcomes into structured EventRecords."""

    def __init__(self, workspace_root: Path, session_id: str):
        self.workspace_root = workspace_root.resolve()
        self.session_id = session_id

    def run_command(self, command_line: str) -> Tuple[int, EventRecord]:
        """
        Executes a command string synchronously, streaming output and capturing execution metrics.
        Returns (exit_code, EventRecord).
        """
        start_time = time.time()

        proc = subprocess.Popen(
            command_line,
            shell=True,
            cwd=str(self.workspace_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )

        stdout, stderr = proc.communicate()
        duration = time.time() - start_time
        exit_code = proc.returncode

        # If a command fails and stderr is empty, stdout typically contains the error/traceback (e.g. pytest, compilers)
        effective_stderr = stderr[-2000:] if (stderr and stderr.strip()) else (
            stdout[-2000:] if (exit_code != 0 and stdout and stdout.strip()) else ""
        )

        # Detect test patterns (pytest, unittest, jest, etc.)
        test_info = self._analyze_test_output(command_line, stdout, stderr, exit_code)
        event_type = EventType.TEST if test_info.get("is_test") else EventType.COMMAND

        record = EventRecord(
            session_id=self.session_id,
            event_type=event_type,
            payload={
                "command": command_line,
                "exit_code": exit_code,
                "stdout_tail": stdout[-2000:] if stdout else "",
                "stderr_tail": effective_stderr,
                "duration_seconds": round(duration, 3),
                "test_info": test_info,
            },
            evidence_meta={
                "success": exit_code == 0,
                "working_dir": str(self.workspace_root),
            },
        )

        return exit_code, record

    def _analyze_test_output(
        self, cmd: str, stdout: str, stderr: str, exit_code: int
    ) -> Dict[str, Any]:
        """Parses output to detect test framework executions and pass/fail statistics."""
        cmd_lower = cmd.lower()
        is_test = any(kw in cmd_lower for kw in ["pytest", "test", "unittest", "jest", "cargo test"])
        combined = (stdout or "") + "\n" + (stderr or "")

        passed = 0
        failed = 0
        errors = 0

        # Pytest pattern: 5 passed, 2 failed in 0.12s
        pytest_match = re.search(r"(\d+)\s+passed", combined)
        if pytest_match:
            passed = int(pytest_match.group(1))

        pytest_failed = re.search(r"(\d+)\s+failed", combined)
        if pytest_failed:
            failed = int(pytest_failed.group(1))

        pytest_err = re.search(r"(\d+)\s+error", combined)
        if pytest_err:
            errors = int(pytest_err.group(1))

        return {
            "is_test": is_test,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "overall_success": exit_code == 0,
        }

