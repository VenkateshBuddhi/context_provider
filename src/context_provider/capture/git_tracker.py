"""
Git Activity Tracker for capturing version control transitions.

Records commit messages, commit hashes, branch switches, diff statistics,
and detected git reverts / reset operations.
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from context_provider.storage.models import EventRecord, EventType


class GitTracker:
    """Interfaces with local Git repository to extract changes and history."""

    def __init__(self, workspace_root: Path, session_id: str):
        self.workspace_root = workspace_root.resolve()
        self.session_id = session_id
        self._last_head: Optional[str] = self.get_head_commit()

    def _run_git(self, args: List[str]) -> Optional[str]:
        """Runs a git command in the workspace root and returns stdout."""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return None

    def is_git_repo(self) -> bool:
        """Checks if current workspace is inside an initialized git repo."""
        res = self._run_git(["rev-parse", "--is-inside-work-tree"])
        return res == "true"

    def get_current_branch(self) -> Optional[str]:
        """Returns the current active git branch name."""
        return self._run_git(["branch", "--show-current"])

    def get_head_commit(self) -> Optional[str]:
        """Returns the full SHA-1 hash of HEAD."""
        return self._run_git(["rev-parse", "HEAD"])

    def get_diff_summary(self, staged: bool = False) -> str:
        """Returns the git diff stat summary."""
        args = ["diff", "--stat"]
        if staged:
            args.append("--staged")
        return self._run_git(args) or ""

    def get_status_short(self) -> str:
        """Returns the porcelain status overview."""
        return self._run_git(["status", "--short"]) or ""

    def capture_commit_event(self) -> Optional[EventRecord]:
        """
        Detects if HEAD has advanced and records a new commit event.
        Returns EventRecord if a new commit was detected.
        """
        current_head = self.get_head_commit()
        if not current_head or current_head == self._last_head:
            return None

        # Fetch commit details: %H (hash), %an (author), %s (subject), %b (body)
        log_out = self._run_git(
            ["log", "-1", "--pretty=format:%H%x00%an%x00%s%x00%b"]
        )
        if not log_out:
            self._last_head = current_head
            return None

        parts = log_out.split("\x00")
        commit_hash = parts[0] if len(parts) > 0 else current_head
        author = parts[1] if len(parts) > 1 else ""
        subject = parts[2] if len(parts) > 2 else ""
        body = parts[3] if len(parts) > 3 else ""

        # Check for revert patterns
        is_revert = subject.lower().startswith("revert") or "reverting" in subject.lower()

        diff_stat = self._run_git(["diff-tree", "--no-commit-id", "--name-status", "-r", commit_hash]) or ""

        event = EventRecord(
            session_id=self.session_id,
            event_type=EventType.GIT,
            payload={
                "action": "commit",
                "commit_hash": commit_hash,
                "author": author,
                "subject": subject,
                "body": body,
                "is_revert": is_revert,
                "branch": self.get_current_branch(),
                "diff_summary": diff_stat,
            },
            evidence_meta={
                "source": "git_cli",
                "previous_head": self._last_head,
            },
        )
        self._last_head = current_head
        return event

    def capture_snapshot(self) -> EventRecord:
        """Creates an immediate snapshot event of the current git state."""
        return EventRecord(
            session_id=self.session_id,
            event_type=EventType.GIT,
            payload={
                "action": "snapshot",
                "branch": self.get_current_branch(),
                "head": self.get_head_commit(),
                "status": self.get_status_short(),
                "unstaged_diff": self.get_diff_summary(staged=False),
                "staged_diff": self.get_diff_summary(staged=True),
            },
            evidence_meta={"source": "git_cli"},
        )

