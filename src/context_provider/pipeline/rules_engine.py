"""
Phase 3: Deterministic Rules Engine.

Scans chronological event sequences for deterministic behavioral signatures:
- Git reverts and rollbacks (abandoned solutions)
- Repeated failed commands and test errors (command thrashing)
- Test regression loops (passing suite followed by breaking changes)
- High churn / thrashing on specific files
- Abandoned hypotheses (failed attempt followed by pivot note)
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from context_provider.storage.models import EventRecord, EventType


class RulesEngine:
    """Evaluates rule-based heuristics on chronological event streams."""

    def analyze_patterns(self, events: List[EventRecord]) -> List[Dict[str, Any]]:
        """Extracts detected patterns and anomalies from the event log."""
        insights = []
        insights.extend(self._detect_git_reverts(events))
        insights.extend(self._detect_repeated_command_failures(events))
        insights.extend(self._detect_test_regressions(events))
        insights.extend(self._detect_failed_attempt_pivots(events))
        insights.extend(self._detect_file_thrashing(events))
        return insights

    def _detect_git_reverts(self, events: List[EventRecord]) -> List[Dict[str, Any]]:
        """Finds git revert actions indicating abandoned attempts."""
        findings = []
        for ev in events:
            if ev.event_type == EventType.GIT and ev.payload.get("is_revert"):
                findings.append({
                    "pattern": "GIT_REVERT_DETECTED",
                    "severity": "HIGH",
                    "description": f"Approach abandoned via commit: {ev.payload.get('subject')}",
                    "evidence_event_id": ev.id,
                    "timestamp": ev.timestamp,
                    "details": {
                        "commit_hash": ev.payload.get("commit_hash"),
                        "branch": ev.payload.get("branch"),
                    },
                })
        return findings

    def _detect_repeated_command_failures(self, events: List[EventRecord]) -> List[Dict[str, Any]]:
        """Detects consecutive non-zero exit codes on the same command."""
        findings = []
        cmd_events = [e for e in events if e.event_type in (EventType.COMMAND, EventType.TEST)]

        consecutive_failures = 0
        last_failed_cmd = None
        first_fail_ev = None

        for ev in cmd_events:
            exit_code = ev.payload.get("exit_code", 0)
            cmd = ev.payload.get("command", "")

            if exit_code != 0:
                if cmd == last_failed_cmd:
                    consecutive_failures += 1
                else:
                    consecutive_failures = 1
                    last_failed_cmd = cmd
                    first_fail_ev = ev

                if consecutive_failures == 2:
                    findings.append({
                        "pattern": "COMMAND_THRASHING",
                        "severity": "MEDIUM",
                        "description": f"Command failed {consecutive_failures} consecutive times: '{cmd}'",
                        "evidence_event_id": ev.id,
                        "timestamp": ev.timestamp,
                        "details": {
                            "command": cmd,
                            "failure_count": consecutive_failures,
                            "last_error": ev.payload.get("stderr_tail", "")[:300],
                        },
                    })
                elif consecutive_failures > 2 and findings and findings[-1]["pattern"] == "COMMAND_THRASHING":
                    findings[-1]["severity"] = "HIGH"
                    findings[-1]["description"] = f"Command failed {consecutive_failures} consecutive times: '{cmd}'"
                    findings[-1]["evidence_event_id"] = ev.id
                    findings[-1]["timestamp"] = ev.timestamp
                    findings[-1]["details"]["failure_count"] = consecutive_failures
                    findings[-1]["details"]["last_error"] = ev.payload.get("stderr_tail", "")[:300]
            else:
                consecutive_failures = 0
                last_failed_cmd = None
                first_fail_ev = None

        return findings

    def _detect_test_regressions(self, events: List[EventRecord]) -> List[Dict[str, Any]]:
        """Detects regressions where a previously passing test suite breaks."""
        findings = []
        test_events = [e for e in events if e.event_type == EventType.TEST]

        had_passed = False
        last_pass_ts = None

        for ev in test_events:
            exit_code = ev.payload.get("exit_code", 0)
            if exit_code == 0:
                had_passed = True
                last_pass_ts = ev.timestamp
            elif had_passed and exit_code != 0:
                findings.append({
                    "pattern": "TEST_REGRESSION",
                    "severity": "HIGH",
                    "description": f"Test suite broke at {ev.timestamp} after passing at {last_pass_ts}",
                    "evidence_event_id": ev.id,
                    "timestamp": ev.timestamp,
                    "details": {
                        "command": ev.payload.get("command"),
                        "last_passing_time": last_pass_ts,
                    },
                })
                had_passed = False  # Reset until passes again

        return findings

    def _detect_failed_attempt_pivots(self, events: List[EventRecord]) -> List[Dict[str, Any]]:
        """Detects when a command fails and a developer note follows declaring a pivot or revert."""
        findings = []
        for i in range(len(events) - 1):
            curr = events[i]
            nxt = events[i + 1]

            if curr.event_type in (EventType.COMMAND, EventType.TEST) and curr.payload.get("exit_code", 0) != 0:
                if nxt.event_type == EventType.NOTE:
                    note_text = nxt.payload.get("note", "").lower()
                    if any(kw in note_text for kw in ["revert", "abandon", "pivot", "instead", "failed", "trap", "decision:"]):
                        findings.append({
                            "pattern": "FAILED_ATTEMPT_PIVOT",
                            "severity": "HIGH",
                            "description": f"Failed execution of '{curr.payload.get('command')}' prompted pivot: \"{nxt.payload.get('note')}\"",
                            "evidence_event_id": nxt.id,
                            "timestamp": nxt.timestamp,
                            "details": {
                                "failed_command": curr.payload.get("command"),
                                "stderr_sample": curr.payload.get("stderr_tail", "")[:200],
                                "decision_note": nxt.payload.get("note"),
                            },
                        })
        return findings

    def _detect_file_thrashing(self, events: List[EventRecord]) -> List[Dict[str, Any]]:
        """Detects files that are repeatedly modified or deleted in short spans."""
        findings = []
        file_events = [e for e in events if e.event_type == EventType.FILE]
        touch_counts: Dict[str, int] = {}

        for ev in file_events:
            path = ev.payload.get("relative_path") or ev.payload.get("file_path")
            if path:
                touch_counts[path] = touch_counts.get(path, 0) + 1

        for path, count in touch_counts.items():
            if count >= 10:
                findings.append({
                    "pattern": "HIGH_FILE_CHURN",
                    "severity": "LOW",
                    "description": f"High churn detected on file: {path} ({count} operations)",
                    "file": path,
                    "count": count,
                })

        return findings
