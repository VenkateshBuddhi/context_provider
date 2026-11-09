"""
Evidence-Based Confidence Scoring Engine.

Calculates confidence values [0.0 - 1.0] for recorded decisions, solutions,
and episode outcomes based on supporting empirical evidence.
"""

from typing import List, Dict, Any
from context_provider.storage.models import EventRecord, EventType, EpisodeStatus


class ConfidenceScorer:
    """Computes empirical confidence ratings based on test evidence, git verification, and notes."""

    # Base weight values
    WEIGHT_PASSING_TEST = 0.40
    WEIGHT_CLEAN_COMMIT = 0.30
    WEIGHT_DEVELOPER_NOTE = 0.20
    WEIGHT_NO_ERRORS = 0.10

    def compute_episode_confidence(self, events: List[EventRecord], status: EpisodeStatus) -> float:
        """
        Calculates a normalized confidence score for an episode.
        Higher scores indicate strong empirical proof (e.g. green test suites, committed diffs).
        """
        if not events:
            return 0.0

        if status == EpisodeStatus.ABANDONED:
            return 0.15

        score = 0.0

        # Check for passing test executions
        test_events = [e for e in events if e.event_type == EventType.TEST]
        if test_events:
            last_test = test_events[-1]
            if last_test.payload.get("exit_code") == 0:
                score += self.WEIGHT_PASSING_TEST
            else:
                score -= 0.20  # Penalize failing final tests

        # Check for git commits
        git_commits = [
            e for e in events
            if e.event_type == EventType.GIT and e.payload.get("action") == "commit" and not e.payload.get("is_revert")
        ]
        if git_commits:
            score += self.WEIGHT_CLEAN_COMMIT

        # Check for developer annotations
        notes = [e for e in events if e.event_type == EventType.NOTE]
        if notes:
            score += self.WEIGHT_DEVELOPER_NOTE

        # Check for clean command runs without errors
        cmd_errors = [
            e for e in events
            if e.event_type == EventType.COMMAND and e.payload.get("exit_code") != 0
        ]
        if not cmd_errors:
            score += self.WEIGHT_NO_ERRORS

        return round(max(0.1, min(1.0, score)), 2)

