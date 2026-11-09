"""
Phase 3: Detector Primitives for Contradiction and Stale/Conflict Detection.

Integrates LayaEngine (System One BERT) with deterministic staleness analysis
to produce 100% schema-safe output without LLM hallucinations.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from context_provider.storage.models import EventRecord, EpisodeRecord, EventType
from context_provider.analysis.laya_engine import LayaEngine


class ConflictDetector:
    """Detects contradictions, outdated hypotheses, and stale context records."""

    def __init__(self, stale_time_threshold_hours: float = 24.0):
        self.stale_time_threshold_hours = stale_time_threshold_hours
        self.laya = LayaEngine()

    def check_staleness(
        self, record_timestamp: str, current_timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Determines if a recorded decision or episode is stale based on elapsed time.
        Returns a schema-safe evaluation.
        """
        t_rec = datetime.fromisoformat(record_timestamp)
        t_curr = datetime.fromisoformat(current_timestamp) if current_timestamp else datetime.now(timezone.utc)
        elapsed_hours = max(0.0, (t_curr - t_rec).total_seconds() / 3600.0)

        is_stale = elapsed_hours > self.stale_time_threshold_hours
        freshness_score = max(0.0, min(1.0, 1.0 - (elapsed_hours / (self.stale_time_threshold_hours * 2))))

        return {
            "is_stale": is_stale,
            "elapsed_hours": round(elapsed_hours, 2),
            "freshness_score": round(freshness_score, 3),
        }

    def detect_contradictions(
        self,
        notes: List[EventRecord],
        subsequent_events: List[EventRecord],
    ) -> List[Dict[str, Any]]:
        """
        Evaluates candidate pairs between developer notes and subsequent events
        using LayaEngine to identify invalidated assumptions or broken hypotheses.
        """
        contradictions = []

        for note in notes:
            note_text = note.payload.get("note", "")
            note_time = datetime.fromisoformat(note.timestamp)

            for ev in subsequent_events:
                ev_time = datetime.fromisoformat(ev.timestamp)
                if ev_time <= note_time:
                    continue

                evidence_text = ""
                conflict_type = None

                # Pattern A: Test failure following positive assumption
                if ev.event_type == EventType.TEST and ev.payload.get("exit_code") != 0:
                    cmd = ev.payload.get("command", "")
                    err = ev.payload.get("stderr_tail", "")[:200]
                    evidence_text = f"Test command '{cmd}' failed with exit code {ev.payload.get('exit_code')}: {err}"
                    conflict_type = "TEST_OUTCOME_CONTRADICTION"

                # Pattern B: Command execution error
                elif ev.event_type == EventType.COMMAND and ev.payload.get("exit_code") != 0:
                    cmd = ev.payload.get("command", "")
                    err = ev.payload.get("stderr_tail", "")[:200]
                    evidence_text = f"Command '{cmd}' failed with error: {err}"
                    conflict_type = "COMMAND_FAILURE_CONTRADICTION"

                # Pattern C: Git revert
                elif ev.event_type == EventType.GIT and ev.payload.get("is_revert"):
                    evidence_text = f"Git revert commit executed: '{ev.payload.get('subject')}'"
                    conflict_type = "GIT_REVERT_CONTRADICTION"

                if not evidence_text:
                    continue

                # Run schema-safe LAYA pass
                laya_eval = self.laya.evaluate_pair(premise=note_text, hypothesis=evidence_text)
                if laya_eval["is_contradiction"]:
                    contradictions.append({
                        "type": conflict_type,
                        "note_id": note.id,
                        "invalidating_event_id": ev.id,
                        "premise_note": note_text,
                        "invalidating_evidence": evidence_text,
                        "confidence": laya_eval["confidence"],
                        "evaluator": laya_eval["model"],
                        "timestamp": ev.timestamp,
                    })

        return contradictions
