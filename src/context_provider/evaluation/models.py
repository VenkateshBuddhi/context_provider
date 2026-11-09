"""
Data models for evaluating cross-session AI coding agent task continuation.

Compares three experimental handoff conditions:
1. REPO_ONLY: No handoff documents; agent inspects raw repository only.
2. MANUAL_CONTEXT: Static manually maintained CONTEXT.md without empirical evidence.
3. AUTO_GENERATED: Full evidence-based context synthesized by LAYA and llama3.1.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone


class EvaluationCondition(str, Enum):
    REPO_ONLY = "repo_only"
    MANUAL_CONTEXT = "manual_context"
    AUTO_GENERATED = "auto_generated"


@dataclass
class HandoffPackage:
    """Represents the context materials handed off to an incoming AI agent."""
    condition: EvaluationCondition
    files: Dict[str, str] = field(default_factory=dict)
    token_count: int = 0
    traps_exposed: int = 0
    contradictions_flagged: int = 0
    evidence_items_count: int = 0


@dataclass
class BenchmarkMetrics:
    """Quantitative performance scores for cross-session continuation."""
    condition: EvaluationCondition
    trap_avoidance_rate: float        # [0.0 - 1.0] Proportion of abandoned traps warned against
    contradiction_resolution_rate: float # [0.0 - 1.0] Proportion of conflicting claims flagged
    evidence_grounding_score: float   # [0.0 - 1.0] Density of empirical evidence (tests, commits, exit codes)
    resume_readiness_score: float     # [0.0 - 1.0] Clarity of immediate next steps and current focus
    token_efficiency: float           # Information density score
    composite_continuation_score: float # Overall effectiveness rating [0.0 - 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition.value,
            "trap_avoidance_rate": round(self.trap_avoidance_rate, 3),
            "contradiction_resolution_rate": round(self.contradiction_resolution_rate, 3),
            "evidence_grounding_score": round(self.evidence_grounding_score, 3),
            "resume_readiness_score": round(self.resume_readiness_score, 3),
            "token_efficiency": round(self.token_efficiency, 3),
            "composite_continuation_score": round(self.composite_continuation_score, 3),
        }


@dataclass
class EvaluationReport:
    """Comprehensive comparative evaluation report across all three conditions."""
    project_name: str
    session_id: str
    evaluated_at: str
    total_known_traps: int
    total_known_contradictions: int
    results: Dict[str, BenchmarkMetrics] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "session_id": self.session_id,
            "evaluated_at": self.evaluated_at,
            "ground_truth": {
                "known_traps": self.total_known_traps,
                "known_contradictions": self.total_known_contradictions,
            },
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }

