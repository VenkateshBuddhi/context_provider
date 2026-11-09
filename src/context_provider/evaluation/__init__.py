"""Cross-session continuation evaluation package."""
from context_provider.evaluation.models import (
    EvaluationCondition,
    HandoffPackage,
    BenchmarkMetrics,
    EvaluationReport,
)
from context_provider.evaluation.harness import EvaluationHarness
from context_provider.evaluation.sandbox import TaskSandbox
from context_provider.evaluation.agent_runner import HeadlessAgentRunner, AgentRunReport
from context_provider.evaluation.differential import DifferentialExperiment

__all__ = [
    "EvaluationCondition",
    "HandoffPackage",
    "BenchmarkMetrics",
    "EvaluationReport",
    "EvaluationHarness",
    "TaskSandbox",
    "HeadlessAgentRunner",
    "AgentRunReport",
    "DifferentialExperiment",
]

