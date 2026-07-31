"""Evaluation helpers for proxy and simulator-backed thesis metrics."""

from evaluation.metrics import (
    compare_to_reference,
    composite_objective,
    fairness_proxy,
    observed_simulation_metrics,
    outcome_score,
    team_signature_from_player_state,
)
from evaluation.protocol import EvaluationProtocol, evaluate_artifact_across_protocol, final_objective_from_metrics, full_protocol_evaluation, pick_policy, run_episode
from evaluation.robustness import paired_outcome_deltas, summarize_robustness

__all__ = [
    "compare_to_reference",
    "composite_objective",
    "EvaluationProtocol",
    "evaluate_artifact_across_protocol",
    "fairness_proxy",
    "final_objective_from_metrics",
    "full_protocol_evaluation",
    "observed_simulation_metrics",
    "outcome_score",
    "paired_outcome_deltas",
    "pick_policy",
    "run_episode",
    "summarize_robustness",
    "team_signature_from_player_state",
]
