"""Safety penalty helpers for generator optimization."""

from __future__ import annotations

from generator.validators import ValidationIssue


PENALTY_BY_CODE_PREFIX = {
    "safety.dominance": 0.35,
    "safety.cooldown_required": 0.20,
    "safety.complexity": 0.15,
    "safety.combo_chain_risk": 0.10,
}


def safety_penalty(issues: tuple[ValidationIssue, ...]) -> float:
    penalty = 0.0
    for issue in issues:
        penalty += PENALTY_BY_CODE_PREFIX.get(issue.code, 0.05)
    return float(min(1.0, penalty))
