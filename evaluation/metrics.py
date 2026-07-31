"""Evaluation helpers for proxy scoring and simulator-backed thesis metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import log
from typing import Any, Iterable, Mapping

import numpy as np

from generator.schema import ArtifactSpec
from generator.validators import TIER_POWER_BUDGET_MAX, TIER_POWER_BUDGET_MIN


DEFAULT_CELL_FIELDS = ("pack", "policy", "comparison_seed")
OUTCOME_WINS_WEIGHT = 0.75
OUTCOME_LIVES_WEIGHT = 0.25
MIN_SPREAD_DENOMINATOR = 0.05
MIN_RETENTION_DENOMINATOR = 0.10
DOMINANCE_DELTA_THRESHOLD = 0.18
ARTIFACT_SATURATION_THRESHOLD = 0.80
DIVERSITY_RETENTION_THRESHOLD = 0.75


def estimated_power_score(spec: ArtifactSpec) -> float:
    return (
        spec.base_attack * 0.9
        + spec.base_health * 0.9
        + spec.effect_value * 1.6
        + spec.cap_or_cooldown * 0.8
    )


def fairness_proxy(spec: ArtifactSpec) -> float:
    """Higher is better. Range is approximately [0, 1]."""
    pscore = estimated_power_score(spec)
    pmin = TIER_POWER_BUDGET_MIN[spec.tier]
    pmax = TIER_POWER_BUDGET_MAX[spec.tier]
    center = (pmin + pmax) / 2.0
    max_distance = max((pmax - pmin) / 2.0, 1.0)
    distance = abs(pscore - center)
    score = max(0.0, 1.0 - (distance / max_distance))
    return float(score)


def diversity_proxy(spec: ArtifactSpec) -> float:
    """Simple diversity proxy from tag coverage and target flexibility."""
    tag_score = min(len(set(spec.tags)), 4) / 4.0
    target_bonus = 0.1 if spec.target_type in {"RandomFriend", "RandomEnemy", "AllFriends"} else 0.0
    return float(min(1.0, tag_score + target_bonus))


def stability_proxy(spec: ArtifactSpec) -> float:
    """Proxy for predictable behavior under repeated simulations."""
    if spec.cap_or_cooldown == 0:
        return 0.2
    return float(min(1.0, 0.4 + (0.2 * spec.cap_or_cooldown)))


def composite_objective(spec: ArtifactSpec) -> dict[str, Any]:
    fairness = fairness_proxy(spec)
    diversity = diversity_proxy(spec)
    stability = stability_proxy(spec)
    objective = (0.65 * fairness) + (0.20 * diversity) + (0.15 * stability)
    return {
        "fairness_proxy": float(fairness),
        "diversity_proxy": float(diversity),
        "stability_proxy": float(stability),
        "objective_raw": float(objective),
        "estimated_power_score": float(estimated_power_score(spec)),
    }


def team_signature_from_player_state(player_state: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(player_state, Mapping):
        return ()
    team_state = player_state.get("team")
    if not isinstance(team_state, Mapping):
        return ()
    slots = team_state.get("team", [])
    if not isinstance(slots, list):
        return ()

    names: list[str] = []
    for slot in slots:
        if not isinstance(slot, Mapping):
            continue
        pet = slot.get("pet")
        if not isinstance(pet, Mapping):
            continue
        name = pet.get("name")
        if not isinstance(name, str) or name == "pet-none":
            continue
        names.append(name)
    return tuple(sorted(names))


def outcome_score(row: Mapping[str, Any]) -> float:
    wins = float(row.get("final_wins", 0.0)) / 10.0
    lives = float(row.get("final_lives", 0.0)) / 10.0
    score = (OUTCOME_WINS_WEIGHT * wins) + (OUTCOME_LIVES_WEIGHT * lives)
    return float(min(1.0, max(0.0, score)))


def normalized_entropy(values: Iterable[tuple[str, ...]]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total <= 1 or len(counts) <= 1:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * log(probability)
    return float(entropy / log(len(counts)))


def concentration_index(values: Iterable[tuple[str, ...]]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return float(sum((count / total) ** 2 for count in counts.values()))


def mean_abs_deviation(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    center = float(arr.mean())
    return float(np.abs(arr - center).mean())


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _cell_key(row: Mapping[str, Any], cell_fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in cell_fields)


def _cell_mean_scores(
    rows: list[dict[str, Any]],
    *,
    cell_fields: tuple[str, ...] = DEFAULT_CELL_FIELDS,
) -> dict[tuple[Any, ...], float]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[_cell_key(row, cell_fields)].append(outcome_score(row))
    return {
        key: float(np.mean(np.array(scores, dtype=float)))
        for key, scores in grouped.items()
    }


def observed_simulation_metrics(
    rows: list[dict[str, Any]],
    *,
    cell_fields: tuple[str, ...] = DEFAULT_CELL_FIELDS,
) -> dict[str, Any]:
    if not rows:
        return {
            "episodes": 0,
            "paired_cell_count": 0,
            "mean_outcome_score": 0.0,
            "outcome_cell_spread": 0.0,
            "normalized_team_entropy": 0.0,
            "team_concentration_index": 0.0,
            "unique_team_ratio": 0.0,
            "artifact_in_final_team_rate": 0.0,
            "mean_artifact_copies_final": 0.0,
            "backend_error_rate": 0.0,
            "mean_invalid_actions": 0.0,
            "mean_episode_runtime_sec": 0.0,
            "episodes_per_second": 0.0,
        }

    team_signatures = [
        tuple(row.get("final_team_signature", []))
        for row in rows
    ]
    cell_scores = _cell_mean_scores(rows, cell_fields=cell_fields)
    runtimes = np.array([float(row.get("episode_runtime_sec", 0.0)) for row in rows], dtype=float)
    invalid_actions = np.array([float(row.get("invalid_action_count", 0.0)) for row in rows], dtype=float)
    backend_errors = np.array(
        [1.0 if row.get("backend_error") else 0.0 for row in rows],
        dtype=float,
    )
    artifact_presence = np.array(
        [1.0 if row.get("artifact_in_final_team") else 0.0 for row in rows],
        dtype=float,
    )
    artifact_copies = np.array(
        [float(row.get("artifact_copies_final", 0.0)) for row in rows],
        dtype=float,
    )
    episode_scores = np.array([outcome_score(row) for row in rows], dtype=float)

    runtime_total = float(runtimes.sum())
    return {
        "episodes": len(rows),
        "paired_cell_count": len(cell_scores),
        "mean_outcome_score": float(episode_scores.mean()),
        "outcome_cell_spread": mean_abs_deviation(cell_scores.values()),
        "normalized_team_entropy": normalized_entropy(team_signatures),
        "team_concentration_index": concentration_index(team_signatures),
        "unique_team_ratio": float(len(set(team_signatures)) / len(rows)),
        "artifact_in_final_team_rate": float(artifact_presence.mean()),
        "mean_artifact_copies_final": float(artifact_copies.mean()),
        "backend_error_rate": float(backend_errors.mean()),
        "mean_invalid_actions": float(invalid_actions.mean()),
        "mean_episode_runtime_sec": float(runtimes.mean()),
        "episodes_per_second": float(len(rows) / runtime_total) if runtime_total > 0 else 0.0,
    }


def compare_to_reference(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    cell_fields: tuple[str, ...] = DEFAULT_CELL_FIELDS,
) -> dict[str, Any]:
    observed = observed_simulation_metrics(rows, cell_fields=cell_fields)
    baseline_observed = observed_simulation_metrics(baseline_rows, cell_fields=cell_fields)
    baseline_scores = _cell_mean_scores(baseline_rows, cell_fields=cell_fields)
    candidate_scores = _cell_mean_scores(rows, cell_fields=cell_fields)
    common_keys = sorted(set(baseline_scores) & set(candidate_scores))

    if not common_keys:
        return {
            "paired_cell_count": 0,
            "fairness_score": 0.0,
            "spread_reduction": 0.0,
            "spread_reduction_pct": 0.0,
            "baseline_mean_outcome_score": baseline_observed["mean_outcome_score"],
            "candidate_mean_outcome_score": observed["mean_outcome_score"],
            "mean_outcome_delta": 0.0,
            "performance_retention": 0.0,
            "baseline_team_entropy": baseline_observed["normalized_team_entropy"],
            "candidate_team_entropy": observed["normalized_team_entropy"],
            "diversity_retention": 0.0,
            "safety_pass": False,
            "safety_violation_count": 1,
            "safety_violations": ["missing_paired_cells"],
        }

    baseline_common = np.array([baseline_scores[key] for key in common_keys], dtype=float)
    candidate_common = np.array([candidate_scores[key] for key in common_keys], dtype=float)

    baseline_spread = mean_abs_deviation(baseline_common)
    candidate_spread = mean_abs_deviation(candidate_common)
    spread_reduction = float(baseline_spread - candidate_spread)
    spread_reduction_pct = float(spread_reduction / max(baseline_spread, MIN_SPREAD_DENOMINATOR))

    baseline_mean = float(baseline_common.mean())
    candidate_mean = float(candidate_common.mean())
    mean_outcome_delta = float(candidate_mean - baseline_mean)
    performance_retention = _clip01(
        1.0 - (abs(mean_outcome_delta) / max(abs(baseline_mean), MIN_RETENTION_DENOMINATOR))
    )
    fairness_score = _clip01(
        (0.65 * max(0.0, spread_reduction_pct)) + (0.35 * performance_retention)
    )

    baseline_entropy = float(baseline_observed["normalized_team_entropy"])
    candidate_entropy = float(observed["normalized_team_entropy"])
    if baseline_entropy <= 1e-9:
        diversity_retention = 1.0 if candidate_entropy <= 1e-9 else 1.0
    else:
        diversity_retention = float(candidate_entropy / baseline_entropy)

    safety_violations: list[str] = []
    if mean_outcome_delta > DOMINANCE_DELTA_THRESHOLD:
        safety_violations.append("dominance_outcome_delta")
    if observed["artifact_in_final_team_rate"] >= ARTIFACT_SATURATION_THRESHOLD and mean_outcome_delta > 0.08:
        safety_violations.append("artifact_saturation")
    if diversity_retention < DIVERSITY_RETENTION_THRESHOLD:
        safety_violations.append("diversity_collapse")
    if observed["backend_error_rate"] > 0.0:
        safety_violations.append("backend_error_rate")

    return {
        "paired_cell_count": len(common_keys),
        "fairness_score": fairness_score,
        "spread_reduction": spread_reduction,
        "spread_reduction_pct": spread_reduction_pct,
        "baseline_outcome_cell_spread": baseline_spread,
        "candidate_outcome_cell_spread": candidate_spread,
        "baseline_mean_outcome_score": baseline_mean,
        "candidate_mean_outcome_score": candidate_mean,
        "mean_outcome_delta": mean_outcome_delta,
        "performance_retention": performance_retention,
        "baseline_team_entropy": baseline_entropy,
        "candidate_team_entropy": candidate_entropy,
        "diversity_retention": diversity_retention,
        "safety_pass": not safety_violations,
        "safety_violation_count": len(safety_violations),
        "safety_violations": safety_violations,
    }
