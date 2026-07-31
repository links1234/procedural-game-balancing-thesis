"""Simulator-backed robustness and stability summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from evaluation.metrics import DEFAULT_CELL_FIELDS, outcome_score


def _cell_key(row: dict[str, Any], cell_fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in cell_fields)


def _mean_scores_by_cell(
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


def paired_outcome_deltas(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    cell_fields: tuple[str, ...] = DEFAULT_CELL_FIELDS,
) -> list[dict[str, Any]]:
    baseline_scores = _mean_scores_by_cell(baseline_rows, cell_fields=cell_fields)
    candidate_scores = _mean_scores_by_cell(rows, cell_fields=cell_fields)
    common_keys = sorted(set(baseline_scores) & set(candidate_scores))

    deltas: list[dict[str, Any]] = []
    for key in common_keys:
        payload = {field: key[idx] for idx, field in enumerate(cell_fields)}
        payload["baseline_outcome_score"] = baseline_scores[key]
        payload["candidate_outcome_score"] = candidate_scores[key]
        payload["delta_outcome_score"] = candidate_scores[key] - baseline_scores[key]
        deltas.append(payload)
    return deltas


def summarize_robustness(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    cell_fields: tuple[str, ...] = DEFAULT_CELL_FIELDS,
) -> dict[str, Any]:
    deltas = paired_outcome_deltas(rows, baseline_rows, cell_fields=cell_fields)
    if not deltas:
        return {
            "paired_cell_count": 0,
            "mean_effect_delta": 0.0,
            "cell_effect_std": 0.0,
            "cell_effect_range": 0.0,
            "policy_effect_std": 0.0,
            "policy_effect_range": 0.0,
            "seed_effect_std": 0.0,
            "stability_score": 0.0,
            "policy_mean_deltas": {},
            "pack_mean_deltas": {},
        }

    delta_arr = np.array([float(row["delta_outcome_score"]) for row in deltas], dtype=float)
    by_policy: dict[str, list[float]] = defaultdict(list)
    by_pack: dict[str, list[float]] = defaultdict(list)
    by_seed: dict[str, list[float]] = defaultdict(list)
    for row in deltas:
        by_policy[str(row.get("policy"))].append(float(row["delta_outcome_score"]))
        by_pack[str(row.get("pack"))].append(float(row["delta_outcome_score"]))
        by_seed[str(row.get("comparison_seed"))].append(float(row["delta_outcome_score"]))

    policy_means = {
        key: float(np.mean(np.array(values, dtype=float)))
        for key, values in by_policy.items()
    }
    pack_means = {
        key: float(np.mean(np.array(values, dtype=float)))
        for key, values in by_pack.items()
    }
    seed_means = np.array(
        [float(np.mean(np.array(values, dtype=float))) for values in by_seed.values()],
        dtype=float,
    )
    policy_mean_arr = np.array(list(policy_means.values()), dtype=float)

    cell_effect_std = float(delta_arr.std(ddof=0))
    policy_effect_std = float(policy_mean_arr.std(ddof=0)) if policy_mean_arr.size else 0.0
    seed_effect_std = float(seed_means.std(ddof=0)) if seed_means.size else 0.0
    stability_score = float(1.0 / (1.0 + (4.0 * cell_effect_std) + (2.0 * policy_effect_std)))

    return {
        "paired_cell_count": len(deltas),
        "mean_effect_delta": float(delta_arr.mean()),
        "cell_effect_std": cell_effect_std,
        "cell_effect_range": float(delta_arr.max() - delta_arr.min()) if delta_arr.size else 0.0,
        "policy_effect_std": policy_effect_std,
        "policy_effect_range": (
            float(policy_mean_arr.max() - policy_mean_arr.min())
            if policy_mean_arr.size
            else 0.0
        ),
        "seed_effect_std": seed_effect_std,
        "stability_score": stability_score,
        "best_cell_effect": float(delta_arr.max()),
        "worst_cell_effect": float(delta_arr.min()),
        "policy_mean_deltas": policy_means,
        "pack_mean_deltas": pack_means,
    }
