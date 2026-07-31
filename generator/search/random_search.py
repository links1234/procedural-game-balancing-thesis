"""Deterministic random-search engine for artifact optimization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.metrics import composite_objective, fairness_proxy
from evaluation.safety import safety_penalty
from generator.schema import ArtifactSpec
from generator.validators import ValidationIssue, validate_artifact
from sap_thesis_env.generated_artifacts import UnsupportedArtifactSpecError, validate_artifact_support


TRIGGER_OPTIONS = [
    "StartOfBattle",
    "EndOfTurn",
    "Faint",
    "Hurt",
    "Sell",
    "BuyFood",
    "BuyFriend",
]
TARGET_OPTIONS = [
    "Self",
    "RandomFriend",
    "AdjacentFriend",
    "AllFriends",
    "RandomEnemy",
    "HighestAttackEnemy",
    "None",
]
EFFECT_OPTIONS = [
    "BuffFriend",
    "DebuffEnemy",
    "Summon",
    "GoldGain",
    "DamageEnemy",
    "GiveStatus",
]
TAG_OPTIONS = [
    "economy",
    "tempo",
    "scaling",
    "summon",
    "debuff",
    "utility",
    "early_game",
    "late_game",
    "risk",
    "combo",
]

SUPPORTED_TRIGGER_OPTIONS = {
    "BuffFriend": ["StartOfBattle", "EndOfTurn", "Faint", "Hurt", "Sell", "BuyFood", "BuyFriend"],
    "DamageEnemy": ["StartOfBattle", "Faint", "Hurt"],
    "GoldGain": ["Sell", "EndOfTurn", "BuyFriend"],
}
SUPPORTED_TARGET_OPTIONS = {
    "BuffFriend": ["Self", "RandomFriend", "AdjacentFriend"],
    "DamageEnemy": ["RandomEnemy"],
    "GoldGain": ["None"],
}
SUPPORTED_TAG_OPTIONS = {
    "BuffFriend": ["tempo", "scaling", "utility"],
    "DamageEnemy": ["tempo", "risk", "early_game"],
    "GoldGain": ["economy", "utility", "tempo"],
}


@dataclass(frozen=True)
class SearchState:
    next_iteration: int
    accepted_count: int
    rejected_count: int
    search_best_record: dict[str, Any] | None
    best_record: dict[str, Any] | None
    best_safe_record: dict[str, Any] | None = None
    evaluated_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_iteration": self.next_iteration,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "search_best_record": self.search_best_record,
            "best_record": self.best_record,
            "best_safe_record": self.best_safe_record,
            "evaluated_count": self.evaluated_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchState":
        return cls(
            next_iteration=int(payload.get("next_iteration", 0)),
            accepted_count=int(payload.get("accepted_count", 0)),
            rejected_count=int(payload.get("rejected_count", 0)),
            search_best_record=payload.get("search_best_record"),
            best_record=payload.get("best_record"),
            best_safe_record=payload.get("best_safe_record"),
            evaluated_count=int(payload.get("evaluated_count", 0)),
        )


def save_checkpoint(path: Path, state: SearchState) -> None:
    path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_checkpoint(path: Path) -> SearchState:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SearchState.from_dict(payload)


def _iteration_rng(seed: int, iteration: int) -> np.random.RandomState:
    return np.random.RandomState(seed + (iteration * 7919))


def _supported_bundle(rng: np.random.RandomState) -> dict[str, Any]:
    effect_type = str(rng.choice(["BuffFriend", "DamageEnemy", "GoldGain"]).item())
    n_tags = int(rng.randint(1, 3))
    return {
        "effect_type": effect_type,
        "trigger_type": str(rng.choice(SUPPORTED_TRIGGER_OPTIONS[effect_type]).item()),
        "target_type": str(rng.choice(SUPPORTED_TARGET_OPTIONS[effect_type]).item()),
        "tags": rng.choice(SUPPORTED_TAG_OPTIONS[effect_type], size=n_tags, replace=False).tolist(),
    }


def sample_candidate(seed: int, iteration: int, *, supported_subset_only: bool = False) -> dict[str, Any]:
    rng = _iteration_rng(seed, iteration)
    if supported_subset_only:
        bundle = _supported_bundle(rng)
        payload = {
            "artifact_type": "pet",
            "tier": int(rng.randint(1, 7)),
            "base_attack": int(rng.randint(1, 9)),
            "base_health": int(rng.randint(1, 9)),
            "trigger_type": bundle["trigger_type"],
            "target_type": bundle["target_type"],
            "effect_type": bundle["effect_type"],
            "effect_value": int(rng.randint(1, 5)),
            "cap_or_cooldown": int(rng.randint(0, 4)),
            "cost": int(rng.randint(2, 4)),
            "tags": bundle["tags"],
            "source": "phase4_random_search",
        }
        return payload

    n_tags = int(rng.randint(1, 5))
    tags = rng.choice(TAG_OPTIONS, size=n_tags, replace=False).tolist()
    payload = {
        "artifact_type": "pet",
        "tier": int(rng.randint(1, 7)),
        "base_attack": int(rng.randint(0, 13)),
        "base_health": int(rng.randint(0, 13)),
        "trigger_type": str(rng.choice(TRIGGER_OPTIONS).item()),
        "target_type": str(rng.choice(TARGET_OPTIONS).item()),
        "effect_type": str(rng.choice(EFFECT_OPTIONS).item()),
        "effect_value": int(rng.randint(0, 6)),
        "cap_or_cooldown": int(rng.randint(0, 4)),
        "cost": int(rng.randint(1, 4)),
        "tags": tags,
        "source": "phase4_random_search",
    }
    return payload


def mutate_candidate(
    seed: int,
    iteration: int,
    base_payload: dict[str, Any],
    *,
    supported_subset_only: bool = False,
) -> dict[str, Any]:
    """Generate a local mutation around an existing candidate."""
    rng = _iteration_rng(seed + 4049, iteration)
    payload = dict(base_payload)
    payload["base_attack"] = int(max(0, min(50, payload["base_attack"] + int(rng.randint(-2, 3)))))
    payload["base_health"] = int(max(0, min(50, payload["base_health"] + int(rng.randint(-2, 3)))))
    payload["effect_value"] = int(max(0, min(20, payload["effect_value"] + int(rng.randint(-1, 2)))))
    payload["cap_or_cooldown"] = int(max(0, min(10, payload["cap_or_cooldown"] + int(rng.randint(-1, 2)))))
    if supported_subset_only:
        if rng.rand() < 0.35:
            bundle = _supported_bundle(rng)
            payload["effect_type"] = bundle["effect_type"]
            payload["trigger_type"] = bundle["trigger_type"]
            payload["target_type"] = bundle["target_type"]
            payload["tags"] = bundle["tags"]
        elif rng.rand() < 0.35:
            payload["trigger_type"] = str(rng.choice(SUPPORTED_TRIGGER_OPTIONS[payload["effect_type"]]).item())
            payload["target_type"] = str(rng.choice(SUPPORTED_TARGET_OPTIONS[payload["effect_type"]]).item())
        payload["base_attack"] = int(max(1, min(20, payload["base_attack"])))
        payload["base_health"] = int(max(1, min(20, payload["base_health"])))
        payload["effect_value"] = int(max(1, min(8, payload["effect_value"])))
        payload["cost"] = int(max(2, min(3, payload.get("cost", 3))))
        payload["source"] = "phase4_iterative_search"
        return payload

    if rng.rand() < 0.25:
        payload["trigger_type"] = str(rng.choice(TRIGGER_OPTIONS).item())
    if rng.rand() < 0.25:
        payload["target_type"] = str(rng.choice(TARGET_OPTIONS).item())
    if rng.rand() < 0.25:
        payload["effect_type"] = str(rng.choice(EFFECT_OPTIONS).item())
    if rng.rand() < 0.35:
        n_tags = int(rng.randint(1, 5))
        payload["tags"] = rng.choice(TAG_OPTIONS, size=n_tags, replace=False).tolist()
    payload["source"] = "phase4_iterative_search"
    return payload


def candidate_record(
    iteration: int,
    payload: dict[str, Any],
    strict_pet_only: bool = True,
    objective_mode: str = "full",
    use_safety_penalty: bool = True,
    require_injection_support: bool = False,
) -> dict[str, Any]:
    result = validate_artifact(payload, strict_pet_only=strict_pet_only)

    base = {
        "iteration": iteration,
        "payload": payload,
        "schema_issues": [i.message for i in result.schema_issues],
        "hard_issues": [i.message for i in result.hard_issues],
        "protocol_issues": [],
        "safety_issues": [i.message for i in result.safety_issues],
        "is_schema_valid": not result.schema_issues,
        "is_hard_valid": not result.hard_issues,
        "fast_filter_pass": (not result.schema_issues and not result.hard_issues),
    }

    if result.artifact is None:
        base["accepted"] = False
        base["objective"] = -1.0
        return base

    spec: ArtifactSpec = result.artifact
    protocol_issues: list[str] = []
    if require_injection_support:
        try:
            validate_artifact_support(spec)
        except UnsupportedArtifactSpecError as exc:
            protocol_issues.append(str(exc))

    metrics = composite_objective(spec)
    if objective_mode == "fairness_only":
        objective_raw = float(fairness_proxy(spec))
    elif objective_mode == "full":
        objective_raw = float(metrics["objective_raw"])
    else:
        raise ValueError(f"Unsupported objective_mode='{objective_mode}'")

    penalty = safety_penalty(result.safety_issues) if use_safety_penalty else 0.0
    objective = float(objective_raw - penalty)
    base.update(
        {
            "artifact_id": spec.artifact_id,
            "metrics": metrics,
            "objective_mode": objective_mode,
            "safety_penalty": penalty,
            "objective": objective,
            "protocol_issues": protocol_issues,
            "fast_filter_pass": base["fast_filter_pass"] and not protocol_issues,
            "accepted": base["fast_filter_pass"] and not protocol_issues,
        }
    )
    return base


def issue_counts(record: dict[str, Any]) -> tuple[int, int, int]:
    return (
        len(record.get("schema_issues", [])),
        len(record.get("hard_issues", [])),
        len(record.get("safety_issues", [])),
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]], mode: str = "a") -> None:
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
