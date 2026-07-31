from __future__ import annotations

import json
from pathlib import Path

from generator.search.random_search import (
    SearchState,
    candidate_record,
    load_checkpoint,
    sample_candidate,
    save_checkpoint,
)


def test_sample_candidate_deterministic_by_iteration() -> None:
    a = sample_candidate(seed=20260301, iteration=12)
    b = sample_candidate(seed=20260301, iteration=12)
    assert a == b


def test_candidate_record_has_fast_filter_flag() -> None:
    payload = sample_candidate(seed=20260301, iteration=1)
    rec = candidate_record(iteration=1, payload=payload, strict_pet_only=True)
    assert "fast_filter_pass" in rec
    assert "objective" in rec


def test_candidate_record_can_require_injection_support() -> None:
    payload = {
        "artifact_type": "pet",
        "tier": 3,
        "base_attack": 4,
        "base_health": 4,
        "trigger_type": "Sell",
        "target_type": "AdjacentFriend",
        "effect_type": "GiveStatus",
        "effect_value": 2,
        "cap_or_cooldown": 1,
        "cost": 2,
        "tags": ["utility"],
        "source": "test",
    }
    rec = candidate_record(
        iteration=0,
        payload=payload,
        strict_pet_only=True,
        require_injection_support=True,
    )
    assert rec["fast_filter_pass"] is False
    assert rec["protocol_issues"]


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    cp = tmp_path / "checkpoint.json"
    state = SearchState(
        next_iteration=10,
        accepted_count=4,
        rejected_count=6,
        search_best_record={"proxy_objective": 0.8},
        best_record={"objective": 0.7},
        best_safe_record={"objective": 0.65},
        evaluated_count=4,
    )
    save_checkpoint(cp, state)
    loaded = load_checkpoint(cp)
    assert loaded == state
