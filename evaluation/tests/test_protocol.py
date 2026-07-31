from __future__ import annotations

from evaluation.protocol import EvaluationProtocol, evaluate_artifact_across_protocol, full_protocol_evaluation


def test_full_protocol_evaluation_runs_for_supported_generated_artifact() -> None:
    protocol = EvaluationProtocol(
        packs=("StandardPack",),
        policies=("heuristic",),
        comparison_seeds=(20260301,),
        max_turn=8,
        max_steps_per_episode=64,
        strict_invalid_actions=True,
    )
    artifact = {
        "artifact_type": "pet",
        "tier": 3,
        "base_attack": 4,
        "base_health": 5,
        "trigger_type": "Faint",
        "target_type": "RandomFriend",
        "effect_type": "BuffFriend",
        "effect_value": 2,
        "cap_or_cooldown": 1,
        "cost": 3,
        "tags": ["tempo", "scaling"],
        "source": "test",
    }

    reference_rows = evaluate_artifact_across_protocol(None, protocol)
    result = full_protocol_evaluation(artifact, protocol=protocol, reference_rows=reference_rows)

    assert result["supported"] is True
    assert result["artifact_id"] is not None
    assert len(result["rows"]) == 1
    assert "observed_simulation" in result
    assert "comparison_to_reference" in result
    assert "robustness" in result
    assert "final_objective" in result


def test_full_protocol_evaluation_supports_multi_artifact_sets() -> None:
    protocol = EvaluationProtocol(
        packs=("StandardPack",),
        policies=("heuristic",),
        comparison_seeds=(20260302,),
        max_turn=8,
        max_steps_per_episode=64,
        strict_invalid_actions=True,
    )
    artifacts = [
        {
            "artifact_type": "pet",
            "tier": 2,
            "base_attack": 3,
            "base_health": 4,
            "trigger_type": "StartOfBattle",
            "target_type": "RandomEnemy",
            "effect_type": "DamageEnemy",
            "effect_value": 1,
            "cap_or_cooldown": 1,
            "cost": 3,
            "tags": ["tempo"],
            "source": "test",
        },
        {
            "artifact_type": "pet",
            "tier": 3,
            "base_attack": 4,
            "base_health": 5,
            "trigger_type": "EndOfTurn",
            "target_type": "Self",
            "effect_type": "BuffFriend",
            "effect_value": 2,
            "cap_or_cooldown": 1,
            "cost": 3,
            "tags": ["scaling"],
            "source": "test",
        },
    ]

    reference_rows = evaluate_artifact_across_protocol(None, protocol)
    result = full_protocol_evaluation(artifacts, protocol=protocol, reference_rows=reference_rows)

    assert result["supported"] is True
    assert result["artifact_id"] is None
    assert len(result["artifact_ids"]) == 2
    assert result["artifact_set_id"] not in {"", "baseline"}
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert len(row["injected_artifact_ids"]) == 2
    assert len(row["injected_pet_names"]) == 2
    assert len(row["injected_shop_slot_indices"]) == 2
