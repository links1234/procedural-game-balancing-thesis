from __future__ import annotations

from evaluation.metrics import compare_to_reference, observed_simulation_metrics, team_signature_from_player_state
from evaluation.robustness import summarize_robustness


def _row(
    *,
    comparison_seed: int,
    policy: str,
    wins: int,
    lives: int,
    signature: list[str],
    artifact_in_final_team: bool = False,
    artifact_copies_final: int = 0,
) -> dict:
    return {
        "pack": "StandardPack",
        "policy": policy,
        "comparison_seed": comparison_seed,
        "final_wins": wins,
        "final_lives": lives,
        "final_team_signature": signature,
        "artifact_in_final_team": artifact_in_final_team,
        "artifact_copies_final": artifact_copies_final,
        "episode_runtime_sec": 0.25,
        "invalid_action_count": 0,
        "backend_error": False,
    }


def test_team_signature_from_player_state_extracts_non_empty_pets() -> None:
    player_state = {
        "team": {
            "team": [
                {"pet": {"name": "pet-fish"}},
                {"pet": {"name": "pet-none"}},
                {"pet": {"name": "pet-ant"}},
            ]
        }
    }

    assert team_signature_from_player_state(player_state) == ("pet-ant", "pet-fish")


def test_compare_to_reference_rewards_lower_spread_with_retained_performance() -> None:
    baseline_rows = [
        _row(comparison_seed=1, policy="heuristic", wins=8, lives=8, signature=["pet-ant", "pet-fish"]),
        _row(comparison_seed=2, policy="heuristic", wins=2, lives=2, signature=["pet-beaver"]),
    ]
    candidate_rows = [
        _row(comparison_seed=1, policy="heuristic", wins=6, lives=6, signature=["pet-ant", "pet-fish"]),
        _row(comparison_seed=2, policy="heuristic", wins=4, lives=4, signature=["pet-beaver", "pet-fish"]),
    ]

    observed = observed_simulation_metrics(candidate_rows)
    comparison = compare_to_reference(candidate_rows, baseline_rows)

    assert observed["normalized_team_entropy"] > 0.0
    assert comparison["paired_cell_count"] == 2
    assert comparison["spread_reduction"] > 0.0
    assert comparison["performance_retention"] == 1.0
    assert comparison["fairness_score"] > 0.0
    assert comparison["safety_pass"] is True


def test_compare_to_reference_flags_dominance_and_diversity_collapse() -> None:
    baseline_rows = [
        _row(comparison_seed=1, policy="heuristic", wins=5, lives=5, signature=["pet-ant", "pet-fish"]),
        _row(comparison_seed=2, policy="heuristic", wins=5, lives=5, signature=["pet-beaver", "pet-otter"]),
    ]
    candidate_rows = [
        _row(
            comparison_seed=1,
            policy="heuristic",
            wins=10,
            lives=10,
            signature=["pet-generated", "pet-generated", "pet-fish"],
            artifact_in_final_team=True,
            artifact_copies_final=2,
        ),
        _row(
            comparison_seed=2,
            policy="heuristic",
            wins=10,
            lives=10,
            signature=["pet-generated", "pet-generated", "pet-fish"],
            artifact_in_final_team=True,
            artifact_copies_final=2,
        ),
    ]

    comparison = compare_to_reference(candidate_rows, baseline_rows)

    assert comparison["safety_pass"] is False
    assert "dominance_outcome_delta" in comparison["safety_violations"]
    assert "diversity_collapse" in comparison["safety_violations"]
    assert "artifact_saturation" in comparison["safety_violations"]


def test_summarize_robustness_reports_stability_over_paired_cells() -> None:
    baseline_rows = [
        _row(comparison_seed=1, policy="heuristic", wins=5, lives=5, signature=["pet-ant"]),
        _row(comparison_seed=2, policy="heuristic", wins=6, lives=6, signature=["pet-fish"]),
        _row(comparison_seed=1, policy="random", wins=3, lives=3, signature=["pet-beaver"]),
        _row(comparison_seed=2, policy="random", wins=4, lives=4, signature=["pet-otter"]),
    ]
    candidate_rows = [
        _row(comparison_seed=1, policy="heuristic", wins=6, lives=6, signature=["pet-ant"]),
        _row(comparison_seed=2, policy="heuristic", wins=7, lives=7, signature=["pet-fish"]),
        _row(comparison_seed=1, policy="random", wins=5, lives=5, signature=["pet-beaver"]),
        _row(comparison_seed=2, policy="random", wins=3, lives=3, signature=["pet-otter"]),
    ]

    robustness = summarize_robustness(candidate_rows, baseline_rows)

    assert robustness["paired_cell_count"] == 4
    assert robustness["cell_effect_std"] > 0.0
    assert 0.0 < robustness["stability_score"] <= 1.0
    assert set(robustness["policy_mean_deltas"]) == {"heuristic", "random"}
