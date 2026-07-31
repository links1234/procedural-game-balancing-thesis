"""Shared simulator-backed evaluation protocol helpers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from evaluation.metrics import compare_to_reference, observed_simulation_metrics, team_signature_from_player_state
from evaluation.robustness import summarize_robustness
from generator.schema import ArtifactSpec
from opponents.heuristic_policy import BasicHeuristicPolicy
from opponents.learned_policy import LearnedPolicyStub, ScriptedValuePolicy
from opponents.random_policy import RandomActionPolicy
from sap_thesis_env.env.frozen_arena_env import FrozenArenaEnv
from sap_thesis_env.generated_artifacts import UnsupportedArtifactSpecError, artifact_set_id, normalize_artifact_specs


@dataclass(frozen=True)
class EvaluationProtocol:
    packs: tuple[str, ...]
    policies: tuple[str, ...]
    comparison_seeds: tuple[int, ...]
    max_turn: int = 25
    max_steps_per_episode: int = 256
    strict_invalid_actions: bool = True


def pick_policy(name: str, seed: int):
    if name == "random":
        return RandomActionPolicy(seed=seed)
    if name == "heuristic":
        return BasicHeuristicPolicy()
    if name == "scripted_value":
        return ScriptedValuePolicy()
    if name == "learned_stub":
        return LearnedPolicyStub()
    raise ValueError(f"Unknown policy '{name}'. Expected one of: random, heuristic, scripted_value, learned_stub")


def run_episode(
    env: FrozenArenaEnv,
    policy,
    episode_seed: int,
    max_steps: int,
) -> dict[str, Any]:
    started = perf_counter()
    obs = env.reset(seed=episode_seed)
    total_reward = 0.0
    steps = 0
    backend_error = False
    backend_error_message = ""

    for _ in range(max_steps):
        try:
            action_id = policy.select_action(env)
            obs, reward, done, _info = env.step(action_id)
            total_reward += float(reward)
            steps += 1
            if done:
                break
        except Exception as exc:  # pragma: no cover - exercised by smoke/debug runs
            backend_error = True
            backend_error_message = f"{type(exc).__name__}: {exc}"
            break

    final_team_signature = team_signature_from_player_state(obs.get("player_state"))
    injected_pet_names = list(obs.get("injected_pet_names", []))
    if not injected_pet_names and obs.get("injected_pet_name") is not None:
        injected_pet_names = [obs["injected_pet_name"]]
    injected_pet_name = injected_pet_names[0] if injected_pet_names else None
    artifact_copies_final = sum(final_team_signature.count(name) for name in injected_pet_names)

    return {
        "steps": steps,
        "final_turn": int(obs["turn"]),
        "final_wins": int(obs["wins"]),
        "final_lives": int(obs["lives"]),
        "total_reward": float(total_reward),
        "invalid_action_count": int(env.invalid_action_count),
        "done": bool(obs["done"]),
        "backend_error": backend_error,
        "backend_error_message": backend_error_message,
        "episode_runtime_sec": float(perf_counter() - started),
        "final_team_signature": list(final_team_signature),
        "final_team_size": len(final_team_signature),
        "artifact_in_final_team": bool(artifact_copies_final > 0),
        "artifact_copies_final": int(artifact_copies_final),
        "injected_artifact_id": obs.get("injected_artifact_id"),
        "injected_artifact_ids": list(obs.get("injected_artifact_ids", [])),
        "injected_pet_name": injected_pet_name,
        "injected_pet_names": injected_pet_names,
        "injected_shop_slot_index": obs.get("injected_shop_slot_index"),
        "injected_shop_slot_indices": list(obs.get("injected_shop_slot_indices", [])),
    }


def evaluate_artifact_across_protocol(
    artifacts: ArtifactSpec | dict[str, Any] | list[ArtifactSpec | dict[str, Any]] | tuple[ArtifactSpec | dict[str, Any], ...] | None,
    protocol: EvaluationProtocol,
) -> list[dict[str, Any]]:
    normalized_artifacts = normalize_artifact_specs(artifacts)
    artifact_payloads = [artifact.to_dict() for artifact in normalized_artifacts]
    single_artifact_payload = artifact_payloads[0] if len(artifact_payloads) == 1 else None
    artifact_ids = [artifact.artifact_id for artifact in normalized_artifacts]
    set_id = artifact_set_id(normalized_artifacts) if normalized_artifacts else "baseline"

    rows: list[dict[str, Any]] = []
    for pack in protocol.packs:
        for policy_name in protocol.policies:
            for comparison_seed in protocol.comparison_seeds:
                env = FrozenArenaEnv(
                    seed=comparison_seed,
                    pack=pack,
                    max_turn=protocol.max_turn,
                    strict_invalid_actions=protocol.strict_invalid_actions,
                    injected_artifacts=normalized_artifacts,
                )
                policy = pick_policy(policy_name, seed=comparison_seed)
                metrics = run_episode(
                    env=env,
                    policy=policy,
                    episode_seed=comparison_seed,
                    max_steps=protocol.max_steps_per_episode,
                )
                row = {
                    "pack": pack,
                    "policy": policy_name,
                    "policy_seed": comparison_seed,
                    "comparison_seed": comparison_seed,
                    "max_turn": protocol.max_turn,
                    "artifact": single_artifact_payload,
                    "artifacts": artifact_payloads,
                    "artifact_ids": artifact_ids,
                    "artifact_set_id": set_id,
                }
                row.update(metrics)
                rows.append(row)
    return rows


def final_objective_from_metrics(
    comparison: dict[str, Any],
    robustness: dict[str, Any],
) -> float:
    fairness = float(comparison.get("fairness_score", 0.0))
    diversity = min(1.0, max(0.0, float(comparison.get("diversity_retention", 0.0))))
    stability = min(1.0, max(0.0, float(robustness.get("stability_score", 0.0))))
    return float((0.60 * fairness) + (0.20 * diversity) + (0.20 * stability))


def full_protocol_evaluation(
    artifacts: ArtifactSpec | dict[str, Any] | list[ArtifactSpec | dict[str, Any]] | tuple[ArtifactSpec | dict[str, Any], ...],
    *,
    protocol: EvaluationProtocol,
    reference_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    started = perf_counter()
    normalized_artifacts = normalize_artifact_specs(artifacts)
    try:
        rows = evaluate_artifact_across_protocol(normalized_artifacts, protocol)
    except UnsupportedArtifactSpecError as exc:
        return {
            "supported": False,
            "support_error": str(exc),
            "artifact_id": normalized_artifacts[0].artifact_id if len(normalized_artifacts) == 1 else None,
            "artifact_ids": [artifact.artifact_id for artifact in normalized_artifacts],
            "artifact_set_id": artifact_set_id(normalized_artifacts) if normalized_artifacts else "baseline",
            "rows": [],
            "observed_simulation": {},
            "comparison_to_reference": {
                "paired_cell_count": 0,
                "fairness_score": 0.0,
                "diversity_retention": 0.0,
                "performance_retention": 0.0,
                "safety_pass": False,
                "safety_violation_count": 1,
                "safety_violations": ["unsupported_artifact"],
            },
            "robustness": {
                "paired_cell_count": 0,
                "stability_score": 0.0,
            },
            "efficiency": {
                "episodes_evaluated": 0,
                "wall_clock_sec": float(perf_counter() - started),
            },
            "final_objective": -1.0,
        }

    observed = observed_simulation_metrics(rows)
    comparison = compare_to_reference(rows, reference_rows)
    robustness = summarize_robustness(rows, reference_rows)
    final_objective = final_objective_from_metrics(comparison, robustness)
    return {
        "supported": True,
        "support_error": "",
        "artifact_id": normalized_artifacts[0].artifact_id if len(normalized_artifacts) == 1 else None,
        "artifact_ids": [artifact.artifact_id for artifact in normalized_artifacts],
        "artifact_set_id": artifact_set_id(normalized_artifacts) if normalized_artifacts else "baseline",
        "rows": rows,
        "observed_simulation": observed,
        "comparison_to_reference": comparison,
        "robustness": robustness,
        "efficiency": {
            "episodes_evaluated": len(rows),
            "wall_clock_sec": float(perf_counter() - started),
            "mean_episode_runtime_sec": observed.get("mean_episode_runtime_sec", 0.0),
            "episodes_per_second": observed.get("episodes_per_second", 0.0),
        },
        "final_objective": final_objective,
    }
