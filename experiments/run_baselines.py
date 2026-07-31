#!/usr/bin/env python3
"""Baseline experiment runner.

This script executes baseline experiment groups and writes reproducible outputs:

- metadata.json (via shared metadata utility)
- episodes.jsonl
- summary.json
- summary_table.csv (aggregated)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compare_to_reference, observed_simulation_metrics
from evaluation.protocol import EvaluationProtocol, evaluate_artifact_across_protocol
from evaluation.robustness import summarize_robustness
from generator.validators import validate_artifact
from sap_thesis_env.config import ProjectConfig
from sap_thesis_env.run_metadata import write_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline experiments.")
    parser.add_argument(
        "--config",
        default="configs/baselines/phase2_baselines.yaml",
        help="Path to baseline YAML config.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config must be a YAML object/mapping.")
    return payload


def resolve_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def build_random_artifact(rng: np.random.RandomState) -> dict[str, Any]:
    effect_type = rng.choice(["BuffFriend", "DamageEnemy", "GoldGain"]).item()
    trigger_options = {
        "BuffFriend": ["StartOfBattle", "EndOfTurn", "Faint", "Hurt", "Sell", "BuyFood", "BuyFriend"],
        "DamageEnemy": ["StartOfBattle", "Faint", "Hurt"],
        "GoldGain": ["Sell", "EndOfTurn", "BuyFriend"],
    }
    target_options = {
        "BuffFriend": ["Self", "RandomFriend", "AdjacentFriend"],
        "DamageEnemy": ["RandomEnemy"],
        "GoldGain": ["None"],
    }
    tag_options = {
        "BuffFriend": [["tempo"], ["scaling"], ["utility"]],
        "DamageEnemy": [["tempo"], ["risk"], ["early_game"]],
        "GoldGain": [["economy"], ["utility"], ["tempo"]],
    }

    while True:
        artifact = {
            "artifact_type": "pet",
            "tier": int(rng.randint(1, 7)),
            "base_attack": int(rng.randint(1, 7)),
            "base_health": int(rng.randint(1, 7)),
            "trigger_type": rng.choice(trigger_options[effect_type]).item(),
            "target_type": rng.choice(target_options[effect_type]).item(),
            "effect_type": effect_type,
            "effect_value": int(rng.randint(1, 4)),
            "cap_or_cooldown": int(rng.randint(0, 3)),
            "cost": 3,
            "tags": list(tag_options[effect_type][int(rng.randint(0, len(tag_options[effect_type])))]),
            "source": "random_generation_baseline",
        }
        validation = validate_artifact(artifact)
        if validation.is_valid and validation.artifact is not None:
            return validation.artifact.to_dict()


def build_heuristic_artifact() -> dict[str, Any]:
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
        "source": "heuristic_generation_baseline",
    }
    validation = validate_artifact(artifact)
    if not validation.is_valid or validation.artifact is None:
        raise ValueError("Heuristic baseline artifact must remain schema-valid and safe.")
    return validation.artifact.to_dict()


def maybe_generate_artifact(
    baseline_name: str,
    rng: np.random.RandomState,
) -> dict[str, Any] | None:
    if baseline_name == "no_generation":
        return None
    if baseline_name == "random_generation":
        return build_random_artifact(rng)
    if baseline_name == "heuristic_generation":
        return build_heuristic_artifact()
    raise ValueError(f"Unknown baseline '{baseline_name}'")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["baseline"], []).append(row)

    reference_rows = grouped.get("no_generation", [])
    out: dict[str, Any] = {
        "num_rows": len(rows),
        "reference_baseline": "no_generation",
        "comparison_protocol": {
            "cell_fields": ["pack", "policy", "comparison_seed"],
            "paired_seeds_reused_across_baselines": True,
            "outcome_score_formula": "0.75 * final_wins/10 + 0.25 * final_lives/10",
            "diversity_measure": "normalized entropy over final team signatures",
            "stability_measure": "paired outcome delta variance across pack/policy/seed cells",
        },
        "baselines": {},
    }
    for baseline, records in grouped.items():
        wins = np.array([r["final_wins"] for r in records], dtype=float)
        lives = np.array([r["final_lives"] for r in records], dtype=float)
        rewards = np.array([r["total_reward"] for r in records], dtype=float)
        steps = np.array([r["steps"] for r in records], dtype=float)
        invalid = np.array([r["invalid_action_count"] for r in records], dtype=float)
        backend_errors = np.array([1.0 if r.get("backend_error") else 0.0 for r in records], dtype=float)

        baseline_summary: dict[str, Any] = {
            "episodes": len(records),
            "mean_final_wins": float(wins.mean()),
            "mean_final_lives": float(lives.mean()),
            "mean_total_reward": float(rewards.mean()),
            "mean_steps": float(steps.mean()),
            "mean_invalid_actions": float(invalid.mean()),
            "backend_error_rate": float(backend_errors.mean()),
            "observed_simulation": observed_simulation_metrics(records),
        }
        if baseline != "no_generation" and reference_rows:
            baseline_summary["comparison_to_no_generation"] = compare_to_reference(records, reference_rows)
            baseline_summary["robustness_vs_no_generation"] = summarize_robustness(records, reference_rows)
        else:
            baseline_summary["comparison_to_no_generation"] = None
            baseline_summary["robustness_vs_no_generation"] = None
        out["baselines"][baseline] = baseline_summary
    return out


def write_summary_table(summary: dict[str, Any], out_csv: Path) -> None:
    fields = [
        "baseline",
        "episodes",
        "mean_final_wins",
        "mean_final_lives",
        "mean_total_reward",
        "mean_steps",
        "mean_invalid_actions",
        "backend_error_rate",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for baseline, metrics in summary["baselines"].items():
            row = {"baseline": baseline}
            for field in fields:
                if field == "baseline":
                    continue
                row[field] = metrics.get(field)
            writer.writerow(row)


def write_protocol_table(summary: dict[str, Any], out_csv: Path) -> None:
    fields = [
        "baseline",
        "episodes",
        "mean_outcome_score",
        "outcome_cell_spread",
        "normalized_team_entropy",
        "artifact_in_final_team_rate",
        "mean_episode_runtime_sec",
        "fairness_score",
        "spread_reduction_pct",
        "mean_outcome_delta",
        "diversity_retention",
        "stability_score",
        "cell_effect_std",
        "safety_pass",
        "safety_violation_count",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for baseline, metrics in summary["baselines"].items():
            observed = metrics.get("observed_simulation", {})
            comparison = metrics.get("comparison_to_no_generation") or {}
            robustness = metrics.get("robustness_vs_no_generation") or {}
            writer.writerow(
                {
                    "baseline": baseline,
                    "episodes": metrics.get("episodes"),
                    "mean_outcome_score": observed.get("mean_outcome_score"),
                    "outcome_cell_spread": observed.get("outcome_cell_spread"),
                    "normalized_team_entropy": observed.get("normalized_team_entropy"),
                    "artifact_in_final_team_rate": observed.get("artifact_in_final_team_rate"),
                    "mean_episode_runtime_sec": observed.get("mean_episode_runtime_sec"),
                    "fairness_score": comparison.get("fairness_score"),
                    "spread_reduction_pct": comparison.get("spread_reduction_pct"),
                    "mean_outcome_delta": comparison.get("mean_outcome_delta"),
                    "diversity_retention": comparison.get("diversity_retention"),
                    "stability_score": robustness.get("stability_score"),
                    "cell_effect_std": robustness.get("cell_effect_std"),
                    "safety_pass": comparison.get("safety_pass"),
                    "safety_violation_count": comparison.get("safety_violation_count"),
                }
            )


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    raw_cfg = load_yaml(config_path)
    cfg = ProjectConfig.from_dict(raw_cfg)

    metadata_path = write_run_metadata(
        config=cfg,
        project_root=PROJECT_ROOT,
        config_path=config_path,
    )
    run_dir = metadata_path.parent

    episodes_per_baseline = int(raw_cfg.get("episodes_per_baseline", 20))
    max_steps_per_episode = int(raw_cfg.get("max_steps_per_episode", 256))
    pack = str(raw_cfg.get("pack", "StandardPack"))
    max_turn = int(raw_cfg.get("max_turn", 25))
    baselines = raw_cfg.get(
        "baselines",
        ["no_generation", "random_generation", "heuristic_generation"],
    )
    policy_name = str(raw_cfg.get("evaluation_policy", "heuristic"))

    rows: list[dict[str, Any]] = []
    seed_rng = np.random.RandomState(cfg.seed)
    comparison_seeds = [
        int(seed_rng.randint(0, 2**31 - 1))
        for _ in range(episodes_per_baseline)
    ]
    protocol = EvaluationProtocol(
        packs=(pack,),
        policies=(policy_name,),
        comparison_seeds=tuple(comparison_seeds),
        max_turn=max_turn,
        max_steps_per_episode=max_steps_per_episode,
        strict_invalid_actions=True,
    )

    for baseline_idx, baseline in enumerate(baselines):
        baseline_seed = int(seed_rng.randint(0, 2**31 - 1))
        artifact_rng = np.random.RandomState(baseline_seed + 7)
        artifact = maybe_generate_artifact(baseline, artifact_rng)
        protocol_rows = evaluate_artifact_across_protocol(artifact, protocol)

        for ep, metrics in enumerate(protocol_rows):
            ep_seed = int(metrics["comparison_seed"])
            row = {
                "baseline": baseline,
                "baseline_index": baseline_idx,
                "episode_index": ep,
                "episode_seed": ep_seed,
                "comparison_seed": ep_seed,
                "policy": policy_name,
                "policy_seed": ep_seed,
                "pack": pack,
                "max_turn": max_turn,
                "artifact": artifact,
            }
            row.update(metrics)
            rows.append(row)

    episodes_path = run_dir / "episodes.jsonl"
    write_jsonl(episodes_path, rows)

    summary = build_summary(rows)
    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    table_path = run_dir / "summary_table.csv"
    write_summary_table(summary, table_path)
    protocol_table_path = run_dir / "protocol_metrics.csv"
    write_protocol_table(summary, protocol_table_path)

    print("[OK] Baseline run complete.")
    print(f"[OK] Run dir: {run_dir}")
    print(f"[OK] Episodes: {episodes_path}")
    print(f"[OK] Summary: {summary_path}")
    print(f"[OK] Table: {table_path}")
    print(f"[OK] Protocol metrics: {protocol_table_path}")
    print("[NOTE] Generated baseline artifacts were injected into the environment for this run.")
    print("[NOTE] Baselines were evaluated on paired pack/policy/seed cells for simulator-backed comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
