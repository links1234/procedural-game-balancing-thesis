#!/usr/bin/env python3
"""Transfer and robustness runner.

Runs baseline evaluations across two packs and three policy families.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run transfer/robustness matrix.")
    parser.add_argument(
        "--config",
        default="configs/experiments/phase5_transfer.yaml",
        help="Transfer matrix config path.",
    )
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _parse_run_dir(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("[OK] Run dir: "):
            return Path(line.split(": ", 1)[1].strip())
    raise RuntimeError("Could not parse run dir from baseline output.")


def _run_baselines_with_config(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.safe_dump(cfg, tmp, sort_keys=False)
            cfg_path = Path(tmp.name)

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "run_baselines.py"),
            "--config",
            str(cfg_path),
        ]
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True)
        run_dir = _parse_run_dir(proc.stdout)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        return {"run_dir": str(run_dir), "summary": summary}
    finally:
        if cfg_path is not None and cfg_path.exists():
            cfg_path.unlink()


def main() -> int:
    args = parse_args()
    cfg_path = resolve(args.config)
    matrix_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(matrix_cfg, dict):
        raise ValueError("Transfer config must be a mapping.")

    packs = list(matrix_cfg.get("packs", ["StandardPack", "ExpansionPack1"]))
    policies = list(matrix_cfg.get("policies", ["random", "heuristic", "scripted_value"]))
    seeds = list(matrix_cfg.get("seeds", [20260301, 20260302, 20260303]))
    episodes_per_baseline = int(matrix_cfg.get("episodes_per_baseline", 12))
    max_steps_per_episode = int(matrix_cfg.get("max_steps_per_episode", 256))
    output_prefix = str(matrix_cfg.get("output_prefix", "phase5_transfer"))

    rows: list[dict[str, Any]] = []
    for pack_idx, pack in enumerate(packs):
        for policy_idx, policy in enumerate(policies):
            for seed_idx, seed in enumerate(seeds):
                cfg = {
                    "project_name": "sap-thesis-project",
                    "experiment_name": f"phase5_transfer_{pack}_{policy}",
                    "seed": int(seed) + (pack_idx * 1000) + (policy_idx * 100) + seed_idx,
                    "output_dir": "outputs",
                    "pack": pack,
                    "max_turn": 25,
                    "episodes_per_baseline": episodes_per_baseline,
                    "max_steps_per_episode": max_steps_per_episode,
                    "evaluation_policy": policy,
                    "baselines": ["no_generation", "random_generation", "heuristic_generation"],
                }
                result = _run_baselines_with_config(cfg)
                summary = result["summary"]
                baselines = summary.get("baselines", {})
                for baseline_name, metrics in baselines.items():
                    observed = metrics.get("observed_simulation", {})
                    comparison = metrics.get("comparison_to_no_generation") or {}
                    robustness = metrics.get("robustness_vs_no_generation") or {}
                    rows.append(
                        {
                            "seed": seed,
                            "pack": pack,
                            "policy": policy,
                            "baseline": baseline_name,
                            "run_dir": result["run_dir"],
                            "episodes": metrics.get("episodes"),
                            "mean_final_wins": metrics.get("mean_final_wins"),
                            "mean_final_lives": metrics.get("mean_final_lives"),
                            "mean_total_reward": metrics.get("mean_total_reward"),
                            "mean_steps": metrics.get("mean_steps"),
                            "mean_invalid_actions": metrics.get("mean_invalid_actions"),
                            "backend_error_rate": metrics.get("backend_error_rate"),
                            "mean_outcome_score": observed.get("mean_outcome_score"),
                            "normalized_team_entropy": observed.get("normalized_team_entropy"),
                            "artifact_in_final_team_rate": observed.get("artifact_in_final_team_rate"),
                            "fairness_score": comparison.get("fairness_score"),
                            "diversity_retention": comparison.get("diversity_retention"),
                            "safety_pass": comparison.get("safety_pass"),
                            "safety_violation_count": comparison.get("safety_violation_count"),
                            "stability_score": robustness.get("stability_score"),
                            "cell_effect_std": robustness.get("cell_effect_std"),
                        }
                    )

    out_dir = PROJECT_ROOT / "outputs" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{output_prefix}_summary.csv"
    json_path = out_dir / f"{output_prefix}_summary.json"

    fields = [
        "seed",
        "pack",
        "policy",
        "baseline",
        "run_dir",
        "episodes",
        "mean_final_wins",
        "mean_final_lives",
        "mean_total_reward",
        "mean_steps",
        "mean_invalid_actions",
        "backend_error_rate",
        "mean_outcome_score",
        "normalized_team_entropy",
        "artifact_in_final_team_rate",
        "fairness_score",
        "diversity_retention",
        "safety_pass",
        "safety_violation_count",
        "stability_score",
        "cell_effect_std",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    # Condition stats with intervals and effect sizes.
    stats_rows: list[dict[str, Any]] = []
    grouped_lives: dict[tuple[str, str, str], list[float]] = {}
    grouped_outcomes: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (row["pack"], row["policy"], row["baseline"])
        grouped_lives.setdefault(key, []).append(float(row["mean_final_lives"]))
        if row.get("mean_outcome_score") is not None:
            grouped_outcomes.setdefault(key, []).append(float(row["mean_outcome_score"]))

    for key, vals in grouped_lives.items():
        pack, policy, baseline = key
        arr_lives = np.array(vals, dtype=float)
        mean_lives = float(arr_lives.mean())
        std_lives = float(arr_lives.std(ddof=1)) if len(arr_lives) > 1 else 0.0
        ci_lives = float(1.96 * std_lives / np.sqrt(len(arr_lives))) if len(arr_lives) > 1 else 0.0
        arr_outcomes = np.array(grouped_outcomes.get(key, []), dtype=float)
        mean_outcome = float(arr_outcomes.mean()) if arr_outcomes.size else 0.0
        std_outcome = float(arr_outcomes.std(ddof=1)) if arr_outcomes.size > 1 else 0.0
        ci_outcome = float(1.96 * std_outcome / np.sqrt(len(arr_outcomes))) if arr_outcomes.size > 1 else 0.0
        stats_rows.append(
            {
                "pack": pack,
                "policy": policy,
                "baseline": baseline,
                "n": len(arr_lives),
                "mean_lives": mean_lives,
                "std_lives": std_lives,
                "ci95_lives_low": mean_lives - ci_lives,
                "ci95_lives_high": mean_lives + ci_lives,
                "mean_outcome_score": mean_outcome,
                "std_outcome_score": std_outcome,
                "ci95_outcome_low": mean_outcome - ci_outcome,
                "ci95_outcome_high": mean_outcome + ci_outcome,
            }
        )

    stats_csv = out_dir / f"{output_prefix}_condition_stats.csv"
    stats_fields = [
        "pack",
        "policy",
        "baseline",
        "n",
        "mean_lives",
        "std_lives",
        "ci95_lives_low",
        "ci95_lives_high",
        "mean_outcome_score",
        "std_outcome_score",
        "ci95_outcome_low",
        "ci95_outcome_high",
    ]
    with stats_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=stats_fields)
        writer.writeheader()
        writer.writerows(sorted(stats_rows, key=lambda r: (r["pack"], r["policy"], r["baseline"])))

    # Pack transfer effect sizes: ExpansionPack1 vs StandardPack for each policy+baseline.
    def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2 or len(b) < 2:
            return 0.0
        s1 = float(np.var(a, ddof=1))
        s2 = float(np.var(b, ddof=1))
        pooled = np.sqrt(((len(a) - 1) * s1 + (len(b) - 1) * s2) / (len(a) + len(b) - 2))
        if pooled == 0:
            return 0.0
        return float((float(np.mean(a)) - float(np.mean(b))) / pooled)

    effects_rows: list[dict[str, Any]] = []
    for policy in policies:
        for baseline in ["no_generation", "random_generation", "heuristic_generation"]:
            std_lives = np.array(grouped_lives.get(("StandardPack", policy, baseline), []), dtype=float)
            exp_lives = np.array(grouped_lives.get(("ExpansionPack1", policy, baseline), []), dtype=float)
            std_outcomes = np.array(grouped_outcomes.get(("StandardPack", policy, baseline), []), dtype=float)
            exp_outcomes = np.array(grouped_outcomes.get(("ExpansionPack1", policy, baseline), []), dtype=float)
            if len(std_lives) == 0 or len(exp_lives) == 0:
                continue
            effects_rows.append(
                {
                    "policy": policy,
                    "baseline": baseline,
                    "mean_lives_standard": float(std_lives.mean()),
                    "mean_lives_expansion": float(exp_lives.mean()),
                    "mean_lives_difference_exp_minus_std": float(exp_lives.mean() - std_lives.mean()),
                    "effect_size_d_lives_exp_vs_std": _cohen_d(exp_lives, std_lives),
                    "mean_outcome_standard": float(std_outcomes.mean()) if len(std_outcomes) else 0.0,
                    "mean_outcome_expansion": float(exp_outcomes.mean()) if len(exp_outcomes) else 0.0,
                    "mean_outcome_difference_exp_minus_std": (
                        float(exp_outcomes.mean() - std_outcomes.mean())
                        if len(std_outcomes) and len(exp_outcomes)
                        else 0.0
                    ),
                    "effect_size_d_outcome_exp_vs_std": (
                        _cohen_d(exp_outcomes, std_outcomes)
                        if len(std_outcomes) and len(exp_outcomes)
                        else 0.0
                    ),
                }
            )

    effects_csv = out_dir / f"{output_prefix}_pack_effects.csv"
    effect_fields = [
        "policy",
        "baseline",
        "mean_lives_standard",
        "mean_lives_expansion",
        "mean_lives_difference_exp_minus_std",
        "effect_size_d_lives_exp_vs_std",
        "mean_outcome_standard",
        "mean_outcome_expansion",
        "mean_outcome_difference_exp_minus_std",
        "effect_size_d_outcome_exp_vs_std",
    ]
    with effects_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=effect_fields)
        writer.writeheader()
        writer.writerows(sorted(effects_rows, key=lambda r: (r["policy"], r["baseline"])))

    print("[OK] Transfer/robustness matrix complete.")
    print(f"[OK] Rows: {len(rows)}")
    print(f"[OK] CSV: {csv_path}")
    print(f"[OK] JSON: {json_path}")
    print(f"[OK] Condition stats: {stats_csv}")
    print(f"[OK] Pack effects: {effects_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
