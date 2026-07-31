#!/usr/bin/env python3
"""One-shot versus iterative balancing runner."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compare_to_reference, observed_simulation_metrics
from evaluation.protocol import EvaluationProtocol, evaluate_artifact_across_protocol, final_objective_from_metrics
from evaluation.robustness import summarize_robustness
from sap_thesis_env.config import ProjectConfig
from sap_thesis_env.generated_artifacts import artifact_set_id, normalize_artifact_specs
from sap_thesis_env.run_metadata import write_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-shot vs iterative balancing experiments.")
    parser.add_argument(
        "--config",
        default="configs/experiments/phase5_iterative_balancing.yaml",
        help="Path to iterative balancing YAML config.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config must deserialize to a mapping/object.")
    return payload


def _parse_run_dir(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("[OK] Run dir: "):
            return Path(line.split(": ", 1)[1].strip())
    raise RuntimeError("Could not parse generator run dir from run_generator output.")


def _build_protocol(raw_cfg: dict[str, Any], *, default_seed: int) -> EvaluationProtocol:
    packs = tuple(str(pack) for pack in raw_cfg.get("evaluation_packs", ["StandardPack"]))
    policies = tuple(str(policy) for policy in raw_cfg.get("evaluation_policies", ["heuristic"]))
    if not packs or not policies:
        raise ValueError("evaluation_packs and evaluation_policies must be non-empty.")

    seeds_raw = raw_cfg.get("evaluation_seeds")
    if seeds_raw is None:
        seed_count = int(raw_cfg.get("evaluation_seed_count", 3))
        comparison_seeds = tuple(default_seed + idx for idx in range(seed_count))
    else:
        comparison_seeds = tuple(int(seed) for seed in seeds_raw)
    if not comparison_seeds:
        raise ValueError("At least one evaluation seed is required.")

    return EvaluationProtocol(
        packs=packs,
        policies=policies,
        comparison_seeds=comparison_seeds,
        max_turn=int(raw_cfg.get("max_turn", 25)),
        max_steps_per_episode=int(raw_cfg.get("evaluation_max_steps_per_episode", 256)),
        strict_invalid_actions=bool(raw_cfg.get("evaluation_strict_invalid_actions", True)),
    )


def _protocol_payload(protocol: EvaluationProtocol) -> dict[str, Any]:
    return {
        "packs": list(protocol.packs),
        "policies": list(protocol.policies),
        "comparison_seeds": list(protocol.comparison_seeds),
        "max_turn": protocol.max_turn,
        "max_steps_per_episode": protocol.max_steps_per_episode,
        "strict_invalid_actions": protocol.strict_invalid_actions,
    }


def _run_generator_with_config(cfg: dict[str, Any]) -> dict[str, Any]:
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.safe_dump(cfg, tmp, sort_keys=False)
            tmp_path = Path(tmp.name)

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "run_generator.py"),
            "--config",
            str(tmp_path),
        ]
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True)
        run_dir = _parse_run_dir(proc.stdout)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        return {"run_dir": run_dir, "summary": summary, "stdout": proc.stdout}
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _load_cached_evaluation(run_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    cache_relpath = record.get("evaluation_cache_path")
    if not cache_relpath:
        raise ValueError("Selected generator record is missing evaluation_cache_path.")
    cache_path = run_dir / str(cache_relpath)
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _selected_artifact_payloads(records: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [artifact.to_dict() for artifact in records]


def _result_from_record(
    *,
    condition: str,
    step: int,
    run_dir: Path,
    record: dict[str, Any] | None,
    baseline_rows: list[dict[str, Any]],
    artifact_count: int,
    artifacts: tuple[Any, ...],
) -> dict[str, Any]:
    base = {
        "condition": condition,
        "step": step,
        "status": "no_safe_artifact_found" if record is None else "ok",
        "run_dir": str(run_dir),
        "artifact_count": artifact_count,
        "artifact_ids": [artifact.artifact_id for artifact in artifacts],
        "artifact_set_id": artifact_set_id(artifacts) if artifacts else "baseline",
        "artifacts": _selected_artifact_payloads(artifacts),
    }
    if record is None:
        return base

    evaluation = _load_cached_evaluation(run_dir, record)
    rows = list(evaluation.get("rows", []))
    overall_observed = observed_simulation_metrics(rows)
    overall_comparison = compare_to_reference(rows, baseline_rows)
    overall_robustness = summarize_robustness(rows, baseline_rows)
    overall_final_objective = final_objective_from_metrics(overall_comparison, overall_robustness)

    base.update(
        {
            "selected_artifact_id": record.get("artifact_id"),
            "selected_payload": record.get("payload"),
            "selection_objective": record.get("objective"),
            "selection_proxy_objective": record.get("proxy_objective"),
            "selection_final_objective": record.get("final_objective"),
            "selection_safety_penalty": record.get("safety_penalty"),
            "incremental_comparison": record.get("final_metrics", {}).get("comparison_to_reference", {}),
            "incremental_robustness": record.get("final_metrics", {}).get("robustness", {}),
            "overall_observed_simulation": overall_observed,
            "overall_comparison_to_baseline": overall_comparison,
            "overall_robustness_to_baseline": overall_robustness,
            "overall_final_objective": overall_final_objective,
        }
    )
    return base


def _generator_cfg(
    raw_cfg: dict[str, Any],
    *,
    experiment_name: str,
    seed: int,
    total_iterations: int,
    base_artifacts: tuple[Any, ...],
    generator_iterative_generation: bool,
) -> dict[str, Any]:
    cfg = dict(raw_cfg)
    cfg.update(
        {
            "project_name": raw_cfg.get("project_name", "sap-thesis-project"),
            "experiment_name": experiment_name,
            "seed": int(seed),
            "output_dir": raw_cfg.get("output_dir", "outputs"),
            "total_iterations": int(total_iterations),
            "base_artifacts": _selected_artifact_payloads(base_artifacts),
            "iterative_generation": bool(generator_iterative_generation),
        }
    )
    return cfg


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_step_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "condition",
        "step",
        "status",
        "run_dir",
        "artifact_count",
        "artifact_set_id",
        "selected_artifact_id",
        "selection_objective",
        "selection_proxy_objective",
        "selection_final_objective",
        "fairness_score",
        "diversity_retention",
        "stability_score",
        "safety_pass",
        "safety_violation_count",
        "mean_outcome_score",
        "mean_artifact_copies_final",
        "artifact_in_final_team_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            comparison = row.get("overall_comparison_to_baseline", {})
            robustness = row.get("overall_robustness_to_baseline", {})
            observed = row.get("overall_observed_simulation", {})
            writer.writerow(
                {
                    "condition": row.get("condition"),
                    "step": row.get("step"),
                    "status": row.get("status"),
                    "run_dir": row.get("run_dir"),
                    "artifact_count": row.get("artifact_count"),
                    "artifact_set_id": row.get("artifact_set_id"),
                    "selected_artifact_id": row.get("selected_artifact_id"),
                    "selection_objective": row.get("selection_objective"),
                    "selection_proxy_objective": row.get("selection_proxy_objective"),
                    "selection_final_objective": row.get("selection_final_objective"),
                    "fairness_score": comparison.get("fairness_score"),
                    "diversity_retention": comparison.get("diversity_retention"),
                    "stability_score": robustness.get("stability_score"),
                    "safety_pass": comparison.get("safety_pass"),
                    "safety_violation_count": comparison.get("safety_violation_count"),
                    "mean_outcome_score": observed.get("mean_outcome_score"),
                    "mean_artifact_copies_final": observed.get("mean_artifact_copies_final"),
                    "artifact_in_final_team_rate": observed.get("artifact_in_final_team_rate"),
                }
            )


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    raw_cfg = load_yaml(config_path)
    cfg = ProjectConfig.from_dict(raw_cfg)

    artifact_budget = int(raw_cfg.get("artifact_budget", 3))
    search_iterations_per_step = int(raw_cfg.get("search_iterations_per_step", 60))
    one_shot_total_iterations = int(
        raw_cfg.get("one_shot_total_iterations", artifact_budget * search_iterations_per_step)
    )
    generator_iterative_generation = bool(raw_cfg.get("generator_iterative_generation", True))
    initial_artifacts = normalize_artifact_specs(raw_cfg.get("base_artifacts"))
    protocol = _build_protocol(raw_cfg, default_seed=cfg.seed)

    metadata_path = write_run_metadata(config=cfg, project_root=PROJECT_ROOT, config_path=config_path)
    run_dir = metadata_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = evaluate_artifact_across_protocol(initial_artifacts, protocol)
    baseline_payload = {
        "protocol": _protocol_payload(protocol),
        "base_artifacts": _selected_artifact_payloads(initial_artifacts),
        "base_artifact_ids": [artifact.artifact_id for artifact in initial_artifacts],
        "artifact_set_id": artifact_set_id(initial_artifacts) if initial_artifacts else "baseline",
        "rows": baseline_rows,
        "observed_simulation": observed_simulation_metrics(baseline_rows),
    }
    _write_json(run_dir / "baseline_reference.json", baseline_payload)

    one_shot_cfg = _generator_cfg(
        raw_cfg,
        experiment_name=f"{cfg.experiment_name}_one_shot",
        seed=cfg.seed,
        total_iterations=one_shot_total_iterations,
        base_artifacts=initial_artifacts,
        generator_iterative_generation=generator_iterative_generation,
    )
    one_shot_subrun = _run_generator_with_config(one_shot_cfg)
    one_shot_summary = one_shot_subrun["summary"]
    one_shot_record = one_shot_summary.get("best_safe_record")
    one_shot_artifacts = (
        initial_artifacts
        if one_shot_record is None
        else initial_artifacts + normalize_artifact_specs(one_shot_record["payload"])
    )
    one_shot_result = _result_from_record(
        condition="one_shot",
        step=1,
        run_dir=one_shot_subrun["run_dir"],
        record=one_shot_record,
        baseline_rows=baseline_rows,
        artifact_count=len(one_shot_artifacts),
        artifacts=one_shot_artifacts,
    )
    one_shot_result["generator_summary"] = one_shot_summary

    iterative_steps: list[dict[str, Any]] = []
    current_artifacts = initial_artifacts
    stop_reason = "artifact_budget_reached"
    for step_idx in range(1, artifact_budget + 1):
        step_cfg = _generator_cfg(
            raw_cfg,
            experiment_name=f"{cfg.experiment_name}_iterative_step{step_idx}",
            seed=cfg.seed + (step_idx * 101),
            total_iterations=search_iterations_per_step,
            base_artifacts=current_artifacts,
            generator_iterative_generation=generator_iterative_generation,
        )
        step_subrun = _run_generator_with_config(step_cfg)
        step_summary = step_subrun["summary"]
        step_record = step_summary.get("best_safe_record")
        next_artifacts = (
            current_artifacts
            if step_record is None
            else current_artifacts + normalize_artifact_specs(step_record["payload"])
        )
        step_result = _result_from_record(
            condition="iterative",
            step=step_idx,
            run_dir=step_subrun["run_dir"],
            record=step_record,
            baseline_rows=baseline_rows,
            artifact_count=len(next_artifacts),
            artifacts=next_artifacts,
        )
        step_result["generator_summary"] = step_summary
        iterative_steps.append(step_result)
        if step_record is None:
            stop_reason = "no_safe_artifact_found"
            break
        current_artifacts = next_artifacts

    iterative_final = iterative_steps[-1] if iterative_steps else {
        "condition": "iterative",
        "step": 0,
        "status": "not_run",
        "artifact_count": len(initial_artifacts),
        "artifact_ids": [artifact.artifact_id for artifact in initial_artifacts],
        "artifact_set_id": artifact_set_id(initial_artifacts) if initial_artifacts else "baseline",
        "artifacts": _selected_artifact_payloads(initial_artifacts),
    }

    comparison = {
        "one_shot_status": one_shot_result.get("status"),
        "iterative_status": iterative_final.get("status"),
        "one_shot_fairness_score": one_shot_result.get("overall_comparison_to_baseline", {}).get("fairness_score"),
        "iterative_fairness_score": iterative_final.get("overall_comparison_to_baseline", {}).get("fairness_score"),
        "one_shot_diversity_retention": one_shot_result.get("overall_comparison_to_baseline", {}).get(
            "diversity_retention"
        ),
        "iterative_diversity_retention": iterative_final.get("overall_comparison_to_baseline", {}).get(
            "diversity_retention"
        ),
        "one_shot_stability_score": one_shot_result.get("overall_robustness_to_baseline", {}).get("stability_score"),
        "iterative_stability_score": iterative_final.get("overall_robustness_to_baseline", {}).get(
            "stability_score"
        ),
        "one_shot_safety_pass": one_shot_result.get("overall_comparison_to_baseline", {}).get("safety_pass"),
        "iterative_safety_pass": iterative_final.get("overall_comparison_to_baseline", {}).get("safety_pass"),
        "iterative_minus_one_shot_fairness": (
            None
            if one_shot_result.get("overall_comparison_to_baseline", {}).get("fairness_score") is None
            or iterative_final.get("overall_comparison_to_baseline", {}).get("fairness_score") is None
            else (
                float(iterative_final["overall_comparison_to_baseline"]["fairness_score"])
                - float(one_shot_result["overall_comparison_to_baseline"]["fairness_score"])
            )
        ),
        "iterative_steps_attempted": len(iterative_steps),
        "iterative_steps_accepted": sum(1 for row in iterative_steps if row.get("status") == "ok"),
        "iterative_steps_completed": sum(1 for row in iterative_steps if row.get("status") == "ok"),
        "iterative_stop_reason": stop_reason,
    }

    summary = {
        "artifact_budget": artifact_budget,
        "search_iterations_per_step": search_iterations_per_step,
        "one_shot_total_iterations": one_shot_total_iterations,
        "generator_iterative_generation": generator_iterative_generation,
        "protocol": _protocol_payload(protocol),
        "initial_artifacts": _selected_artifact_payloads(initial_artifacts),
        "initial_artifact_ids": [artifact.artifact_id for artifact in initial_artifacts],
        "baseline_reference": {
            "artifact_set_id": baseline_payload["artifact_set_id"],
            "observed_simulation": baseline_payload["observed_simulation"],
        },
        "one_shot": one_shot_result,
        "iterative": {
            "steps": iterative_steps,
            "final": iterative_final,
            "stop_reason": stop_reason,
        },
        "comparison": comparison,
    }

    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "iterative_trajectory.json", iterative_steps)
    _write_step_csv(run_dir / "comparison_rows.csv", [one_shot_result, iterative_final])
    _write_step_csv(run_dir / "iterative_trajectory.csv", iterative_steps)

    print("[OK] Iterative balancing run complete.")
    print(f"[OK] Run dir: {run_dir}")
    print(f"[OK] One-shot status: {one_shot_result.get('status')}")
    print(
        "[OK] Iterative status: "
        f"{iterative_final.get('status')} "
        f"(steps_attempted={comparison['iterative_steps_attempted']}, "
        f"steps_accepted={comparison['iterative_steps_accepted']}, "
        f"stop_reason={stop_reason})"
    )
    print(
        "[OK] Fairness comparison: "
        f"one_shot={comparison['one_shot_fairness_score']} "
        f"iterative={comparison['iterative_fairness_score']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
