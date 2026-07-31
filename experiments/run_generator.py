#!/usr/bin/env python3
"""Generator optimization runner with checkpoint and resume support."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import observed_simulation_metrics
from evaluation.protocol import EvaluationProtocol, evaluate_artifact_across_protocol, full_protocol_evaluation
from generator.schema import parse_artifact
from generator.search.random_search import (
    SearchState,
    candidate_record,
    load_checkpoint,
    mutate_candidate,
    sample_candidate,
    save_checkpoint,
    write_jsonl,
)
from sap_thesis_env.config import ProjectConfig
from sap_thesis_env.generated_artifacts import artifact_set_id, normalize_artifact_specs
from sap_thesis_env.run_metadata import write_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generator optimization.")
    parser.add_argument(
        "--config",
        default="configs/generator/phase4_generator.yaml",
        help="YAML config path.",
    )
    parser.add_argument(
        "--resume-run-dir",
        default="",
        help="Optional existing run dir to resume (contains checkpoint.json).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=-1,
        help="Optional cap for this invocation only. Useful for pause/resume testing.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config must deserialize to mapping.")
    return payload


def _init_run_dir(config_path: Path, cfg: ProjectConfig, resume_run_dir: str) -> Path:
    if resume_run_dir:
        run_dir = resolve_path(resume_run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run dir not found: {run_dir}")
        return run_dir
    metadata_path = write_run_metadata(config=cfg, project_root=PROJECT_ROOT, config_path=config_path)
    return metadata_path.parent


def _load_or_init_state(checkpoint_path: Path) -> SearchState:
    if checkpoint_path.exists():
        return load_checkpoint(checkpoint_path)
    return SearchState(
        next_iteration=0,
        accepted_count=0,
        rejected_count=0,
        search_best_record=None,
        best_record=None,
        best_safe_record=None,
        evaluated_count=0,
    )


def _build_protocol(raw_cfg: dict[str, Any], *, default_seed: int) -> EvaluationProtocol:
    packs = tuple(raw_cfg.get("evaluation_packs", ["StandardPack"]))
    policies = tuple(raw_cfg.get("evaluation_policies", ["heuristic"]))
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
        packs=tuple(str(pack) for pack in packs),
        policies=tuple(str(policy) for policy in policies),
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


def _load_or_create_reference(
    run_dir: Path,
    *,
    protocol: EvaluationProtocol,
    base_artifacts: tuple[Any, ...],
) -> dict[str, Any]:
    reference_path = run_dir / "reference_evaluation.json"
    if reference_path.exists():
        return json.loads(reference_path.read_text(encoding="utf-8"))

    rows = evaluate_artifact_across_protocol(base_artifacts, protocol)
    payload = {
        "protocol": _protocol_payload(protocol),
        "base_artifacts": [artifact.to_dict() for artifact in base_artifacts],
        "base_artifact_ids": [artifact.artifact_id for artifact in base_artifacts],
        "artifact_set_id": artifact_set_id(base_artifacts) if base_artifacts else "baseline",
        "rows": rows,
        "observed_simulation": observed_simulation_metrics(rows),
    }
    reference_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _evaluate_candidate_with_cache(
    record: dict[str, Any],
    *,
    cache_dir: Path,
    protocol: EvaluationProtocol,
    reference_rows: list[dict[str, Any]],
    base_artifacts: tuple[Any, ...],
) -> tuple[dict[str, Any], Path, bool]:
    spec = parse_artifact(record["payload"])
    artifact_bundle = tuple(base_artifacts) + (spec,)
    cache_path = cache_dir / f"{artifact_set_id(artifact_bundle)}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8")), cache_path, True

    evaluation = full_protocol_evaluation(artifact_bundle, protocol=protocol, reference_rows=reference_rows)
    cache_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8")
    return evaluation, cache_path, False


def _selection_objective(
    *,
    final_objective: float,
    proxy_objective: float,
    safety_penalty_value: float,
    proxy_guidance_weight: float,
    use_safety_penalty: bool,
) -> float:
    guided = float(final_objective + (proxy_guidance_weight * proxy_objective))
    if use_safety_penalty:
        guided -= safety_penalty_value
    return guided


def _random_baseline_objective(
    seed: int,
    n: int,
    *,
    cache_dir: Path,
    protocol: EvaluationProtocol,
    reference_rows: list[dict[str, Any]],
    base_artifacts: tuple[Any, ...],
    objective_mode: str,
    proxy_guidance_weight: float,
    require_injection_support: bool,
    use_safety_penalty: bool,
) -> dict[str, float | int]:
    objectives: list[float] = []
    safe_objectives: list[float] = []
    iteration = 0
    # Gather n records that pass fast filter.
    while len(objectives) < n and iteration < (n * 200):
        payload = sample_candidate(
            seed=seed + 999_983,
            iteration=iteration,
            supported_subset_only=require_injection_support,
        )
        rec = candidate_record(
            iteration=iteration,
            payload=payload,
            strict_pet_only=True,
            objective_mode=objective_mode,
            use_safety_penalty=use_safety_penalty,
            require_injection_support=require_injection_support,
        )
        if rec.get("fast_filter_pass"):
            evaluation, _cache_path, _cache_hit = _evaluate_candidate_with_cache(
                rec,
                cache_dir=cache_dir,
                protocol=protocol,
                reference_rows=reference_rows,
                base_artifacts=base_artifacts,
            )
            final_objective = float(evaluation.get("final_objective", -1.0))
            selection_objective = _selection_objective(
                final_objective=final_objective,
                proxy_objective=float(rec.get("objective", -1.0)),
                safety_penalty_value=float(rec.get("safety_penalty", 0.0)),
                proxy_guidance_weight=proxy_guidance_weight,
                use_safety_penalty=use_safety_penalty,
            )
            objectives.append(selection_objective)
            comparison = evaluation.get("comparison_to_reference", {})
            if comparison.get("safety_pass"):
                safe_objectives.append(selection_objective)
        iteration += 1
    if not objectives:
        return {"mean": -1.0, "std": 0.0, "safe_mean": -1.0, "safe_std": 0.0, "safe_count": 0}
    arr = np.array(objectives, dtype=float)
    safe_arr = np.array(safe_objectives, dtype=float) if safe_objectives else np.array([], dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "safe_mean": float(safe_arr.mean()) if safe_arr.size else -1.0,
        "safe_std": float(safe_arr.std()) if safe_arr.size else 0.0,
        "safe_count": int(safe_arr.size),
    }


def _save_summary(
    path: Path,
    *,
    state: SearchState,
    total_iterations: int,
    random_baseline: dict[str, float | int],
    protocol: EvaluationProtocol,
    base_artifacts: tuple[Any, ...],
    objective_mode: str,
    proxy_guidance_weight: float,
    use_safety_penalty: bool,
    iterative_generation: bool,
) -> None:
    best_objective = state.best_record["objective"] if state.best_record else None
    best_safe_objective = state.best_safe_record["objective"] if state.best_safe_record else None
    comparison_target = (
        float(random_baseline["safe_mean"])
        if int(random_baseline.get("safe_count", 0)) > 0
        else float(random_baseline["mean"])
    )
    summary = {
        "completed_iterations": state.next_iteration,
        "target_iterations": total_iterations,
        "accepted_count": state.accepted_count,
        "rejected_count": state.rejected_count,
        "evaluated_count": state.evaluated_count,
        "search_best_record": state.search_best_record,
        "best_record": state.best_record,
        "best_safe_record": state.best_safe_record,
        "random_baseline": random_baseline,
        "pilot_beats_random": bool(best_objective is not None and best_objective > float(random_baseline["mean"])),
        "pilot_safe_beats_random": bool(best_safe_objective is not None and best_safe_objective > comparison_target),
        "evaluation_protocol": _protocol_payload(protocol),
        "base_artifacts": [artifact.to_dict() for artifact in base_artifacts],
        "base_artifact_ids": [artifact.artifact_id for artifact in base_artifacts],
        "artifact_set_id": artifact_set_id(base_artifacts) if base_artifacts else "baseline",
        "objective_mode": objective_mode,
        "proxy_guidance_weight": proxy_guidance_weight,
        "use_safety_penalty": use_safety_penalty,
        "iterative_generation": iterative_generation,
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _save_leaderboard(
    path: Path,
    *,
    best_record: dict[str, Any] | None,
    best_safe_record: dict[str, Any] | None,
) -> None:
    fields = [
        "record_type",
        "artifact_id",
        "objective",
        "proxy_objective",
        "final_objective",
        "fairness_score",
        "diversity_retention",
        "stability_score",
        "safety_pass",
        "proxy_safety_penalty",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record_type, record in (("best_overall", best_record), ("best_safe", best_safe_record)):
            if record is None:
                continue
            final_metrics = record.get("final_metrics", {})
            comparison = final_metrics.get("comparison_to_reference", {})
            robustness = final_metrics.get("robustness", {})
            writer.writerow(
                {
                    "record_type": record_type,
                    "artifact_id": record.get("artifact_id"),
                    "objective": record.get("objective"),
                    "proxy_objective": record.get("proxy_objective"),
                    "final_objective": record.get("final_objective"),
                    "fairness_score": comparison.get("fairness_score"),
                    "diversity_retention": comparison.get("diversity_retention"),
                    "stability_score": robustness.get("stability_score"),
                    "safety_pass": comparison.get("safety_pass"),
                    "proxy_safety_penalty": record.get("safety_penalty"),
                }
            )


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    raw_cfg = load_yaml(config_path)
    cfg = ProjectConfig.from_dict(raw_cfg)

    total_iterations = int(raw_cfg.get("total_iterations", 300))
    checkpoint_every = int(raw_cfg.get("checkpoint_every", 25))
    baseline_samples = int(raw_cfg.get("random_baseline_samples", 80))
    strict_pet_only = bool(raw_cfg.get("strict_pet_only", True))
    objective_mode = str(raw_cfg.get("objective_mode", "full"))
    use_safety_penalty = bool(raw_cfg.get("use_safety_penalty", True))
    iterative_generation = bool(raw_cfg.get("iterative_generation", False))
    proxy_guidance_weight = float(raw_cfg.get("proxy_guidance_weight", 0.10))
    require_injection_support = bool(raw_cfg.get("require_injection_support", True))
    base_artifacts = normalize_artifact_specs(raw_cfg.get("base_artifacts"))
    protocol = _build_protocol(raw_cfg, default_seed=cfg.seed)

    run_dir = _init_run_dir(config_path=config_path, cfg=cfg, resume_run_dir=args.resume_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.json"
    candidates_path = run_dir / "candidates.jsonl"
    summary_path = run_dir / "summary.json"
    leaderboard_path = run_dir / "leaderboard.csv"
    cache_dir = run_dir / "evaluation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    reference_payload = _load_or_create_reference(run_dir, protocol=protocol, base_artifacts=base_artifacts)
    reference_rows = list(reference_payload.get("rows", []))

    state = _load_or_init_state(checkpoint_path)

    stop_at = total_iterations
    if args.max_iterations > 0:
        stop_at = min(stop_at, state.next_iteration + args.max_iterations)

    batch: list[dict[str, Any]] = []
    for i in range(state.next_iteration, stop_at):
        if iterative_generation and state.search_best_record is not None and (i % 3 != 0):
            payload = mutate_candidate(
                seed=cfg.seed,
                iteration=i,
                base_payload=state.search_best_record["payload"],
                supported_subset_only=require_injection_support,
            )
        else:
            payload = sample_candidate(
                seed=cfg.seed,
                iteration=i,
                supported_subset_only=require_injection_support,
            )

        rec = candidate_record(
            iteration=i,
            payload=payload,
            strict_pet_only=strict_pet_only,
            objective_mode=objective_mode,
            use_safety_penalty=use_safety_penalty,
            require_injection_support=require_injection_support,
        )
        rec["proxy_objective"] = rec.get("objective", -1.0)
        rec["proxy_metrics"] = rec.get("metrics", {})
        rec["final_metrics"] = None
        rec["final_objective"] = -1.0
        rec["final_protocol_pass"] = False
        rec["evaluation_cache_path"] = None

        if rec.get("fast_filter_pass"):
            evaluation, cache_path, _cache_hit = _evaluate_candidate_with_cache(
                rec,
                cache_dir=cache_dir,
                protocol=protocol,
                reference_rows=reference_rows,
                base_artifacts=base_artifacts,
            )
            rec["evaluation_cache_path"] = str(cache_path.relative_to(run_dir))
            rec["final_metrics"] = {
                "supported": evaluation.get("supported"),
                "support_error": evaluation.get("support_error"),
                "observed_simulation": evaluation.get("observed_simulation", {}),
                "comparison_to_reference": evaluation.get("comparison_to_reference", {}),
                "robustness": evaluation.get("robustness", {}),
                "efficiency": evaluation.get("efficiency", {}),
            }
            rec["final_objective"] = float(evaluation.get("final_objective", -1.0))
            rec["final_protocol_pass"] = bool(
                evaluation.get("supported")
                and rec["final_metrics"]["comparison_to_reference"].get("safety_pass")
            )
            rec["objective"] = _selection_objective(
                final_objective=float(rec["final_objective"]),
                proxy_objective=float(rec.get("proxy_objective", -1.0)),
                safety_penalty_value=float(rec.get("safety_penalty", 0.0)),
                proxy_guidance_weight=proxy_guidance_weight,
                use_safety_penalty=use_safety_penalty,
            )
            rec["accepted"] = True
            state = SearchState(
                next_iteration=i + 1,
                accepted_count=state.accepted_count + 1,
                rejected_count=state.rejected_count,
                search_best_record=state.search_best_record,
                best_record=state.best_record,
                best_safe_record=state.best_safe_record,
                evaluated_count=state.evaluated_count + 1,
            )
            if rec.get("proxy_objective", -1.0) > (state.search_best_record or {}).get("proxy_objective", float("-inf")):
                state = SearchState(
                    next_iteration=state.next_iteration,
                    accepted_count=state.accepted_count,
                    rejected_count=state.rejected_count,
                    search_best_record=rec,
                    best_record=state.best_record,
                    best_safe_record=state.best_safe_record,
                    evaluated_count=state.evaluated_count,
                )
            if rec.get("objective", -1.0) > (state.best_record or {}).get("objective", float("-inf")):
                state = SearchState(
                    next_iteration=state.next_iteration,
                    accepted_count=state.accepted_count,
                    rejected_count=state.rejected_count,
                    search_best_record=state.search_best_record,
                    best_record=rec,
                    best_safe_record=state.best_safe_record,
                    evaluated_count=state.evaluated_count,
                )
            if rec.get("final_protocol_pass") and rec.get("objective", -1.0) > (
                (state.best_safe_record or {}).get("objective", float("-inf"))
            ):
                state = SearchState(
                    next_iteration=state.next_iteration,
                    accepted_count=state.accepted_count,
                    rejected_count=state.rejected_count,
                    search_best_record=state.search_best_record,
                    best_record=state.best_record,
                    best_safe_record=rec,
                    evaluated_count=state.evaluated_count,
                )
        else:
            state = SearchState(
                next_iteration=i + 1,
                accepted_count=state.accepted_count,
                rejected_count=state.rejected_count + 1,
                search_best_record=state.search_best_record,
                best_record=state.best_record,
                best_safe_record=state.best_safe_record,
                evaluated_count=state.evaluated_count,
            )

        batch.append(rec)
        if (i + 1) % checkpoint_every == 0:
            write_jsonl(candidates_path, batch, mode="a")
            batch = []
            save_checkpoint(checkpoint_path, state)

    if batch:
        write_jsonl(candidates_path, batch, mode="a")
        save_checkpoint(checkpoint_path, state)

    random_baseline = _random_baseline_objective(
        seed=cfg.seed,
        n=baseline_samples,
        cache_dir=cache_dir,
        protocol=protocol,
        reference_rows=reference_rows,
        base_artifacts=base_artifacts,
        objective_mode=objective_mode,
        proxy_guidance_weight=proxy_guidance_weight,
        require_injection_support=require_injection_support,
        use_safety_penalty=use_safety_penalty,
    )
    _save_summary(
        summary_path,
        state=state,
        total_iterations=total_iterations,
        random_baseline=random_baseline,
        protocol=protocol,
        base_artifacts=base_artifacts,
        objective_mode=objective_mode,
        proxy_guidance_weight=proxy_guidance_weight,
        use_safety_penalty=use_safety_penalty,
        iterative_generation=iterative_generation,
    )
    _save_leaderboard(
        leaderboard_path,
        best_record=state.best_record,
        best_safe_record=state.best_safe_record,
    )

    print("[OK] Generator run complete.")
    print(f"[OK] Run dir: {run_dir}")
    print(f"[OK] Iterations completed: {state.next_iteration}/{total_iterations}")
    print(f"[OK] Accepted: {state.accepted_count} Rejected: {state.rejected_count}")
    if state.best_record:
        print(f"[OK] Best selection objective: {state.best_record['objective']:.6f}")
        print(f"[OK] Best final objective: {state.best_record['final_objective']:.6f}")
    if state.best_safe_record:
        print(f"[OK] Best safe selection objective: {state.best_safe_record['objective']:.6f}")
        print(f"[OK] Best safe final objective: {state.best_safe_record['final_objective']:.6f}")
    print(
        "[OK] Selection comparison: "
        f"best_selection={state.best_record['objective'] if state.best_record else None}, "
        f"best_safe_selection={state.best_safe_record['objective'] if state.best_safe_record else None}, "
        f"random_mean={float(random_baseline['mean']):.6f}, "
        f"random_safe_mean={float(random_baseline['safe_mean']):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
