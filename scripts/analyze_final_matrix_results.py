#!/usr/bin/env python3
"""Posthoc analysis for the thesis final matrix run.

This script turns the final matrix outputs into durable comparison artifacts that
are useful for thesis writing:

- apples-to-apples evaluation of the fixed heuristic baseline artifact under the
  generator's final protocol,
- cross-pack evaluation of the selected generated artifacts,
- grouped transfer summary means for quick result inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.protocol import EvaluationProtocol, evaluate_artifact_across_protocol, full_protocol_evaluation
from experiments.run_baselines import build_heuristic_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a completed final matrix run.")
    parser.add_argument(
        "--final-matrix-run-dir",
        default="",
        help="Optional final matrix run dir. Defaults to the latest run under outputs/raw/final_matrix/.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/processed",
        help="Output directory for JSON/CSV analysis artifacts.",
    )
    parser.add_argument(
        "--prefix",
        default="final_matrix_posthoc",
        help="Prefix for generated analysis files.",
    )
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def latest_run_dir(base_dir: Path) -> Path:
    candidates = [path for path in base_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {base_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_from_payload(payload: dict[str, Any]) -> EvaluationProtocol:
    return EvaluationProtocol(
        packs=tuple(str(pack) for pack in payload.get("packs", [])),
        policies=tuple(str(policy) for policy in payload.get("policies", [])),
        comparison_seeds=tuple(int(seed) for seed in payload.get("comparison_seeds", [])),
        max_turn=int(payload.get("max_turn", 25)),
        max_steps_per_episode=int(payload.get("max_steps_per_episode", 256)),
        strict_invalid_actions=bool(payload.get("strict_invalid_actions", True)),
    )


def mean_by_group(rows: list[dict[str, Any]], group_fields: tuple[str, ...], value_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        metrics = grouped.setdefault(key, {field: [] for field in value_fields})
        for field in value_fields:
            value = row.get(field)
            if value in (None, ""):
                continue
            metrics[field].append(float(value))

    output: list[dict[str, Any]] = []
    for key, metrics in sorted(grouped.items()):
        grouped_row = {field: key[idx] for idx, field in enumerate(group_fields)}
        for field in value_fields:
            values = metrics[field]
            grouped_row[field] = float(sum(values) / len(values)) if values else None
        output.append(grouped_row)
    return output


def flatten_cross_pack_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact_label, artifact_result in sorted(results.items()):
        for pack, pack_result in sorted(artifact_result["packs"].items()):
            comparison = pack_result.get("comparison_to_reference", {})
            robustness = pack_result.get("robustness", {})
            observed = pack_result.get("observed_simulation", {})
            rows.append(
                {
                    "artifact_label": artifact_label,
                    "artifact_id": artifact_result.get("artifact_id"),
                    "pack": pack,
                    "final_objective": pack_result.get("final_objective"),
                    "fairness_score": comparison.get("fairness_score"),
                    "diversity_retention": comparison.get("diversity_retention"),
                    "mean_outcome_delta": comparison.get("mean_outcome_delta"),
                    "safety_pass": comparison.get("safety_pass"),
                    "stability_score": robustness.get("stability_score"),
                    "mean_outcome_score": observed.get("mean_outcome_score"),
                    "backend_error_rate": observed.get("backend_error_rate"),
                    "artifact_in_final_team_rate": observed.get("artifact_in_final_team_rate"),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.final_matrix_run_dir:
        final_matrix_run_dir = resolve(args.final_matrix_run_dir)
    else:
        final_matrix_run_dir = latest_run_dir(PROJECT_ROOT / "outputs" / "raw" / "final_matrix")

    top_summary = load_json(final_matrix_run_dir / "summary.json")
    generator_run_dir = Path(top_summary["steps"]["generator"]["outputs"]["run_dir"])
    iterative_run_dir = Path(top_summary["steps"]["iterative_balancing"]["outputs"]["run_dir"])
    transfer_csv_path = Path(top_summary["steps"]["transfer"]["outputs"]["csv"])

    generator_summary = load_json(generator_run_dir / "summary.json")
    iterative_summary = load_json(iterative_run_dir / "summary.json")
    generator_reference = load_json(generator_run_dir / "reference_evaluation.json")

    generator_protocol = protocol_from_payload(generator_summary["evaluation_protocol"])
    heuristic_eval = full_protocol_evaluation(
        build_heuristic_artifact(),
        protocol=generator_protocol,
        reference_rows=generator_reference["rows"],
    )

    transfer_rows: list[dict[str, Any]] = []
    with transfer_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        transfer_rows.extend(reader)

    transfer_group_means = mean_by_group(
        transfer_rows,
        group_fields=("pack", "policy", "baseline"),
        value_fields=(
            "mean_outcome_score",
            "fairness_score",
            "diversity_retention",
            "stability_score",
            "backend_error_rate",
            "artifact_in_final_team_rate",
        ),
    )

    cross_pack_protocol = EvaluationProtocol(
        packs=("StandardPack", "ExpansionPack1"),
        policies=("random", "heuristic", "scripted_value"),
        comparison_seeds=(20260401, 20260402, 20260403, 20260404, 20260405),
        max_turn=25,
        max_steps_per_episode=160,
        strict_invalid_actions=True,
    )
    cross_pack_results: dict[str, Any] = {}
    artifact_sets = {
        "generator_best_safe": {
            "artifact_id": generator_summary["best_safe_record"]["artifact_id"],
            "artifacts": [generator_summary["best_safe_record"]["payload"]],
        },
        "one_shot_selected": {
            "artifact_id": iterative_summary["one_shot"]["selected_artifact_id"],
            "artifacts": list(iterative_summary["one_shot"]["artifacts"]),
        },
        "iterative_final_set": {
            "artifact_id": ",".join(iterative_summary["iterative"]["final"]["artifact_ids"]),
            "artifacts": list(iterative_summary["iterative"]["final"]["artifacts"]),
        },
    }

    for artifact_label, artifact_payload in artifact_sets.items():
        pack_results: dict[str, Any] = {}
        for pack in cross_pack_protocol.packs:
            pack_protocol = replace(cross_pack_protocol, packs=(pack,))
            reference_rows = evaluate_artifact_across_protocol(None, pack_protocol)
            evaluation = full_protocol_evaluation(
                artifact_payload["artifacts"],
                protocol=pack_protocol,
                reference_rows=reference_rows,
            )
            pack_results[pack] = {
                "final_objective": evaluation.get("final_objective"),
                "comparison_to_reference": evaluation.get("comparison_to_reference", {}),
                "robustness": evaluation.get("robustness", {}),
                "observed_simulation": evaluation.get("observed_simulation", {}),
            }
        cross_pack_results[artifact_label] = {
            "artifact_id": artifact_payload["artifact_id"],
            "artifacts": artifact_payload["artifacts"],
            "packs": pack_results,
        }

    heuristic_comparison = {
        "candidate": "heuristic_generation_baseline",
        "artifact": build_heuristic_artifact(),
        "protocol": generator_summary["evaluation_protocol"],
        "final_objective": heuristic_eval.get("final_objective"),
        "comparison_to_reference": heuristic_eval.get("comparison_to_reference", {}),
        "robustness": heuristic_eval.get("robustness", {}),
        "observed_simulation": heuristic_eval.get("observed_simulation", {}),
        "generator_best_safe_final_objective": generator_summary["best_safe_record"]["final_objective"],
        "generator_best_safe_selection_objective": generator_summary["best_safe_record"]["objective"],
        "generator_best_safe_delta_vs_heuristic_final_objective": (
            float(generator_summary["best_safe_record"]["final_objective"]) - float(heuristic_eval.get("final_objective", 0.0))
        ),
    }

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "final_matrix_run_dir": str(final_matrix_run_dir),
        "generator_run_dir": str(generator_run_dir),
        "iterative_run_dir": str(iterative_run_dir),
        "transfer_csv": str(transfer_csv_path),
        "heuristic_generator_protocol_comparison": heuristic_comparison,
        "cross_pack_selected_artifact_results": cross_pack_results,
        "transfer_group_means": transfer_group_means,
    }

    json_path = out_dir / f"{args.prefix}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    heuristic_csv_path = out_dir / f"{args.prefix}_heuristic_generator_protocol.csv"
    write_csv(
        heuristic_csv_path,
        [
            {
                "candidate": heuristic_comparison["candidate"],
                "final_objective": heuristic_comparison["final_objective"],
                "fairness_score": heuristic_comparison["comparison_to_reference"].get("fairness_score"),
                "diversity_retention": heuristic_comparison["comparison_to_reference"].get("diversity_retention"),
                "mean_outcome_delta": heuristic_comparison["comparison_to_reference"].get("mean_outcome_delta"),
                "safety_pass": heuristic_comparison["comparison_to_reference"].get("safety_pass"),
                "stability_score": heuristic_comparison["robustness"].get("stability_score"),
                "generator_best_safe_final_objective": heuristic_comparison["generator_best_safe_final_objective"],
                "generator_best_safe_delta_vs_heuristic_final_objective": heuristic_comparison[
                    "generator_best_safe_delta_vs_heuristic_final_objective"
                ],
            }
        ],
    )

    cross_pack_csv_path = out_dir / f"{args.prefix}_cross_pack_selected_artifacts.csv"
    write_csv(cross_pack_csv_path, flatten_cross_pack_results(cross_pack_results))

    transfer_means_csv_path = out_dir / f"{args.prefix}_transfer_group_means.csv"
    write_csv(transfer_means_csv_path, transfer_group_means)

    print(f"[OK] Final matrix run: {final_matrix_run_dir}")
    print(f"[OK] JSON: {json_path}")
    print(f"[OK] Heuristic protocol CSV: {heuristic_csv_path}")
    print(f"[OK] Cross-pack CSV: {cross_pack_csv_path}")
    print(f"[OK] Transfer means CSV: {transfer_means_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
