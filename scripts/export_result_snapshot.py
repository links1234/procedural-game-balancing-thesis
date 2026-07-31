#!/usr/bin/env python3
"""Export a portable snapshot of the thesis-facing episode outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.protocol import full_protocol_evaluation
from experiments.run_baselines import build_heuristic_artifact
from scripts.analyze_final_matrix_results import load_json, protocol_from_payload


ROW_FIELDS = (
    "pack",
    "policy",
    "comparison_seed",
    "final_wins",
    "final_lives",
    "final_team_signature",
    "artifact_in_final_team",
    "artifact_copies_final",
    "invalid_action_count",
    "backend_error",
    "backend_error_message",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the episode rows needed to verify the thesis-facing results."
    )
    parser.add_argument(
        "--final-matrix-run-dir",
        required=True,
        help="Completed top-level final matrix run directory.",
    )
    parser.add_argument(
        "--out",
        default="results/archived_evaluations.json",
        help="Portable JSON snapshot to write.",
    )
    return parser.parse_args()


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def slim_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {field: row.get(field) for field in ROW_FIELDS}
        for row in rows
    ]


def load_cached_evaluation(run_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    cache_path = run_dir / str(record["evaluation_cache_path"])
    return load_json(cache_path)


def evaluation_payload(
    evaluation: dict[str, Any],
    *,
    artifact: dict[str, Any] | None,
    artifact_id: str | None,
) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "artifact_id": artifact_id,
        "rows": slim_rows(list(evaluation["rows"])),
    }


def main() -> int:
    args = parse_args()
    matrix_run_dir = resolve(args.final_matrix_run_dir)
    top_summary = load_json(matrix_run_dir / "summary.json")
    matrix_metadata = load_json(matrix_run_dir / "metadata.json")

    generator_run_dir = Path(top_summary["steps"]["generator"]["outputs"]["run_dir"])
    iterative_run_dir = Path(top_summary["steps"]["iterative_balancing"]["outputs"]["run_dir"])
    generator_summary = load_json(generator_run_dir / "summary.json")
    iterative_summary = load_json(iterative_run_dir / "summary.json")
    reference = load_json(generator_run_dir / "reference_evaluation.json")

    protocol = protocol_from_payload(generator_summary["evaluation_protocol"])
    heuristic_artifact = build_heuristic_artifact()
    heuristic = full_protocol_evaluation(
        heuristic_artifact,
        protocol=protocol,
        reference_rows=reference["rows"],
    )

    generator_record = generator_summary["best_safe_record"]
    generator = load_cached_evaluation(generator_run_dir, generator_record)

    one_shot = iterative_summary["one_shot"]
    one_shot_record = one_shot["generator_summary"]["best_safe_record"]
    one_shot_evaluation = load_cached_evaluation(Path(one_shot["run_dir"]), one_shot_record)

    iterative_step = iterative_summary["iterative"]["steps"][0]
    iterative_record = iterative_step["generator_summary"]["best_safe_record"]
    iterative_evaluation = load_cached_evaluation(Path(iterative_step["run_dir"]), iterative_record)

    payload = {
        "source": {
            "matrix_run_id": matrix_metadata["run_id"],
            "timestamp_utc": matrix_metadata["timestamp_utc"],
            "git_hash": matrix_metadata["git_hash"],
            "protocol": generator_summary["evaluation_protocol"],
        },
        "evaluations": {
            "reference": evaluation_payload(reference, artifact=None, artifact_id=None),
            "heuristic_baseline": evaluation_payload(
                heuristic,
                artifact=heuristic_artifact,
                artifact_id=heuristic["artifact_id"],
            ),
            "generator_best_safe": evaluation_payload(
                generator,
                artifact=generator_record["payload"],
                artifact_id=generator_record["artifact_id"],
            ),
            "one_shot_selected": evaluation_payload(
                one_shot_evaluation,
                artifact=one_shot_record["payload"],
                artifact_id=one_shot_record["artifact_id"],
            ),
            "iterative_step_1": evaluation_payload(
                iterative_evaluation,
                artifact=iterative_record["payload"],
                artifact_id=iterative_record["artifact_id"],
            ),
        },
    }

    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OK] Wrote portable result snapshot: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
