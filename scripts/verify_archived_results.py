#!/usr/bin/env python3
"""Recompute the published table from the archived episode snapshot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compare_to_reference
from evaluation.protocol import final_objective_from_metrics
from evaluation.robustness import summarize_robustness


CONDITION_KEYS = {
    "Heuristic baseline": "heuristic_baseline",
    "Generator best safe": "generator_best_safe",
    "One-shot selected": "one_shot_selected",
    "Iterative step 1": "iterative_step_1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the compact result table from archived episode outcomes."
    )
    parser.add_argument(
        "--snapshot",
        default="results/archived_evaluations.json",
        help="Archived episode snapshot.",
    )
    parser.add_argument(
        "--table",
        default="results/final_condition_comparison.csv",
        help="Published comparison table.",
    )
    parser.add_argument(
        "--expected-reference-errors",
        type=int,
        default=1,
        help="Expected reference-row backend error count.",
    )
    return parser.parse_args()


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def assert_close(label: str, actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: actual={actual!r}, expected={expected!r}")


def main() -> int:
    args = parse_args()
    snapshot = json.loads(resolve(args.snapshot).read_text(encoding="utf-8"))
    evaluations = snapshot["evaluations"]
    reference_rows = evaluations["reference"]["rows"]

    with resolve(args.table).open("r", encoding="utf-8", newline="") as handle:
        table_rows = {row["condition"]: row for row in csv.DictReader(handle)}

    for condition, evaluation_key in CONDITION_KEYS.items():
        candidate_rows = evaluations[evaluation_key]["rows"]
        comparison = compare_to_reference(candidate_rows, reference_rows)
        robustness = summarize_robustness(candidate_rows, reference_rows)
        final_objective = final_objective_from_metrics(comparison, robustness)
        expected = table_rows[condition]

        assert_close(f"{condition} final_objective", final_objective, float(expected["final_objective"]))
        assert_close(
            f"{condition} fairness_score",
            float(comparison["fairness_score"]),
            float(expected["fairness_score"]),
        )
        assert_close(
            f"{condition} diversity_retention",
            float(comparison["diversity_retention"]),
            float(expected["diversity_retention"]),
        )
        assert_close(
            f"{condition} stability_score",
            float(robustness["stability_score"]),
            float(expected["stability_score"]),
        )
        print(f"[OK] {condition}: final_objective={final_objective:.4f}")

    reference_errors = sum(1 for row in reference_rows if row.get("backend_error"))
    if reference_errors != args.expected_reference_errors:
        raise AssertionError(
            "Expected "
            f"{args.expected_reference_errors} reference backend errors, "
            f"found {reference_errors}"
        )

    print(f"[OK] Reference backend errors: {reference_errors}/{len(reference_rows)}")
    print("[OK] Published result table matches the supplied episode outcomes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
