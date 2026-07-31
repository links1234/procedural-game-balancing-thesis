# Result provenance

This directory contains a compact, portable subset of the submitted thesis evidence and current-code validation.
Large candidate logs, evaluation caches, temporary outputs, and machine-specific run paths are intentionally excluded.

The archived final matrix recorded source revision `6f45a2e`, run ID `final_matrix_20260330T060255Z_679b03dd`, and eight comparison seeds from `20260401` through `20260408`.
The archived snapshot and submitted CSVs match the tables and figure in the preserved thesis report; the corrected validation CSV is labeled separately.

## Files

- [`archived_evaluations.json`](archived_evaluations.json) contains the 24 reference episode rows and the 24 rows for each of the four thesis-facing candidate conditions.
- [`final_condition_comparison.csv`](final_condition_comparison.csv) contains the four-condition table plotted in the report.
- [`heuristic_generator_protocol.csv`](heuristic_generator_protocol.csv) records the direct fixed-heuristic versus dedicated-generator comparison.
- [`cross_pack_selected_artifacts.csv`](cross_pack_selected_artifacts.csv) contains the post hoc selected-artifact results for two packs.
- [`corrected_cross_pack_validation.csv`](corrected_cross_pack_validation.csv) records the same post hoc command's output after the empty-shop-slot fix.

The archived rows retain the one no-generation reference backend error observed in the submitted run.
All four candidate conditions in the main table have zero candidate-side backend errors and passed the configured safety gate.

Run the following command to recompute fairness, diversity retention, stability, and the final objective from the archived episode rows and compare them with the tracked table:

```bash
python scripts/verify_archived_results.py
```

The public code fixes the empty-shop-slot action that caused the archived reference error.
A corrected full-matrix validation reproduced all four StandardPack condition values exactly with zero reference errors.
The StandardPack rows in the post hoc cross-pack table also reproduced, while the three ExpansionPack1 rows changed because the corrected random-policy reference followed a different trajectory.
Those ExpansionPack1 candidate rows retained the same mean outcome scores, zero backend errors, and passing safety status.
The submitted and corrected CSVs are both retained so that the reproducibility boundary is explicit.
