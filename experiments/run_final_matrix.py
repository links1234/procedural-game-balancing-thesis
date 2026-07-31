#!/usr/bin/env python3
"""Pilot/final matrix orchestrator for the thesis experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sap_thesis_env.config import ProjectConfig
from sap_thesis_env.run_metadata import write_run_metadata


SCRIPT_MAP = {
    "baselines": PROJECT_ROOT / "experiments" / "run_baselines.py",
    "generator": PROJECT_ROOT / "experiments" / "run_generator.py",
    "iterative_balancing": PROJECT_ROOT / "experiments" / "run_iterative_balancing.py",
    "transfer": PROJECT_ROOT / "experiments" / "run_transfer.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the thesis final pilot or final matrix.")
    parser.add_argument(
        "--config",
        default="configs/experiments/final_pilot.yaml",
        help="Path to the top-level matrix YAML config.",
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


def _load_section_cfg(section: dict[str, Any]) -> dict[str, Any]:
    base: dict[str, Any] = {}
    config_path = section.get("config_path")
    if config_path:
        base = load_yaml(resolve_path(str(config_path)))
    overrides = section.get("config", {})
    if overrides and not isinstance(overrides, dict):
        raise ValueError("Section config overrides must be a mapping.")
    base.update(overrides or {})
    return base


def _parse_keyed_path(stdout: str, label: str) -> str | None:
    prefix = f"[OK] {label}: "
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return line.split(": ", 1)[1].strip()
    return None


def _run_script(step_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.safe_dump(cfg, tmp, sort_keys=False)
            tmp_path = Path(tmp.name)

        cmd = [
            sys.executable,
            str(SCRIPT_MAP[step_name]),
            "--config",
            str(tmp_path),
        ]
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True)
        stdout = proc.stdout

        outputs: dict[str, Any]
        if step_name in {"baselines", "generator", "iterative_balancing"}:
            run_dir = _parse_keyed_path(stdout, "Run dir")
            if run_dir is None:
                raise RuntimeError(f"Could not parse run dir from {step_name} output.")
            outputs = {"run_dir": run_dir}
        elif step_name == "transfer":
            outputs = {
                "csv": _parse_keyed_path(stdout, "CSV"),
                "json": _parse_keyed_path(stdout, "JSON"),
                "condition_stats_csv": _parse_keyed_path(stdout, "Condition stats"),
                "pack_effects_csv": _parse_keyed_path(stdout, "Pack effects"),
            }
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported step_name='{step_name}'")

        return {
            "step_name": step_name,
            "command": cmd,
            "config": cfg,
            "stdout": stdout,
            "outputs": outputs,
        }
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_experiment_index(path: Path, run_payload: dict[str, Any]) -> None:
    lines = [
        "# Experiment Index",
        "",
        f"- Top-level run dir: `{run_payload['run_dir']}`",
        f"- Mode: `{run_payload['mode']}`",
        f"- Config: `{run_payload['config_path']}`",
        "",
        "## Research Question Coverage",
        "",
        "- Primary question: `baselines` + `generator`",
        "- One-shot vs iterative: `iterative_balancing`",
        "- Multi-policy robustness / transfer: `transfer`",
        "",
        "## Executed Steps",
        "",
    ]

    for step_name, result in run_payload["steps"].items():
        lines.extend(
            [
                f"### {step_name}",
                "",
                f"- Script: `{SCRIPT_MAP[step_name]}`",
                f"- Outputs: `{json.dumps(result['outputs'], sort_keys=True)}`",
                "",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_command_list(path: Path, config_path: Path) -> None:
    display_config = _portable_path(config_path)
    lines = [
        "# Final Command List",
        "",
        "Run the full configured matrix:",
        "",
        f"```bash\npython experiments/run_final_matrix.py --config {display_config}\n```",
        "",
        "Underlying sub-runners are launched automatically from the nested config sections.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stdout_value(result: dict[str, Any], label: str) -> str | None:
    stdout = str(result.get("stdout", ""))
    return _parse_keyed_path(stdout, label)


def _write_run_summary_note(path: Path, run_payload: dict[str, Any]) -> None:
    lines = [
        f"# {'Pilot' if run_payload['mode'] == 'pilot' else 'Matrix'} Summary",
        "",
        f"- Mode: `{run_payload['mode']}`",
        f"- Top-level run dir: `{run_payload['run_dir']}`",
        "",
        "## Step Outcomes",
        "",
    ]

    baselines = run_payload["steps"].get("baselines")
    if baselines:
        lines.append(f"- Baselines: completed at `{baselines['outputs']['run_dir']}`")

    generator = run_payload["steps"].get("generator")
    if generator:
        lines.append(f"- Generator: completed at `{generator['outputs']['run_dir']}`")
        best_safe = _stdout_value(generator, "Best safe final objective")
        if best_safe is not None:
            lines.append(f"- Generator best safe final objective: `{best_safe}`")

    iterative = run_payload["steps"].get("iterative_balancing")
    if iterative:
        lines.append(f"- One-shot vs iterative: completed at `{iterative['outputs']['run_dir']}`")
        iterative_status = _stdout_value(iterative, "Iterative status")
        one_shot_status = _stdout_value(iterative, "One-shot status")
        if one_shot_status is not None:
            lines.append(f"- One-shot status: `{one_shot_status}`")
        if iterative_status is not None:
            lines.append(f"- Iterative status: `{iterative_status}`")
        if iterative_status is not None and "no_safe_artifact_found" in iterative_status:
            lines.append(
                "- Iterative note: reinjection hit the safety gate before exhausting the artifact budget. "
                "This is a search-budget/acceptance outcome, not a simulator crash."
            )

    transfer = run_payload["steps"].get("transfer")
    if transfer:
        lines.append(f"- Transfer/robustness: completed with summary `{transfer['outputs']['csv']}`")

    completion_note = (
        "- A clean pilot pass means the pipeline is ready for the larger configured matrix."
        if run_payload["mode"] == "pilot"
        else f"- All configured {run_payload['mode']} stages completed."
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            completion_note,
            "- Any non-`ok` iterative status should be interpreted first as an empirical search result, not automatically as an implementation defect.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _detect_gpu_names() -> list[str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _write_reproducibility_appendix(path: Path, run_payload: dict[str, Any], hardware_note: str) -> None:
    gpu_names = _detect_gpu_names()
    lines = [
        "# Reproducibility Appendix",
        "",
        f"- Mode: `{run_payload['mode']}`",
        f"- Top-level run dir: `{run_payload['run_dir']}`",
        f"- Config path: `{run_payload['config_path']}`",
        f"- Python: `{platform.python_version()}`",
        f"- Platform: `{platform.platform()}`",
        f"- CPU count: `{os.cpu_count()}`",
    ]
    if hardware_note:
        lines.append(f"- Intended hardware note: `{hardware_note}`")
    if gpu_names:
        lines.append(f"- Detected GPUs: `{', '.join(gpu_names)}`")

    lines.extend(
        [
            "",
            "## Commands",
            "",
            f"- Top-level matrix command: `python experiments/run_final_matrix.py --config {_portable_path(Path(run_payload['config_path']))}`",
            "",
            "## Step Outputs",
            "",
        ]
    )
    for step_name, result in run_payload["steps"].items():
        lines.append(f"- {step_name}: `{json.dumps(result['outputs'], sort_keys=True)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    raw_cfg = load_yaml(config_path)
    cfg = ProjectConfig.from_dict(raw_cfg)
    hardware_note = str(raw_cfg.get("hardware_note", "")).strip()

    metadata_path = write_run_metadata(config=cfg, project_root=PROJECT_ROOT, config_path=config_path)
    run_dir = metadata_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    mode = str(raw_cfg.get("mode", "pilot"))
    steps: dict[str, Any] = {}
    for step_name in ("baselines", "generator", "iterative_balancing", "transfer"):
        section = raw_cfg.get(step_name, {})
        if not section:
            continue
        if not isinstance(section, dict):
            raise ValueError(f"Section '{step_name}' must be a mapping.")
        if not bool(section.get("enabled", True)):
            continue
        section_cfg = _load_section_cfg(section)
        steps[step_name] = _run_script(step_name, section_cfg)

    summary = {
        "mode": mode,
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "steps": steps,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_experiment_index(run_dir / "experiment_index.md", summary)
    _write_command_list(run_dir / "command_list.md", config_path)
    _write_run_summary_note(run_dir / "run_summary.md", summary)
    _write_reproducibility_appendix(run_dir / "reproducibility_appendix.md", summary, hardware_note)

    print("[OK] Final matrix orchestration complete.")
    print(f"[OK] Run dir: {run_dir}")
    print(f"[OK] Steps: {list(steps.keys())}")
    print(f"[OK] Summary: {run_dir / 'summary.json'}")
    print(f"[OK] Experiment index: {run_dir / 'experiment_index.md'}")
    print(f"[OK] Command list: {run_dir / 'command_list.md'}")
    print(f"[OK] Run summary: {run_dir / 'run_summary.md'}")
    print(f"[OK] Reproducibility appendix: {run_dir / 'reproducibility_appendix.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
