from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from sap_thesis_env.config import ProjectConfig


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_git_hash(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "nogit"


def ensure_output_dirs(project_root: Path, output_dir: str) -> Path:
    root = project_root / output_dir
    for sub in ("raw", "processed", "figures", "tables"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def build_run_id(experiment_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = uuid.uuid4().hex[:8]
    return f"{experiment_name}_{stamp}_{token}"


def write_run_metadata(
    config: ProjectConfig,
    project_root: Path,
    config_path: Path,
) -> Path:
    outputs_root = ensure_output_dirs(project_root, config.output_dir)
    run_id = build_run_id(config.experiment_name)
    run_dir = outputs_root / "raw" / config.experiment_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "run_id": run_id,
        "timestamp_utc": utc_now_iso(),
        "git_hash": safe_git_hash(project_root),
        "config_path": str(config_path.resolve()),
        "project_root": str(project_root.resolve()),
        "config": asdict(config),
    }

    metadata_path = run_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    return metadata_path
