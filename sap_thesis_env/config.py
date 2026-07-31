from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a run config is missing required fields or has invalid values."""


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str
    experiment_name: str
    seed: int
    output_dir: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectConfig":
        required = ("project_name", "experiment_name", "seed", "output_dir")
        missing = [key for key in required if key not in data]
        if missing:
            raise ConfigError(f"Missing required config keys: {missing}")

        project_name = data["project_name"]
        experiment_name = data["experiment_name"]
        seed = data["seed"]
        output_dir = data["output_dir"]

        if not isinstance(project_name, str) or not project_name.strip():
            raise ConfigError("project_name must be a non-empty string")
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise ConfigError("experiment_name must be a non-empty string")
        if not isinstance(seed, int):
            raise ConfigError("seed must be an integer")
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ConfigError("output_dir must be a non-empty string")

        return cls(
            project_name=project_name.strip(),
            experiment_name=experiment_name.strip(),
            seed=seed,
            output_dir=output_dir.strip(),
        )


def load_config(config_path: Path) -> ProjectConfig:
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ConfigError("Config must deserialize to a mapping/object")

    return ProjectConfig.from_dict(payload)
