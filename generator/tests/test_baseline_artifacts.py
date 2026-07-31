from __future__ import annotations

import numpy as np

from experiments.run_baselines import build_heuristic_artifact, build_random_artifact
from generator.validators import validate_artifact


def test_random_baseline_artifact_is_valid() -> None:
    artifact = build_random_artifact(np.random.RandomState(12345))
    result = validate_artifact(artifact)
    assert result.is_valid
    assert result.artifact is not None


def test_heuristic_baseline_artifact_is_valid() -> None:
    artifact = build_heuristic_artifact()
    result = validate_artifact(artifact)
    assert result.is_valid
    assert result.artifact is not None
