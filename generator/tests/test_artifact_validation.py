from __future__ import annotations

from generator.schema import parse_artifact
from generator.tests.corpus import INVALID_ARTIFACTS, VALID_ARTIFACTS
from generator.validators import assert_valid_artifact, validate_artifact


def test_valid_corpus_is_accepted() -> None:
    for payload in VALID_ARTIFACTS:
        result = validate_artifact(payload, strict_pet_only=True)
        assert result.is_valid, f"Expected valid artifact, got issues: {result}"
        assert result.artifact is not None
        assert len(result.artifact.artifact_id) == 16


def test_invalid_corpus_is_rejected() -> None:
    for payload in INVALID_ARTIFACTS:
        result = validate_artifact(payload, strict_pet_only=True)
        assert not result.is_valid
        assert result.schema_issues or result.hard_issues or result.safety_issues


def test_schema_parse_gives_deterministic_id() -> None:
    payload = VALID_ARTIFACTS[0]
    a = parse_artifact(payload)
    b = parse_artifact(payload)
    assert a.artifact_id == b.artifact_id


def test_assert_valid_artifact_raises_on_invalid() -> None:
    bad = INVALID_ARTIFACTS[2]
    raised = False
    try:
        assert_valid_artifact(bad, strict_pet_only=True)
    except ValueError:
        raised = True
    assert raised
