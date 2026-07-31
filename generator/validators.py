"""Hard and safety validation logic for generated artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from generator.schema import ArtifactSpec, parse_artifact


TIER_POWER_BUDGET_MAX = {
    1: 12,
    2: 16,
    3: 20,
    4: 24,
    5: 28,
    6: 32,
}
TIER_POWER_BUDGET_MIN = {
    1: 2,
    2: 4,
    3: 6,
    4: 8,
    5: 10,
    6: 12,
}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    artifact: ArtifactSpec | None
    schema_issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)
    hard_issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)
    safety_issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _estimated_power_score(spec: ArtifactSpec) -> float:
    # Lightweight proxy used for early filtering.
    return (
        spec.base_attack * 0.9
        + spec.base_health * 0.9
        + spec.effect_value * 1.6
        + spec.cap_or_cooldown * 0.8
    )


def _hard_constraint_issues(spec: ArtifactSpec, *, strict_pet_only: bool) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if strict_pet_only and spec.artifact_type != "pet":
        issues.append(
            ValidationIssue(
                code="hard.pet_only",
                message="This experiment is configured for pet-only artifact generation.",
            )
        )

    # Trigger/effect compatibility.
    if spec.effect_type == "Summon" and spec.trigger_type not in {"StartOfBattle", "Faint", "Hurt"}:
        issues.append(
            ValidationIssue(
                code="hard.trigger_effect_combo",
                message="Summon effects require StartOfBattle, Faint, or Hurt trigger.",
            )
        )
    if spec.effect_type == "GoldGain" and spec.trigger_type not in {"Sell", "EndOfTurn", "BuyFriend"}:
        issues.append(
            ValidationIssue(
                code="hard.gold_trigger_combo",
                message="GoldGain effects require Sell, EndOfTurn, or BuyFriend trigger.",
            )
        )

    # Target/effect compatibility.
    if spec.effect_type in {"BuffFriend"} and spec.target_type in {"RandomEnemy", "HighestAttackEnemy"}:
        issues.append(
            ValidationIssue(
                code="hard.target_effect_combo",
                message="BuffFriend cannot target enemy-only target types.",
            )
        )
    if spec.effect_type in {"DebuffEnemy", "DamageEnemy"} and spec.target_type in {"Self", "RandomFriend", "AllFriends", "AdjacentFriend"}:
        issues.append(
            ValidationIssue(
                code="hard.target_effect_combo",
                message="Enemy debuff/damage effects must target enemies.",
            )
        )

    # Banned interaction patterns.
    banned_triplets = {
        ("Faint", "Summon", "AllFriends"),
        ("Hurt", "Summon", "AllFriends"),
        ("StartOfBattle", "GoldGain", "AllFriends"),
    }
    if (spec.trigger_type, spec.effect_type, spec.target_type) in banned_triplets:
        issues.append(
            ValidationIssue(
                code="hard.banned_interaction",
                message="Artifact matches banned trigger/effect/target interaction.",
            )
        )

    power_score = _estimated_power_score(spec)
    if power_score > TIER_POWER_BUDGET_MAX[spec.tier]:
        issues.append(
            ValidationIssue(
                code="hard.power_budget_high",
                message=(
                    f"Estimated power score {power_score:.2f} exceeds "
                    f"tier-{spec.tier} max {TIER_POWER_BUDGET_MAX[spec.tier]}."
                ),
            )
        )
    if power_score < TIER_POWER_BUDGET_MIN[spec.tier]:
        issues.append(
            ValidationIssue(
                code="hard.power_budget_low",
                message=(
                    f"Estimated power score {power_score:.2f} below "
                    f"tier-{spec.tier} min {TIER_POWER_BUDGET_MIN[spec.tier]}."
                ),
            )
        )

    return issues


def _safety_constraint_issues(spec: ArtifactSpec) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    # Dominance cap proxy.
    dominance_score = spec.base_attack + spec.base_health + (2 * spec.effect_value)
    if dominance_score > 34:
        issues.append(
            ValidationIssue(
                code="safety.dominance",
                message=f"Dominance proxy score {dominance_score} exceeds cap 34.",
            )
        )

    # Degeneracy guard: high-value repeatable effects need cooldown.
    if spec.effect_value >= 4 and spec.cap_or_cooldown == 0:
        issues.append(
            ValidationIssue(
                code="safety.cooldown_required",
                message="High effect_value artifacts must include non-zero cap_or_cooldown.",
            )
        )

    # Complexity cap proxy.
    if len(spec.tags) > 4:
        issues.append(
            ValidationIssue(
                code="safety.complexity",
                message="Tag count exceeds recommended complexity cap of 4.",
            )
        )

    if "combo" in spec.tags and "summon" in spec.tags and spec.effect_type == "Summon":
        issues.append(
            ValidationIssue(
                code="safety.combo_chain_risk",
                message="combo+summon tagged summon effect flagged for chain-risk.",
            )
        )

    return issues


def validate_artifact(
    payload: dict[str, Any],
    *,
    strict_pet_only: bool = True,
) -> ValidationResult:
    try:
        spec = parse_artifact(payload)
    except Exception as exc:
        issue = ValidationIssue(code="schema.parse_error", message=str(exc))
        return ValidationResult(
            is_valid=False,
            artifact=None,
            schema_issues=(issue,),
            diagnostics={},
        )

    hard_issues = _hard_constraint_issues(spec, strict_pet_only=strict_pet_only)
    safety_issues = _safety_constraint_issues(spec)
    is_valid = not hard_issues and not safety_issues

    diagnostics = {
        "artifact_id": spec.artifact_id,
        "estimated_power_score": _estimated_power_score(spec),
        "dominance_proxy_score": spec.base_attack + spec.base_health + (2 * spec.effect_value),
    }

    return ValidationResult(
        is_valid=is_valid,
        artifact=spec,
        hard_issues=tuple(hard_issues),
        safety_issues=tuple(safety_issues),
        diagnostics=diagnostics,
    )


def assert_valid_artifact(payload: dict[str, Any], *, strict_pet_only: bool = True) -> ArtifactSpec:
    result = validate_artifact(payload, strict_pet_only=strict_pet_only)
    if not result.is_valid:
        messages = []
        for issue in list(result.schema_issues) + list(result.hard_issues) + list(result.safety_issues):
            messages.append(f"{issue.code}: {issue.message}")
        raise ValueError("Artifact failed validation: " + " | ".join(messages))
    assert result.artifact is not None
    return result.artifact
