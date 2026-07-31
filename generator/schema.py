"""Artifact schema and deterministic identifiers for generated content."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any


ALLOWED_ARTIFACT_TYPES = {"pet", "food"}
ALLOWED_TRIGGER_TYPES = {
    "StartOfBattle",
    "EndOfTurn",
    "Faint",
    "Hurt",
    "Sell",
    "BuyFood",
    "BuyFriend",
}
ALLOWED_TARGET_TYPES = {
    "Self",
    "RandomFriend",
    "AdjacentFriend",
    "AllFriends",
    "RandomEnemy",
    "HighestAttackEnemy",
    "None",
}
ALLOWED_EFFECT_TYPES = {
    "BuffFriend",
    "DebuffEnemy",
    "Summon",
    "GoldGain",
    "DamageEnemy",
    "GiveStatus",
}
ALLOWED_TAGS = {
    "economy",
    "tempo",
    "scaling",
    "summon",
    "debuff",
    "utility",
    "late_game",
    "early_game",
    "risk",
    "combo",
}

DEFAULT_SOURCE = "generator"


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_type: str
    tier: int
    base_attack: int
    base_health: int
    trigger_type: str
    target_type: str
    effect_type: str
    effect_value: int
    cap_or_cooldown: int
    cost: int
    tags: tuple[str, ...] = field(default_factory=tuple)
    source: str = DEFAULT_SOURCE

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "tier": self.tier,
            "base_attack": self.base_attack,
            "base_health": self.base_health,
            "trigger_type": self.trigger_type,
            "target_type": self.target_type,
            "effect_type": self.effect_type,
            "effect_value": self.effect_value,
            "cap_or_cooldown": self.cap_or_cooldown,
            "cost": self.cost,
            "tags": list(self.tags),
            "source": self.source,
        }

    @property
    def artifact_id(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _require_int(name: str, value: Any, *, low: int, high: int) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be int, got {type(value).__name__}")
    if value < low or value > high:
        raise ValueError(f"{name}={value} out of bounds [{low}, {high}]")
    return value


def _require_choice(name: str, value: Any, allowed: set[str]) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be str, got {type(value).__name__}")
    if value not in allowed:
        raise ValueError(f"{name}='{value}' not in allowed values: {sorted(allowed)}")
    return value


def parse_artifact(payload: dict[str, Any]) -> ArtifactSpec:
    """Parse and validate schema-level constraints.

    This function only performs syntax/schema checks and value bounds.
    Hard/safety design constraints are applied in `generator.validators`.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact payload must be dict, got {type(payload).__name__}")

    required = {
        "artifact_type",
        "tier",
        "base_attack",
        "base_health",
        "trigger_type",
        "target_type",
        "effect_type",
        "effect_value",
        "cap_or_cooldown",
        "cost",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    artifact_type = _require_choice("artifact_type", payload["artifact_type"], ALLOWED_ARTIFACT_TYPES)
    tier = _require_int("tier", payload["tier"], low=1, high=6)
    base_attack = _require_int("base_attack", payload["base_attack"], low=0, high=50)
    base_health = _require_int("base_health", payload["base_health"], low=0, high=50)
    trigger_type = _require_choice("trigger_type", payload["trigger_type"], ALLOWED_TRIGGER_TYPES)
    target_type = _require_choice("target_type", payload["target_type"], ALLOWED_TARGET_TYPES)
    effect_type = _require_choice("effect_type", payload["effect_type"], ALLOWED_EFFECT_TYPES)
    effect_value = _require_int("effect_value", payload["effect_value"], low=0, high=20)
    cap_or_cooldown = _require_int("cap_or_cooldown", payload["cap_or_cooldown"], low=0, high=10)
    cost = _require_int("cost", payload["cost"], low=0, high=3)

    tags_raw = payload.get("tags", [])
    if tags_raw is None:
        tags_raw = []
    if not isinstance(tags_raw, list):
        raise ValueError(f"tags must be list[str], got {type(tags_raw).__name__}")
    tags: list[str] = []
    for idx, tag in enumerate(tags_raw):
        if not isinstance(tag, str):
            raise ValueError(f"tags[{idx}] must be str, got {type(tag).__name__}")
        if tag not in ALLOWED_TAGS:
            raise ValueError(f"tags[{idx}]='{tag}' not in allowed tags: {sorted(ALLOWED_TAGS)}")
        tags.append(tag)
    if len(tags) > 5:
        raise ValueError("tags length exceeds limit of 5")

    source = payload.get("source", DEFAULT_SOURCE)
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty string")

    return ArtifactSpec(
        artifact_type=artifact_type,
        tier=tier,
        base_attack=base_attack,
        base_health=base_health,
        trigger_type=trigger_type,
        target_type=target_type,
        effect_type=effect_type,
        effect_value=effect_value,
        cap_or_cooldown=cap_or_cooldown,
        cost=cost,
        tags=tuple(tags),
        source=source.strip(),
    )
