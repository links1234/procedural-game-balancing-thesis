"""Helpers for compiling and injecting generated artifacts into the frozen SAP env."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

from generator.schema import ArtifactSpec, parse_artifact
from sap_thesis_env.adapters.sapai_backend import ensure_sapai_importable


SUPPORTED_TRIGGER_TYPES = {
    "StartOfBattle",
    "EndOfTurn",
    "Faint",
    "Hurt",
    "Sell",
    "BuyFood",
    "BuyFriend",
}
SUPPORTED_TARGET_TYPES = {
    "Self",
    "RandomFriend",
    "AdjacentFriend",
    "RandomEnemy",
}
SUPPORTED_EFFECT_TYPES = {
    "BuffFriend",
    "DamageEnemy",
    "GoldGain",
}

TRIGGER_MAP: dict[str, tuple[str, dict[str, str]]] = {
    "StartOfBattle": ("StartOfBattle", {"kind": "Player"}),
    "EndOfTurn": ("EndOfTurn", {"kind": "Player"}),
    "Faint": ("Faint", {"kind": "Self"}),
    "Hurt": ("Hurt", {"kind": "Self"}),
    "Sell": ("Sell", {"kind": "Self"}),
    "BuyFood": ("BuyFood", {"kind": "Player"}),
    "BuyFriend": ("Buy", {"kind": "Self"}),
}

TARGET_MAP: dict[str, dict[str, Any]] = {
    "Self": {"kind": "Self"},
    "RandomFriend": {"kind": "RandomFriend", "n": 1},
    "AdjacentFriend": {"kind": "AdjacentFriends"},
    "RandomEnemy": {"kind": "RandomEnemy", "n": 1},
}


class UnsupportedArtifactSpecError(ValueError):
    """Raised when an artifact is valid at schema level but unsupported for injection."""


@dataclass(frozen=True)
class InjectionRecord:
    artifact_id: str
    pet_name: str
    shop_slot_index: int


def normalize_artifact_spec(artifact: ArtifactSpec | dict[str, Any] | None) -> ArtifactSpec | None:
    if artifact is None:
        return None
    if isinstance(artifact, ArtifactSpec):
        return artifact
    if isinstance(artifact, dict):
        return parse_artifact(artifact)
    raise TypeError(f"Unsupported artifact input type: {type(artifact).__name__}")


def normalize_artifact_specs(
    artifacts: ArtifactSpec | dict[str, Any] | list[ArtifactSpec | dict[str, Any]] | tuple[ArtifactSpec | dict[str, Any], ...] | None,
) -> tuple[ArtifactSpec, ...]:
    if artifacts is None:
        return ()
    if isinstance(artifacts, (ArtifactSpec, dict)):
        spec = normalize_artifact_spec(artifacts)
        return (spec,) if spec is not None else ()
    if isinstance(artifacts, (list, tuple)):
        out: list[ArtifactSpec] = []
        for artifact in artifacts:
            spec = normalize_artifact_spec(artifact)
            if spec is not None:
                out.append(spec)
        return tuple(out)
    raise TypeError(f"Unsupported artifacts input type: {type(artifacts).__name__}")


def artifact_set_id(specs: tuple[ArtifactSpec, ...] | list[ArtifactSpec]) -> str:
    artifact_ids = [spec.artifact_id for spec in specs]
    canonical = json.dumps(artifact_ids, separators=(",", ":"), sort_keys=False)
    return sha256(canonical.encode("utf-8")).hexdigest()[:16]


def validate_artifact_support(spec: ArtifactSpec) -> None:
    if spec.artifact_type != "pet":
        raise UnsupportedArtifactSpecError("Only pet artifacts are supported for simulator injection.")
    if spec.trigger_type not in SUPPORTED_TRIGGER_TYPES:
        raise UnsupportedArtifactSpecError(f"Unsupported trigger_type for injection: {spec.trigger_type}")
    if spec.effect_type not in SUPPORTED_EFFECT_TYPES:
        raise UnsupportedArtifactSpecError(f"Unsupported effect_type for injection: {spec.effect_type}")

    if spec.effect_type == "GoldGain":
        if spec.target_type not in {"None", "Self"}:
            raise UnsupportedArtifactSpecError(
                "GoldGain artifacts must use target_type None or Self for simulator injection."
            )
        return

    if spec.target_type not in SUPPORTED_TARGET_TYPES:
        raise UnsupportedArtifactSpecError(
            f"Unsupported target_type for simulator injection: {spec.target_type}"
        )


def generated_pet_name(spec: ArtifactSpec) -> str:
    return f"pet-generated-{spec.artifact_id}"


def _scaled_effect_value(spec: ArtifactSpec, level: int) -> int:
    return max(1, spec.effect_value * level)


def _compile_effect(spec: ArtifactSpec, level: int) -> dict[str, Any]:
    magnitude = _scaled_effect_value(spec, level)

    if spec.effect_type == "BuffFriend":
        return {
            "kind": "ModifyStats",
            "attackAmount": magnitude,
            "healthAmount": magnitude,
            "target": TARGET_MAP[spec.target_type],
            "untilEndOfBattle": False,
        }
    if spec.effect_type == "DamageEnemy":
        return {
            "kind": "DealDamage",
            "target": TARGET_MAP[spec.target_type],
            "amount": magnitude,
        }
    if spec.effect_type == "GoldGain":
        return {
            "kind": "GainGold",
            "amount": magnitude,
        }
    raise UnsupportedArtifactSpecError(f"Unsupported effect_type: {spec.effect_type}")


def _compile_level_ability(spec: ArtifactSpec, level: int) -> dict[str, Any]:
    trigger, triggered_by = TRIGGER_MAP[spec.trigger_type]
    ability: dict[str, Any] = {
        "description": (
            f"Generated artifact {spec.artifact_id}: "
            f"{trigger} -> {spec.effect_type} ({spec.target_type})"
        ),
        "trigger": trigger,
        "triggeredBy": dict(triggered_by),
        "effect": _compile_effect(spec, level),
    }
    if spec.cap_or_cooldown > 0:
        # The frozen backend supports maxTriggers more directly than true cooldowns.
        ability["maxTriggers"] = spec.cap_or_cooldown
    return ability


def compile_generated_pet_entry(spec: ArtifactSpec, pack: str) -> tuple[str, dict[str, Any]]:
    validate_artifact_support(spec)
    pet_name = generated_pet_name(spec)
    pet_entry = {
        "name": f"Generated {spec.artifact_id}",
        "id": pet_name,
        "tier": spec.tier,
        "baseAttack": spec.base_attack,
        "baseHealth": spec.base_health,
        "packs": [pack],
        "level1Ability": _compile_level_ability(spec, level=1),
        "level2Ability": _compile_level_ability(spec, level=2),
        "level3Ability": _compile_level_ability(spec, level=3),
        "probabilities": "none",
    }
    return pet_name, pet_entry


def register_generated_pet(spec: ArtifactSpec, pack: str) -> str:
    ensure_sapai_importable()
    from sapai.data import data  # type: ignore

    pet_name, pet_entry = compile_generated_pet_entry(spec, pack=pack)
    if pet_name not in data["pets"]:
        data["pets"][pet_name] = pet_entry
    return pet_name


def choose_injection_pet_slot(shop) -> int | None:
    preferred: list[int] = []
    fallback: list[int] = []
    for idx, slot in enumerate(shop.slots):
        if slot.slot_type != "pet":
            continue
        fallback.append(idx)
        if not slot.frozen:
            preferred.append(idx)
    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    return None


def choose_injection_pet_slots(shop, count: int) -> list[int]:
    preferred: list[int] = []
    fallback: list[int] = []
    for idx, slot in enumerate(shop.slots):
        if slot.slot_type != "pet":
            continue
        fallback.append(idx)
        if not slot.frozen:
            preferred.append(idx)

    ordered = preferred + [idx for idx in fallback if idx not in preferred]
    return ordered[: max(0, count)]


def inject_artifact_into_player_shop(player, spec: ArtifactSpec) -> InjectionRecord | None:
    ensure_sapai_importable()
    from sapai.pets import Pet  # type: ignore
    from sapai.shop import ShopSlot  # type: ignore

    pet_name = register_generated_pet(spec, pack=player.pack)
    slot_index = choose_injection_pet_slot(player.shop)
    if slot_index is None:
        return None

    current_slot = player.shop.slots[slot_index]
    seed_state = getattr(current_slot, "seed_state", None)
    generated_pet = Pet(pet_name, shop=player.shop, player=player, seed_state=seed_state)
    injected_slot = ShopSlot(
        generated_pet,
        slot_type="pet",
        frozen=False,
        turn=player.shop.turn,
        cost=spec.cost,
        pack=player.pack,
        seed_state=seed_state,
    )
    injected_slot.obj.player = player
    injected_slot.obj.shop = player.shop
    injected_slot.obj.team = None
    player.shop.slots[slot_index] = injected_slot

    return InjectionRecord(
        artifact_id=spec.artifact_id,
        pet_name=pet_name,
        shop_slot_index=slot_index,
    )


def inject_artifacts_into_player_shop(player, specs: tuple[ArtifactSpec, ...] | list[ArtifactSpec]) -> list[InjectionRecord]:
    ensure_sapai_importable()
    from sapai.pets import Pet  # type: ignore
    from sapai.shop import ShopSlot  # type: ignore

    normalized_specs = tuple(specs)
    if not normalized_specs:
        return []

    chosen_slots = choose_injection_pet_slots(player.shop, len(normalized_specs))
    if len(chosen_slots) != len(normalized_specs):
        raise UnsupportedArtifactSpecError(
            "Unable to inject full artifact set into shop: insufficient pet slots available."
        )
    records: list[InjectionRecord] = []
    for slot_index, spec in zip(chosen_slots, normalized_specs):
        pet_name = register_generated_pet(spec, pack=player.pack)
        current_slot = player.shop.slots[slot_index]
        seed_state = getattr(current_slot, "seed_state", None)
        generated_pet = Pet(pet_name, shop=player.shop, player=player, seed_state=seed_state)
        injected_slot = ShopSlot(
            generated_pet,
            slot_type="pet",
            frozen=False,
            turn=player.shop.turn,
            cost=spec.cost,
            pack=player.pack,
            seed_state=seed_state,
        )
        injected_slot.obj.player = player
        injected_slot.obj.shop = player.shop
        injected_slot.obj.team = None
        player.shop.slots[slot_index] = injected_slot
        records.append(
            InjectionRecord(
                artifact_id=spec.artifact_id,
                pet_name=pet_name,
                shop_slot_index=slot_index,
            )
        )
    return records
