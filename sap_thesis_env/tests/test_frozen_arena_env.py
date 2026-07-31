from __future__ import annotations

from opponents.heuristic_policy import BasicHeuristicPolicy
from sap_thesis_env import FrozenArenaEnv, play_random_episode


def test_reset_step_api_contract() -> None:
    env = FrozenArenaEnv(seed=12345)
    obs = env.reset()
    assert {"turn", "wins", "lives", "gold", "state_hash", "available_action_ids"} <= set(obs)
    action_id = min(obs["available_action_ids"])
    next_obs, reward, done, info = env.step(action_id)
    assert isinstance(next_obs, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_action_mask_consistency() -> None:
    env = FrozenArenaEnv(seed=12345)
    env.reset()
    ok, reason = env.validate_action_mask()
    assert ok, reason


def test_empty_shop_slot_does_not_expose_freeze_action() -> None:
    env = FrozenArenaEnv(seed=12345)
    env.reset()
    assert env.player is not None
    shop_index = 0
    slot_type = type(env.player.shop.slots[shop_index])
    env.player.shop.slots[shop_index] = slot_type()

    assert env.player.shop.slots[shop_index].empty
    available = env.available_actions()
    assert env.ACTION_BASE["freeze"] + shop_index not in available
    assert env.ACTION_BASE["unfreeze"] + shop_index not in available


def test_replay_determinism_small() -> None:
    env_a = FrozenArenaEnv(seed=12345)
    env_b = FrozenArenaEnv(seed=12345)
    t1 = play_random_episode(env_a, action_seed=2026, max_steps=64)
    t2 = play_random_episode(env_b, action_seed=2026, max_steps=64)
    assert t1 == t2


def test_generated_artifact_injection_is_deterministic() -> None:
    artifact = {
        "artifact_type": "pet",
        "tier": 3,
        "base_attack": 4,
        "base_health": 4,
        "trigger_type": "StartOfBattle",
        "target_type": "RandomEnemy",
        "effect_type": "DamageEnemy",
        "effect_value": 2,
        "cap_or_cooldown": 1,
        "cost": 3,
        "tags": ["tempo"],
    }
    env_a = FrozenArenaEnv(seed=12345, injected_artifact=artifact)
    env_b = FrozenArenaEnv(seed=12345, injected_artifact=artifact)

    obs_a = env_a.reset()
    obs_b = env_b.reset()

    assert obs_a["injected_artifact_id"] == obs_b["injected_artifact_id"]
    assert obs_a["injected_pet_name"] == obs_b["injected_pet_name"]
    assert obs_a["injected_shop_slot_index"] == 0
    assert obs_b["injected_shop_slot_index"] == 0
    assert env_a.player.shop.slots[0].obj.name == env_b.player.shop.slots[0].obj.name
    assert env_a.player.shop.slots[0].obj.name.startswith("pet-generated-")
    assert all(slot.pet.name == "pet-none" for slot in env_a.player.team.slots)
    assert all(slot.pet.name == "pet-none" for slot in env_b.player.team.slots)
    assert obs_a["state_hash"] == obs_b["state_hash"]


def test_multi_artifact_injection_is_deterministic() -> None:
    artifacts = [
        {
            "artifact_type": "pet",
            "tier": 2,
            "base_attack": 3,
            "base_health": 4,
            "trigger_type": "StartOfBattle",
            "target_type": "RandomEnemy",
            "effect_type": "DamageEnemy",
            "effect_value": 1,
            "cap_or_cooldown": 1,
            "cost": 3,
            "tags": ["tempo"],
        },
        {
            "artifact_type": "pet",
            "tier": 3,
            "base_attack": 4,
            "base_health": 5,
            "trigger_type": "EndOfTurn",
            "target_type": "Self",
            "effect_type": "BuffFriend",
            "effect_value": 2,
            "cap_or_cooldown": 1,
            "cost": 3,
            "tags": ["scaling"],
        },
    ]
    env_a = FrozenArenaEnv(seed=12345, injected_artifacts=artifacts)
    env_b = FrozenArenaEnv(seed=12345, injected_artifacts=artifacts)

    obs_a = env_a.reset()
    obs_b = env_b.reset()

    assert obs_a["injected_artifact_ids"] == obs_b["injected_artifact_ids"]
    assert obs_a["injected_pet_names"] == obs_b["injected_pet_names"]
    assert obs_a["injected_shop_slot_indices"] == [0, 1]
    assert obs_b["injected_shop_slot_indices"] == [0, 1]
    assert [env_a.player.shop.slots[idx].obj.name for idx in [0, 1]] == [
        env_b.player.shop.slots[idx].obj.name for idx in [0, 1]
    ]
    assert all(name.startswith("pet-generated-") for name in obs_a["injected_pet_names"])
    assert all(slot.pet.name == "pet-none" for slot in env_a.player.team.slots)
    assert obs_a["state_hash"] == obs_b["state_hash"]


def test_generated_artifact_injection_changes_controlled_episode_outcome() -> None:
    artifact = {
        "artifact_type": "pet",
        "tier": 4,
        "base_attack": 8,
        "base_health": 8,
        "trigger_type": "StartOfBattle",
        "target_type": "RandomEnemy",
        "effect_type": "DamageEnemy",
        "effect_value": 2,
        "cap_or_cooldown": 1,
        "cost": 3,
        "tags": ["tempo"],
    }
    policy = BasicHeuristicPolicy()
    env_without = FrozenArenaEnv(seed=12345)
    env_with = FrozenArenaEnv(seed=12345, injected_artifact=artifact)

    def run_episode(env: FrozenArenaEnv) -> tuple[int, int, str]:
        obs = env.reset(seed=12345)
        for _ in range(64):
            action_id = policy.select_action(env)
            obs, _reward, done, _info = env.step(action_id)
            if done:
                break
        return int(obs["wins"]), int(obs["lives"]), str(obs["state_hash"])

    result_without = run_episode(env_without)
    result_with = run_episode(env_with)

    assert result_without != result_with


def test_generated_artifact_shop_injection_does_not_raise_backend_error_under_heuristic_policy() -> None:
    artifact = {
        "artifact_type": "pet",
        "tier": 3,
        "base_attack": 4,
        "base_health": 5,
        "trigger_type": "Faint",
        "target_type": "RandomFriend",
        "effect_type": "BuffFriend",
        "effect_value": 2,
        "cap_or_cooldown": 1,
        "cost": 3,
        "tags": ["tempo", "scaling"],
    }
    env = FrozenArenaEnv(seed=12345, max_turn=10, strict_invalid_actions=True, injected_artifact=artifact)
    policy = BasicHeuristicPolicy()

    obs = env.reset(seed=20260301)
    backend_error = None
    for _ in range(64):
        try:
            action_id = policy.select_action(env)
            obs, _reward, done, _info = env.step(action_id)
        except Exception as exc:  # pragma: no cover - failure path only
            backend_error = str(exc)
            break
        if done:
            break

    assert backend_error is None, backend_error
    assert obs["injected_pet_name"] is not None
