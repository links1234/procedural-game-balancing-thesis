"""Scripted fallback policy used as the thesis third policy family."""

from __future__ import annotations

from sap_thesis_env import FrozenArenaEnv


class ScriptedValuePolicy:
    """State-aware deterministic policy that scores actions from board context."""

    FOOD_VALUE = {
        "food-apple": 1.8,
        "food-meat-bone": 2.2,
        "food-garlic": 2.1,
        "food-pear": 2.3,
        "food-canned-food": 1.9,
        "food-salad-bowl": 2.0,
        "food-chocolate": 2.8,
    }

    def select_action(self, env: FrozenArenaEnv) -> int:
        actions = env.available_actions()
        ranked: list[tuple[float, int, int]] = []
        for action_id, spec in actions.items():
            score = self._score_action(env, spec.name, spec.args)
            ranked.append((score, -action_id, action_id))
        ranked.sort(reverse=True)
        return ranked[0][2]

    def _score_action(self, env: FrozenArenaEnv, action_name: str, args: tuple[int, ...]) -> float:
        assert env.player is not None
        empty_slots = sum(1 for slot in env.player.team.slots if slot.empty)
        team_strength = sum(self._team_pet_value(slot.pet, env.player.turn) for slot in env.player.team.slots if not slot.empty)

        if action_name == "buy_pet":
            shop_idx = args[0]
            pet = env.player.shop.slots[shop_idx].obj
            score = 8.5 + self._team_pet_value(pet, env.player.turn)
            if empty_slots > 0:
                score += 1.25
            if getattr(pet, "name", "").startswith("pet-generated-"):
                score += 1.5
            if env.player.turn <= 3 and empty_slots >= 3:
                score += 0.5
            return score

        if action_name == "buy_combine":
            shop_idx, team_idx = args
            shop_pet = env.player.shop.slots[shop_idx].obj
            team_pet = env.player.team[team_idx].pet
            score = 11.0
            score += 0.70 * self._team_pet_value(shop_pet, env.player.turn)
            score += 0.60 * self._team_pet_value(team_pet, env.player.turn)
            score += 0.40 * max(0, 3 - int(getattr(team_pet, "level", 1)))
            return score

        if action_name == "combine":
            src_idx, dst_idx = args
            src_pet = env.player.team[src_idx].pet
            dst_pet = env.player.team[dst_idx].pet
            score = 10.5
            score += 0.55 * self._team_pet_value(src_pet, env.player.turn)
            score += 0.55 * self._team_pet_value(dst_pet, env.player.turn)
            score += 0.35 * max(0, 3 - int(getattr(dst_pet, "level", 1)))
            return score

        if action_name == "buy_food_target":
            shop_idx, team_idx = args
            food = env.player.shop.slots[shop_idx].obj
            target_pet = env.player.team[team_idx].pet
            score = 6.2 + self._food_value(getattr(food, "name", ""))
            score += 0.35 * self._team_pet_value(target_pet, env.player.turn)
            if empty_slots >= 2 and env.player.turn <= 4:
                score -= 1.25
            return score

        if action_name == "buy_food_auto":
            shop_idx = args[0]
            food = env.player.shop.slots[shop_idx].obj
            score = 5.8 + self._food_value(getattr(food, "name", ""))
            if empty_slots >= 2 and env.player.turn <= 4:
                score -= 1.0
            return score

        if action_name == "freeze":
            slot = env.player.shop.slots[args[0]]
            hold_value = self._shop_hold_value(env, slot)
            if hold_value < 0.0:
                return -100.0
            if env.player.gold < getattr(slot, "cost", 99) or empty_slots == 0:
                return 2.4 + hold_value
            return -0.4 + (0.15 * hold_value)

        if action_name == "unfreeze":
            slot = env.player.shop.slots[args[0]]
            if self._shop_hold_value(env, slot) < 0.0:
                return -100.0
            return -0.75 - (0.10 * self._shop_hold_value(env, slot))

        if action_name == "roll":
            score = 2.2 + (0.22 * env.player.gold) + (0.55 * empty_slots)
            score -= 0.06 * team_strength
            score -= 0.10 * max(0, env.player.turn - 3)
            return score

        if action_name == "sell":
            pet = env.player.team[args[0]].pet
            return -1.8 - (0.18 * self._team_pet_value(pet, env.player.turn)) + (0.05 * env.player.turn)

        if action_name == "end_turn":
            score = 0.4 + (0.9 if env.player.gold <= 1 else 0.0)
            score -= 0.55 * empty_slots
            score += 0.04 * team_strength
            return score

        return -5.0

    def _shop_hold_value(self, env: FrozenArenaEnv, slot) -> float:
        obj = getattr(slot, "obj", None)
        obj_name = getattr(obj, "name", "")
        if obj is None or getattr(slot, "slot_type", "") in {"", "none"} or obj_name in {"", "none", "pet-none", "food-none"}:
            return -5.0
        if slot.slot_type == "pet":
            return 0.20 * self._team_pet_value(slot.obj, env.player.turn)
        if slot.slot_type == "food":
            return self._food_value(getattr(slot.obj, "name", ""))
        return 0.0

    def _food_value(self, food_name: str) -> float:
        return self.FOOD_VALUE.get(food_name, 1.2)

    def _team_pet_value(self, pet, turn: int) -> float:
        attack = int(getattr(pet, "attack", 0) or 0)
        health = int(getattr(pet, "health", 0) or 0)
        level = int(getattr(pet, "level", 1) or 1)
        tier = int(getattr(pet, "tier", min(max(turn, 1), 6)) or min(max(turn, 1), 6))

        value = (0.95 * attack) + (1.05 * health)
        value += 0.45 * level
        value += 0.25 * tier
        if getattr(pet, "name", "").startswith("pet-generated-"):
            value += 1.75
        return value


class LearnedPolicyStub(ScriptedValuePolicy):
    """Backward-compatible alias retained for historical experiment configs."""
