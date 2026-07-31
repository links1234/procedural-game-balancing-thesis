"""Deterministic Arena-style environment built on a frozen SAP backend snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import itertools
import json
import random
from typing import Callable

import numpy as np

from generator.schema import ArtifactSpec
from sap_thesis_env.adapters.sapai_backend import load_sapai_symbols
from sap_thesis_env.generated_artifacts import (
    InjectionRecord,
    UnsupportedArtifactSpecError,
    inject_artifacts_into_player_shop,
    inject_artifact_into_player_shop,
    normalize_artifact_spec,
    normalize_artifact_specs,
)


@dataclass(frozen=True)
class ActionSpec:
    action_id: int
    name: str
    args: tuple[int, ...]
    description: str


class FrozenArenaEnv:
    """Minimal deterministic environment for thesis experimentation."""

    MAX_ACTIONS = 256
    INVALID_ACTION_PENALTY = -0.1

    ACTION_BASE = {
        "end_turn": 0,
        "roll": 1,
        "buy_pet": 10,  # + shop_idx (max 7)
        "buy_food_auto": 20,  # + shop_idx (max 7)
        "buy_food_target": 40,  # + shop_idx*5 + team_idx
        "buy_combine": 90,  # + shop_idx*5 + team_idx
        "combine": 140,  # + pair_idx in combinations(5,2)
        "sell": 170,  # + team_idx
        "freeze": 180,  # + shop_idx
        "unfreeze": 190,  # + shop_idx
    }

    # Conservative opponent pool for stable baseline runs.
    # These are regular shop pets that avoid many complex summon/chaining edge-cases
    # seen in older backend battle implementations.
    SAFE_OPPONENT_PETS = [
        "pet-ant",
        "pet-beaver",
        "pet-duck",
        "pet-fish",
        "pet-ladybug",
        "pet-mosquito",
        "pet-otter",
        "pet-pig",
        "pet-swan",
    ]

    # The frozen backend is fragile around a small summon/revive food subset.
    # Excluding those purchase actions keeps simulator-backed artifact evaluation stable.
    UNSAFE_PURCHASE_FOODS = {
        "food-honey",
        "food-mushroom",
        "food-sleeping-pill",
    }

    def __init__(
        self,
        seed: int = 12345,
        pack: str = "StandardPack",
        max_turn: int = 25,
        strict_invalid_actions: bool = False,
        injected_artifact: ArtifactSpec | dict | None = None,
        injected_artifacts: list[ArtifactSpec | dict] | tuple[ArtifactSpec | dict, ...] | None = None,
    ) -> None:
        symbols = load_sapai_symbols()
        self.Player = symbols["Player"]
        self.Team = symbols["Team"]
        self.Battle = symbols["Battle"]
        self.data = symbols["data"]
        self.get_shop_rules = symbols["get_shop_rules"]

        self.seed = seed
        self.pack = pack
        self.max_turn = max_turn
        self.strict_invalid_actions = strict_invalid_actions
        if injected_artifact is not None and injected_artifacts is not None:
            raise ValueError("Pass either injected_artifact or injected_artifacts, not both.")
        if injected_artifacts is not None:
            self.injected_artifacts = normalize_artifact_specs(injected_artifacts)
        else:
            self.injected_artifacts = normalize_artifact_specs(injected_artifact)
        self.injected_artifact = self.injected_artifacts[0] if self.injected_artifacts else None

        self._rng = np.random.RandomState(seed)
        self.player = None
        self.opponents: list[object] = []
        self.last_action: int | None = None
        self.invalid_action_count = 0
        self.step_count = 0
        self._done = False
        self.last_injection_record: InjectionRecord | None = None
        self.last_injection_records: list[InjectionRecord] = []

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.seed = seed
        # Some backend code paths rely on global RNG state. Reset both to keep
        # trajectories reproducible across independent environment instances.
        random.seed(self.seed)
        np.random.seed(self.seed)
        self._rng = np.random.RandomState(self.seed)
        self.player = self.Player(pack=self.pack, seed_state=self._rng.get_state())
        self.opponents = self._generate_opponents(num_turns=self.max_turn)
        self._apply_generated_artifact_injection()
        self.last_action = None
        self.invalid_action_count = 0
        self.step_count = 0
        self._done = False
        return self.observe()

    def observe(self) -> dict:
        assert self.player is not None
        available = self.available_actions()
        state = self.player.state
        return {
            "turn": self.player.turn,
            "wins": self.player.wins,
            "lives": self.player.lives,
            "gold": self.player.gold,
            "done": self._done,
            "state_hash": self._state_hash(state),
            "available_action_ids": sorted(available.keys()),
            "player_state": state,
            "injected_artifact_id": self.injected_artifact.artifact_id if self.injected_artifact else None,
            "injected_pet_name": self.last_injection_record.pet_name if self.last_injection_record else None,
            "injected_shop_slot_index": (
                self.last_injection_record.shop_slot_index if self.last_injection_record else None
            ),
            "injected_artifact_ids": [spec.artifact_id for spec in self.injected_artifacts],
            "injected_pet_names": [record.pet_name for record in self.last_injection_records],
            "injected_shop_slot_indices": [record.shop_slot_index for record in self.last_injection_records],
        }

    def action_masks(self) -> list[bool]:
        mask = [False] * self.MAX_ACTIONS
        for action_id in self.available_actions().keys():
            if 0 <= action_id < self.MAX_ACTIONS:
                mask[action_id] = True
        return mask

    def validate_action_mask(self) -> tuple[bool, str]:
        mask = self.action_masks()
        available = self.available_actions()
        for action_id in available.keys():
            if action_id >= self.MAX_ACTIONS:
                return False, f"Action id {action_id} exceeds MAX_ACTIONS={self.MAX_ACTIONS}"
            if not mask[action_id]:
                return False, f"Action id {action_id} missing from mask"
        return True, "ok"

    def step(self, action_id: int) -> tuple[dict, float, bool, dict]:
        assert self.player is not None
        info: dict = {"action_id": int(action_id)}

        if self._done:
            info["warning"] = "step called on done environment"
            return self.observe(), 0.0, True, info

        available = self.available_actions()
        spec = available.get(int(action_id))
        if spec is None:
            self.invalid_action_count += 1
            info["invalid_action"] = True
            info["invalid_reason"] = "Action not currently available"
            info["valid_action_ids"] = sorted(available.keys())
            if self.strict_invalid_actions:
                raise ValueError(
                    f"Invalid action {action_id}; valid actions are {sorted(available.keys())}"
                )
            reward = self.INVALID_ACTION_PENALTY
            self.step_count += 1
            return self.observe(), reward, self._done, info

        try:
            reward = self._apply_action(spec, info)
        except Exception as exc:
            self.invalid_action_count += 1
            info["invalid_action"] = True
            info["invalid_reason"] = f"Action execution failed: {type(exc).__name__}: {exc}"
            info["valid_action_ids"] = sorted(available.keys())
            if self.strict_invalid_actions:
                raise
            reward = self.INVALID_ACTION_PENALTY

        self.last_action = int(action_id)
        self.step_count += 1
        self._done = self._is_done()
        obs = self.observe()
        done = self._done
        return obs, reward, done, info

    def available_actions(self) -> dict[int, ActionSpec]:
        assert self.player is not None
        actions: dict[int, ActionSpec] = {}

        actions[self.ACTION_BASE["end_turn"]] = ActionSpec(
            action_id=self.ACTION_BASE["end_turn"],
            name="end_turn",
            args=(),
            description="End shop turn, run battle, and start next turn.",
        )

        if self.player.gold >= 1:
            aid = self.ACTION_BASE["roll"]
            actions[aid] = ActionSpec(
                action_id=aid,
                name="roll",
                args=(),
                description="Roll the shop for 1 gold.",
            )

        team_filled = self._team_filled_indices()
        team_name_to_indices = self._team_name_to_indices()

        for shop_idx, slot in enumerate(self.player.shop.slots):
            if slot.slot_type == "pet":
                if len(team_filled) < 5 and slot.cost <= self.player.gold:
                    aid = self.ACTION_BASE["buy_pet"] + shop_idx
                    actions[aid] = ActionSpec(
                        action_id=aid,
                        name="buy_pet",
                        args=(shop_idx,),
                        description=f"Buy pet at shop index {shop_idx}.",
                    )

                pet_name = slot.obj.name
                if pet_name in team_name_to_indices and slot.cost <= self.player.gold:
                    for team_idx in team_name_to_indices[pet_name]:
                        aid = self.ACTION_BASE["buy_combine"] + (shop_idx * 5) + team_idx
                        actions[aid] = ActionSpec(
                            action_id=aid,
                            name="buy_combine",
                            args=(shop_idx, team_idx),
                            description=f"Buy-combine pet at shop {shop_idx} into team slot {team_idx}.",
                        )

            if (
                slot.slot_type == "food"
                and slot.cost <= self.player.gold
                and team_filled
                and slot.obj.name not in self.UNSAFE_PURCHASE_FOODS
            ):
                if self._food_is_auto_target(slot.obj.name):
                    aid = self.ACTION_BASE["buy_food_auto"] + shop_idx
                    actions[aid] = ActionSpec(
                        action_id=aid,
                        name="buy_food_auto",
                        args=(shop_idx,),
                        description=f"Buy auto-target food at shop index {shop_idx}.",
                    )
                else:
                    for team_idx in team_filled:
                        aid = self.ACTION_BASE["buy_food_target"] + (shop_idx * 5) + team_idx
                        actions[aid] = ActionSpec(
                            action_id=aid,
                            name="buy_food_target",
                            args=(shop_idx, team_idx),
                            description=f"Buy targeted food at shop {shop_idx} for team slot {team_idx}.",
                        )

            # The backend rejects freeze operations after a purchase has left a
            # shop slot empty, so empty slots must not be exposed as actions.
            if slot.empty:
                continue
            if slot.frozen:
                aid = self.ACTION_BASE["unfreeze"] + shop_idx
                actions[aid] = ActionSpec(
                    action_id=aid,
                    name="unfreeze",
                    args=(shop_idx,),
                    description=f"Unfreeze shop slot {shop_idx}.",
                )
            else:
                aid = self.ACTION_BASE["freeze"] + shop_idx
                actions[aid] = ActionSpec(
                    action_id=aid,
                    name="freeze",
                    args=(shop_idx,),
                    description=f"Freeze shop slot {shop_idx}.",
                )

        for team_idx in team_filled:
            aid = self.ACTION_BASE["sell"] + team_idx
            actions[aid] = ActionSpec(
                action_id=aid,
                name="sell",
                args=(team_idx,),
                description=f"Sell team pet at index {team_idx}.",
            )

        combo_pairs = list(itertools.combinations(range(5), 2))
        for pair_idx, (a, b) in enumerate(combo_pairs):
            if self.player.team[a].empty or self.player.team[b].empty:
                continue
            if self.player.team[a].pet.name != self.player.team[b].pet.name:
                continue
            aid = self.ACTION_BASE["combine"] + pair_idx
            actions[aid] = ActionSpec(
                action_id=aid,
                name="combine",
                args=(a, b),
                description=f"Combine team slot {a} with slot {b}.",
            )

        return actions

    def _apply_action(self, spec: ActionSpec, info: dict) -> float:
        assert self.player is not None

        if spec.name == "end_turn":
            self.player.end_turn()
            reward = self._resolve_battle_and_progress()
        elif spec.name == "roll":
            self.player.roll()
            reward = 0.0
        elif spec.name == "buy_pet":
            self.player.buy_pet(spec.args[0])
            reward = 0.0
        elif spec.name == "buy_combine":
            self.player.buy_combine(spec.args[0], spec.args[1])
            reward = 0.0
        elif spec.name == "buy_food_auto":
            self.player.buy_food(spec.args[0])
            reward = 0.0
        elif spec.name == "buy_food_target":
            self.player.buy_food(spec.args[0], spec.args[1])
            reward = 0.0
        elif spec.name == "sell":
            self.player.sell(spec.args[0])
            reward = 0.0
        elif spec.name == "combine":
            self.player.combine(spec.args[0], spec.args[1])
            reward = 0.0
        elif spec.name == "freeze":
            self.player.freeze(spec.args[0])
            reward = 0.0
        elif spec.name == "unfreeze":
            self.player.unfreeze(spec.args[0])
            reward = 0.0
        else:
            raise ValueError(f"Unknown action name: {spec.name}")

        info["invalid_action"] = False
        info["action_name"] = spec.name
        info["action_args"] = list(spec.args)
        return reward

    def _resolve_battle_and_progress(self) -> float:
        assert self.player is not None
        opponent = self.opponents[min(self.player.turn - 1, len(self.opponents) - 1)]
        battle_result = self.Battle(self.player.team, opponent).battle()
        self._apply_fight_outcome(battle_result)
        if not self._is_done():
            self.player.start_turn()
            self._apply_generated_artifact_injection()
        # Dense but simple reward shaping for Phase 1 integration checks.
        return (self.player.wins / 10.0) - (self.invalid_action_count * 0.0)

    def _apply_fight_outcome(self, outcome: int) -> None:
        assert self.player is not None
        if outcome == 0:
            self.player.lf_winner = True
            self.player.wins += 1
        elif outcome == 2:
            self.player.lf_winner = False
        elif outcome == 1:
            self.player.lf_winner = False
            if self.player.turn <= 2:
                self.player.lives -= 1
            elif self.player.turn <= 4:
                self.player.lives -= 2
            else:
                self.player.lives -= 3
            self.player.lives = max(self.player.lives, 0)

    def _is_done(self) -> bool:
        assert self.player is not None
        return (
            self.player.wins >= 10
            or self.player.lives <= 0
            or self.player.turn >= self.max_turn
        )

    def _generate_opponents(self, num_turns: int) -> list[object]:
        opps: list[object] = []
        for turn in range(1, num_turns + 1):
            avail_pets = self.get_shop_rules(min(turn, 11), pack=self.pack)[4]
            avail_pets = [name for name in avail_pets if name != "pet-none"]
            safe_avail = [name for name in avail_pets if name in self.SAFE_OPPONENT_PETS]
            if safe_avail:
                avail_pets = safe_avail
            team_size = int(self._rng.randint(1, 6))
            sampled = self._rng.choice(avail_pets, size=team_size, replace=True).tolist()
            opps.append(self.Team(sampled, seed_state=self._rng.get_state()))
        return opps

    def _apply_generated_artifact_injection(self) -> None:
        if not self.injected_artifacts:
            self.last_injection_record = None
            self.last_injection_records = []
            return
        assert self.player is not None
        try:
            if len(self.injected_artifacts) == 1:
                record = inject_artifact_into_player_shop(self.player, self.injected_artifacts[0])
                self.last_injection_record = record
                self.last_injection_records = [record] if record is not None else []
            else:
                self.last_injection_records = inject_artifacts_into_player_shop(self.player, self.injected_artifacts)
                self.last_injection_record = self.last_injection_records[0] if self.last_injection_records else None
        except UnsupportedArtifactSpecError:
            raise

    def _team_filled_indices(self) -> list[int]:
        assert self.player is not None
        return [idx for idx, slot in enumerate(self.player.team.slots) if not slot.empty]

    def _team_name_to_indices(self) -> dict[str, list[int]]:
        assert self.player is not None
        mapping: dict[str, list[int]] = {}
        for idx, slot in enumerate(self.player.team.slots):
            if slot.empty:
                continue
            mapping.setdefault(slot.pet.name, []).append(idx)
        return mapping

    def _food_is_auto_target(self, food_name: str) -> bool:
        effect = self.data["foods"][food_name]["ability"]["effect"]
        if food_name == "food-canned-food":
            return True
        target = effect.get("target")
        if isinstance(target, dict) and target.get("kind") == "RandomFriend":
            return True
        return False

    @staticmethod
    def _state_hash(state: dict) -> str:
        canonical = json.dumps(state, sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode("utf-8")).hexdigest()


def random_valid_action_id(env: FrozenArenaEnv, rng: np.random.RandomState) -> int:
    available_ids = sorted(env.available_actions().keys())
    idx = int(rng.randint(0, len(available_ids)))
    return available_ids[idx]


def play_random_episode(
    env: FrozenArenaEnv,
    action_seed: int,
    max_steps: int = 256,
) -> dict:
    obs = env.reset()
    action_rng = np.random.RandomState(action_seed)
    trajectory: list[dict] = []

    for _ in range(max_steps):
        action_id = random_valid_action_id(env, action_rng)
        obs, reward, done, info = env.step(action_id)
        trajectory.append(
            {
                "action_id": action_id,
                "state_hash": obs["state_hash"],
                "turn": obs["turn"],
                "wins": obs["wins"],
                "lives": obs["lives"],
                "reward": reward,
                "done": done,
                "invalid_action": info.get("invalid_action", False),
            }
        )
        if done:
            break

    return {
        "final_observation": obs,
        "trajectory": trajectory,
        "steps": len(trajectory),
    }
