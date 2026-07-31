"""Simple deterministic heuristic policy for baseline bootstrapping."""

from __future__ import annotations

from sap_thesis_env import FrozenArenaEnv


class BasicHeuristicPolicy:
    """Greedy baseline policy: buy first affordable item, roll, or end the turn."""

    def select_action(self, env: FrozenArenaEnv) -> int:
        actions = env.available_actions()
        # Prefer buy actions in deterministic id order.
        for action_id in sorted(actions):
            if actions[action_id].name in {"buy_pet", "buy_combine", "buy_food_target", "buy_food_auto"}:
                return action_id
        for action_id in sorted(actions):
            if actions[action_id].name == "roll":
                return action_id
        return min(actions.keys())
