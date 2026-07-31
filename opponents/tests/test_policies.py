from __future__ import annotations

from evaluation.protocol import pick_policy
from opponents.heuristic_policy import BasicHeuristicPolicy
from opponents.learned_policy import ScriptedValuePolicy
from sap_thesis_env.env.frozen_arena_env import FrozenArenaEnv


def test_pick_policy_supports_scripted_value() -> None:
    policy = pick_policy("scripted_value", seed=123)
    assert isinstance(policy, ScriptedValuePolicy)


def test_scripted_value_policy_differs_from_basic_heuristic_on_ranked_shop() -> None:
    env = FrozenArenaEnv(seed=12342)
    env.reset(seed=12342)

    basic_action = BasicHeuristicPolicy().select_action(env)
    scripted_action = ScriptedValuePolicy().select_action(env)

    assert basic_action == 10
    assert scripted_action == 11


def test_scripted_value_policy_runs_without_backend_error() -> None:
    env = FrozenArenaEnv(seed=12345, max_turn=10, strict_invalid_actions=True)
    policy = ScriptedValuePolicy()

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
    assert obs["done"] is True or obs["turn"] >= 1
