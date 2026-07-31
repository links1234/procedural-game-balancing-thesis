"""Random action policies for frozen arena environment integration tests."""

from __future__ import annotations

import numpy as np

from sap_thesis_env import FrozenArenaEnv, random_valid_action_id


class RandomActionPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.RandomState(seed)

    def select_action(self, env: FrozenArenaEnv) -> int:
        return random_valid_action_id(env, self.rng)
