#!/usr/bin/env python3
"""Random policy stress test for environment stability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from sap_thesis_env import FrozenArenaEnv, random_valid_action_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test the frozen arena env.")
    parser.add_argument("--steps", type=int, default=10000, help="Total environment steps")
    parser.add_argument("--env-seed", type=int, default=12345, help="Environment seed")
    parser.add_argument("--action-seed", type=int, default=2026, help="Action policy seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = FrozenArenaEnv(seed=args.env_seed)
    obs = env.reset()
    policy_rng = np.random.RandomState(args.action_seed)
    resets = 0

    for step in range(args.steps):
        action_id = random_valid_action_id(env, policy_rng)
        obs, reward, done, info = env.step(action_id)
        if done:
            env.reset(seed=args.env_seed + resets + 1)
            resets += 1

        if step and step % 1000 == 0:
            print(
                f"[INFO] step={step} resets={resets} "
                f"invalid_actions={env.invalid_action_count} "
                f"turn={obs['turn']} wins={obs['wins']} lives={obs['lives']}"
            )

    ok, reason = env.validate_action_mask()
    if not ok:
        print(f"[FAIL] Action mask validation failed: {reason}")
        return 1

    print(
        f"[OK] Stress test completed: steps={args.steps}, resets={resets}, "
        f"invalid_actions={env.invalid_action_count}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
