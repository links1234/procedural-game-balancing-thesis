#!/usr/bin/env python3
"""Determinism check: same env seed + action seed should yield identical trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sap_thesis_env import FrozenArenaEnv, play_random_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay determinism check")
    parser.add_argument("--trials", type=int, default=100, help="Number of deterministic trials")
    parser.add_argument("--env-seed", type=int, default=12345, help="Environment RNG seed")
    parser.add_argument("--action-seed", type=int, default=2026, help="Action sampling RNG seed")
    parser.add_argument("--max-steps", type=int, default=256, help="Per-episode step cap")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mismatches = 0

    for trial in range(args.trials):
        env_a = FrozenArenaEnv(seed=args.env_seed + trial)
        env_b = FrozenArenaEnv(seed=args.env_seed + trial)

        traj_a = play_random_episode(env_a, action_seed=args.action_seed + trial, max_steps=args.max_steps)
        traj_b = play_random_episode(env_b, action_seed=args.action_seed + trial, max_steps=args.max_steps)

        if traj_a != traj_b:
            mismatches += 1
            print(f"[MISMATCH] trial={trial}")
            print(json.dumps({"a": traj_a, "b": traj_b}, indent=2)[:3000])
            break

    if mismatches == 0:
        print(f"[OK] Deterministic replay passed {args.trials}/{args.trials} trials.")
        return 0

    print(f"[FAIL] Deterministic replay mismatch count: {mismatches}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
