"""
Exact-k capture (E4, k_max): with k_req = k_max = 3,
  2 in-range capturers -> undershoot, 3 -> capture, 4 -> overkill (no capture,
  fail_rew); k_max=None keeps the original >= k_req rule (4 -> capture).

Run:  python test_exact_k.py
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multiagent.MPE_env import GraphMPEEnv
from test_navigation_joint_graph import check, make_args, place, step_with


def fire_n(k_max, n_fire):
    env = GraphMPEEnv(make_args(num_agents=5, k_req=3, k_max=k_max))
    env.reset()
    tgt = env.world.landmarks[0].state.p_pos
    for i in range(5):
        place(env, i, tgt + np.array([0.01 * i, 0.0]))
    before = int(env.world.target_active.sum())
    _, _, _, _, reward_n, _, info_n = step_with(env, set(range(n_fire)))
    after = int(env.world.target_active.sum())
    return before - after, float(reward_n[0][0]), info_n[0]


def main():
    ok = True
    print("== exact-k, k_req = k_max = 3 ==")
    cap, rew, info = fire_n(3, 2)
    ok &= check("2 fire: undershoot", cap == 0 and info["undershoots"] == 1
                and info["overkills"] == 0 and np.isclose(rew, -1.0), f"{cap} {rew} {info}")
    cap, rew, info = fire_n(3, 3)
    ok &= check("3 fire: capture", cap == 1 and info["joint_captures"] == 1
                and info["overkills"] == 0 and np.isclose(rew, 10.0), f"{cap} {rew}")
    cap, rew, info = fire_n(3, 4)
    ok &= check("4 fire: overkill, no capture", cap == 0 and info["overkills"] == 1
                and info["undershoots"] == 0 and np.isclose(rew, -1.0), f"{cap} {rew} {info}")
    cap, rew, info = fire_n(3, 5)
    ok &= check("5 fire: overkill", cap == 0 and info["overkills"] == 1, f"{cap} {info}")

    print("\n== k_max=None: original rule ==")
    cap, rew, info = fire_n(None, 4)
    ok &= check("4 fire: capture", cap == 1 and info["overkills"] == 0
                and np.isclose(rew, 10.0), f"{cap} {rew}")

    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
