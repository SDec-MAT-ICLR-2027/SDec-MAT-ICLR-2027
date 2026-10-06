"""Is target placement biased by target id?

Targets are placed sequentially by rejection sampling with min_target_sep.
With a large separation on a small board, target 0 is placed freely while
later targets are pushed away from it, so target 0 may end up more central
and more often the nearest target to the agents at spawn.

Per layout (2000 resets, env only, no policy), per target id:
  d_center   mean distance from the board centre
  nearest    P(it is the nearest target for a random agent at spawn)
  near>=3    P(it is the nearest target for >= 3 of the 6 agents at spawn)

  python probe_target_placement.py
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multiagent.MPE_env import GraphMPEEnv
from probe_radius_connectivity import make_args

N_RESETS = 2000
LAYOUTS = (("radius 2x2 sep1.25", 2.0, 0.6, 1.25),
           ("shared 2x2 sep0.6", 2.0, 0.6, 0.6))


def main():
    print("%-20s %3s %9s %9s %9s" % ("layout", "id", "d_center", "nearest",
                                     "near>=3"))
    print("-" * 56)
    for name, ws, R, sep in LAYOUTS:
        a = make_args(R)
        a.world_size, a.min_target_sep = ws, sep
        env = GraphMPEEnv(a)
        w = env.world
        n_t = 3
        dc = np.zeros(n_t)
        nearest = np.zeros(n_t)
        near3 = np.zeros(n_t)
        np.random.seed(0)
        for _ in range(N_RESETS):
            env.reset()
            T = np.array([w.landmarks[t].state.p_pos for t in range(n_t)])
            P = np.array([ag.state.p_pos for ag in w.agents])
            dc += np.linalg.norm(T, axis=1)
            nn_ = np.linalg.norm(P[:, None] - T[None], axis=-1).argmin(1)
            for t in range(n_t):
                nearest[t] += (nn_ == t).mean()
                near3[t] += float((nn_ == t).sum() >= 3)
        for t in range(n_t):
            print("%-20s %3d %9.3f %9.3f %9.3f" % (
                name, t, dc[t] / N_RESETS, nearest[t] / N_RESETS,
                near3[t] / N_RESETS))


if __name__ == "__main__":
    main()
