"""Radius-env groups read from the obs must equal the ground truth: connected
components (size >= 2) of the scenario's agent-radius graph -- the same graph
CT-SDE communicates over. There is only this one grouping in the radius env.

Run:  python -m mat.test_radius_commit_groups
"""
import sys

import numpy as np

from mat.envs.joint_nav import JointNavMATEnv
from mat.runner.shared.joint_nav_runner import JointNavRunner
from mat.test_decoder_self_only import args as base_args


def components(adj):
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    out = []
    for s in range(n):
        if seen[s]:
            continue
        comp, stack = [], [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.nonzero(adj[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        if len(comp) >= 2:
            out.append(sorted(comp))
    return sorted(out)


def main():
    a = base_args(0)
    a.algorithm_name = "mappo"
    a.world_size, a.min_target_sep, a.sense_radius = 2.0, 1.25, 0.6
    env = JointNavMATEnv(a)
    sc, w = env._scenario, env._world
    rng = np.random.RandomState(0)
    n_steps = n_groups = bad = 0
    for s in range(40):
        env.seed(s)
        obs, _, _ = env.reset()
        for _t in range(50):
            got = sorted(JointNavRunner.radius_comm_groups(obs, 6))
            want = components(sc.agent_radius_adjacency(w))
            n_steps += 1
            n_groups += len(want)
            bad += int(got != want)
            obs, _, _, _, _, _ = env.step(rng.randint(5, size=6))
    ok = bad == 0 and n_groups > 0
    print("  [%s] obs-derived radius groups == ground-truth R-graph components  "
          "--  %d steps, %d groups, %d mismatches" % (
              "PASS" if ok else "FAIL", n_steps, n_groups, bad))
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
