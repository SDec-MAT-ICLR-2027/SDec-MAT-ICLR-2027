"""How often do k agents meet at one target, per env config?

The failed radius runs learned "walk to nearest visible target, hold, never
fire" and never saw enough READY events (>= k_req agents in range of the same
live target) to discover the joint capture. This measures READY/episode for
exactly that behaviour (and for random play) across candidate configs, plus
P(all agents connected | >= 2 targets alive) so the partial-connectivity
constraint stays visible.

  hold    obs-limited: go to the nearest target within R and hold in capture
          range; nothing within R -> sticky-direction search. Never fires.
  random  uniform actions.

Reference: the working shared-env CT-SDE run reached ~9 READY/ep by update 50.

  python probe_radius_assembly.py
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multiagent.MPE_env import GraphMPEEnv
from probe_radius_connectivity import T, components, make_args, move_toward, onehot

N_EPS = 60
# (label, world_size, R, sep, n_agents, n_targets, k_req)
CONFIGS = (
    ("current", 3.0, 0.6, 1.25, 6, 3, 3),
    ("2x2 sep1.25", 2.0, 0.6, 1.25, 6, 3, 3),
    ("2x2 sep0.6", 2.0, 0.6, 0.60, 6, 3, 3),
    ("3x3 2 targets", 3.0, 0.6, 1.25, 6, 2, 3),
    ("2x2 2 targets", 2.0, 0.6, 1.25, 6, 2, 3),
    ("3x3 k=2", 3.0, 0.6, 1.25, 6, 3, 2),
    ("full-view ref", 2.0, 10.0, 0.60, 6, 3, 3),
)


def hold_actions(env, sc, R, dirs, rng, ws):
    w = env.world
    acts = []
    for ag in w.agents:
        best, bd = None, np.inf
        for t in range(sc.num_targets):
            if not w.target_active[t]:
                continue
            d = np.linalg.norm(w.landmarks[t].state.p_pos - ag.state.p_pos)
            if d <= R and d < bd:
                best, bd = t, d
        if best is not None:
            if bd <= sc.capture_radius * 0.8:
                acts.append(onehot(0))
            else:
                rel = w.landmarks[best].state.p_pos - ag.state.p_pos
                acts.append(onehot(move_toward(rel, ag.state.p_vel)))
            continue
        lim = 0.85 * ws / 2
        dx = {1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}[dirs[ag.id]]
        if np.abs(ag.state.p_pos + 0.3 * np.array(dx)).max() > lim or rng.rand() < 0.05:
            dirs[ag.id] = rng.randint(1, 5)
        acts.append(onehot(dirs[ag.id]))
    return acts


def run(cfg, policy):
    _, ws, R, sep, n, m, k = cfg
    a = make_args(max(R, 0.6))
    a.world_size, a.min_target_sep = ws, sep
    a.num_agents, a.num_targets, a.k_req = n, m, k
    env = GraphMPEEnv(a)
    sc = env.reward_callback.__self__
    rng = np.random.RandomState(0)
    np.random.seed(0)
    ready, full2 = [], []
    for _ in range(N_EPS):
        env.reset()
        dirs = rng.randint(1, 5, size=n)
        r_ep = 0
        for _t in range(T):
            for t in range(m):
                if env.world.target_active[t] and sc._agents_in_range(env.world, t).sum() >= k:
                    r_ep += 1
            if int(np.sum(env.world.target_active)) >= 2:
                sizes = components(sc.agent_radius_adjacency(env.world))
                full2.append(float(max(sizes) == n))
            if policy == "random":
                acts = [onehot(rng.randint(6)) for _ in range(n)]
            else:
                acts = hold_actions(env, sc, R, dirs, rng, ws)
            env.step(acts)
        ready.append(r_ep)
    return np.mean(ready), (np.mean(full2) if full2 else float("nan"))


def main():
    print("%-15s %4s %4s %5s %3s %3s %3s | %12s %12s | %10s" % (
        "config", "ws", "R", "sep", "N", "M", "k", "READY random", "READY hold",
        "P(full|2+)"))
    print("-" * 90)
    for cfg in CONFIGS:
        r_rand, _ = run(cfg, "random")
        r_hold, f2 = run(cfg, "hold")
        label, ws, R, sep, n, m, k = cfg
        print("%-15s %4.1f %4.1f %5.2f %3d %3d %3d | %12.2f %12.2f | %10.3f" % (
            label, ws, R, sep, n, m, k, r_rand, r_hold, f2), flush=True)


if __name__ == "__main__":
    main()
