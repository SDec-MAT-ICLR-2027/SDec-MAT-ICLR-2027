"""Is the radius env solvable from the radius observation alone?

Env-side vs algorithm-side triage for the failed CT-SDE radius run.

  vis stats   under random play: share of agent-steps that see >= 1 live
              target / >= 1 other agent; share of agents seeing a target at t=0
  obs-only    a heuristic that reads ONLY the agent's radius obs:
                visible live target  -> go to nearest visible one
                in capture range and >= k_req-1 visible others also within
                capture range of that target -> CAPTURE, else hold
                nothing visible      -> sticky-direction search (pos from obs)
  state split the privileged scripted policy (reference upper bound)

If obs-only captures well, the env is solvable from obs -> algorithm side.

  python probe_radius_solvable.py
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multiagent.MPE_env import GraphMPEEnv
from probe_radius_connectivity import (CAPTURE, T, make_args, move_toward,
                                       onehot, split_actions)

N, M, K = 6, 3, 3
TB = 4 + 3 * (N - 1)          # first target slot
N_EPS = 60


def parse(o):
    others = [(o[4 + 3 * k:6 + 3 * k], o[6 + 3 * k]) for k in range(N - 1)]
    tg = [(o[TB + 4 * t:TB + 4 * t + 2], o[TB + 4 * t + 2], o[TB + 4 * t + 3])
          for t in range(M)]
    return o[0:2], o[2:4], others, tg, o[TB + 4 * M]


class ObsOnly:
    def __init__(self, ws, rng):
        self.ws, self.rng = ws, rng
        self.dir = rng.randint(1, 5, size=N)

    def act(self, obs):
        acts = []
        for i, o in enumerate(obs):
            vel, pos, others, tg, in_cap = parse(o)
            live = [(np.linalg.norm(rel), rel) for rel, alive, vis in tg
                    if vis > 0.5 and alive > 0.5]
            if live:
                d, rel = min(live, key=lambda x: x[0])
                if d <= 0.3:
                    n_in = sum(1 for orel, ov in others if ov > 0.5
                               and np.linalg.norm(orel - rel) <= 0.3)
                    acts.append(onehot(CAPTURE if n_in >= K - 1 else 0))
                else:
                    acts.append(onehot(move_toward(rel, vel)))
                continue
            # search: sticky direction, turn at walls or with small prob
            lim = 0.85 * self.ws / 2
            dx = {1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}[self.dir[i]]
            nxt = pos + 0.3 * np.array(dx)
            if np.abs(nxt).max() > lim or self.rng.rand() < 0.05:
                self.dir[i] = self.rng.randint(1, 5)
            acts.append(onehot(self.dir[i]))
        return acts


def run(ws, R, sep, policy):
    a = make_args(R)
    a.world_size, a.min_target_sep = ws, sep
    env = GraphMPEEnv(a)
    sc = env.reward_callback.__self__
    rng = np.random.RandomState(0)
    np.random.seed(0)
    caps, see_t, see_a, see_t0, rets = [], [], [], [], []
    for _ in range(N_EPS):
        obs, _, _, _ = env.reset()
        pol = ObsOnly(ws, rng)
        ret = 0.0
        for t in range(T):
            for o in obs:
                _, _, others, tg, _ = parse(o)
                st = any(v > 0.5 and al > 0.5 for _, al, v in tg)
                see_t.append(st)
                see_a.append(any(v > 0.5 for _, v in others))
                if t == 0:
                    see_t0.append(st)
            if policy == "random":
                acts = [onehot(rng.randint(6)) for _ in range(N)]
            elif policy == "obs-only":
                acts = pol.act(obs)
            else:
                acts = split_actions(env, sc)
            # GraphMPEEnv.step -> obs, agent_id, node_obs, adj, rew, done, info
            obs, _, _, _, rew, _, info = env.step(acts)
            ret += float(np.mean(rew))
        caps.append(info[0]["joint_captures"])
        rets.append(ret)
    return (np.mean(caps), np.mean(rets), np.mean(see_t), np.mean(see_a),
            np.mean(see_t0))


def main():
    print("%4s %4s %5s %-11s %6s %8s %9s %9s %9s" % (
        "ws", "R", "sep", "policy", "caps", "return", "seeTarget", "seeAgent",
        "seeTgt@t0"))
    print("-" * 76)
    for ws, R, sep in ((3.0, 0.6, 1.25), (2.0, 0.6, 1.25), (2.0, 1.0, 0.6)):
        for pol in ("random", "obs-only", "state-split"):
            c, r, st, sa, s0 = run(ws, R, sep, pol)
            print("%4.1f %4.1f %5.2f %-11s %6.2f %8.2f %9.3f %9.3f %9.3f" % (
                ws, R, sep, pol, c, r, st, sa, s0))


if __name__ == "__main__":
    main()
