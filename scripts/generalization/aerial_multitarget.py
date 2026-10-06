#!/usr/bin/env python3
"""Zero-shot concurrent multi-target test for SDec-MAT on Aerial Ordering.

A policy trained on ONE sensor + 5 workers + ONE target is frozen and dropped
into an arena with T sensors, T targets and N workers at once, every worker
permanently assigned to one target (mac/envs/aerial_ordering_multitarget.py).
Each team must reach its own crate and commit exactly K_t workers. Nothing is
retrained.

Communication is physical and NOT separated by team: a worker within range of
another team's sensor joins that sensor's group, so two teams can end up in one
decode chain and the policy has to ignore the other team's agents itself.

Conditions form a ladder, each adding one restriction:
  random/shared        uneven teams (K_t ~ U[2, n_t - 1]), physical chains
  random/segregated    + links between different teams cut in the harness
  balanced/segregated  + every team is the trained 1 sensor + 5 workers, K in [2, 4]

    python scripts/generalization/aerial_multitarget.py --seed 42
    python scripts/generalization/aerial_multitarget.py --checkpoint <run dir or .pt>
    python scripts/generalization/aerial_multitarget.py --seed 42 --configs 2x10,3x15,4x20

Reported per (T x N, condition): simultaneous success (every target lifted in
the same episode), per-target success, return, and the episode-level success of
episodes in which some team committed while its chain held another team.
"""
import argparse
import contextlib
import csv
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policies as P  # noqa: E402

import mac  # noqa: E402
from mac.envs import AerialOrderingMultiTargetEvalEnv  # noqa: E402

TRAINED = mac.PAPER["aerial_ordering"]
ENV_CFG = {k: TRAINED[k] for k in ("k_min", "k_max", "comm_radius", "sense_radius",
                                   "wind_prob", "time_limit")}
LADDER_LABEL = {
    ("random", "shared"): "as deployed: uneven teams, shared chains",
    ("random", "segregated"): "  + chains segregated by target",
    ("balanced", "segregated"): "  + trained team: 1 sensor + 5 workers, K[2,4]",
    ("balanced", "shared"): "  + trained team, chains still shared",
}


def scaled_grid(n_targets):
    """Arena grows by 5 cells per extra target, keeping occupancy comparable."""
    return TRAINED["grid_size"] + 5 * (n_targets - 1)


def run_cell(pol, n_targets, n_workers, grid, assign, comm, seed, episodes):
    with contextlib.redirect_stdout(io.StringIO()):
        env = AerialOrderingMultiTargetEvalEnv(n_targets=n_targets, n_workers=n_workers,
                                               grid_size=grid, assign_mode=assign,
                                               seed=seed, **ENV_CFG)
    env.seed(seed)
    sim, ret = np.zeros(episodes), np.zeros(episodes)
    lifted = total = 0
    mixed_ep = mixed_ok = clean_ep = clean_ok = 0
    for ep in range(episodes):
        env.reset()
        done, info, had_mixed = False, {}, False
        while not done:
            obs = np.stack(env.get_obs()).astype(np.float32)
            avail = np.asarray(env.get_avail_actions(), np.float32)
            adj = env.comm_adjacency()
            if comm == "segregated":            # cut every link between different teams
                team = np.asarray(env.assigned)
                adj = adj * (team[:, None] == team[None, :])
            gm = pol.group_mask(adj)
            mixed_now = {}
            for ai in range(env.n_agents):      # does my chain hold another team?
                t = int(env.assigned[ai])
                comp = np.flatnonzero(gm[0, ai].numpy())
                mixed_now[t] = mixed_now.get(t, False) or any(env.assigned[j] != t for j in comp)
            pending = [t for t in range(env.n_targets) if not env._resolved(t)]
            a = pol.act(obs, avail=avail, group_mask=gm)
            _, r, term, trunc, info = env.step(a.tolist())
            ret[ep] += r
            done = term or trunc
            had_mixed |= any(env._resolved(t) and mixed_now.get(t, False) for t in pending)
        sim[ep] = info["simultaneous_success"]
        lifted += int(sum(info["per_target_lifted"]))
        total += env.n_targets
        if had_mixed:
            mixed_ep, mixed_ok = mixed_ep + 1, mixed_ok + int(sim[ep])
        else:
            clean_ep, clean_ok = clean_ep + 1, clean_ok + int(sim[ep])
    return dict(simultaneous=float(sim.mean()), per_target=lifted / total,
                return_mean=float(ret.mean()), mixed_episodes=mixed_ep,
                mixed_episode_success=mixed_ok / mixed_ep if mixed_ep else float("nan"),
                clean_episodes=clean_ep,
                clean_episode_success=clean_ok / clean_ep if clean_ep else float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="SDec-MAT run dir, models dir or .pt file")
    src.add_argument("--seed", type=int,
                     help="seed of an SDec-MAT run trained with run_experiment.py")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: newest)")
    ap.add_argument("--n-head", type=int, default=1)
    ap.add_argument("--configs", default="2x10,3x15", help="TxN pairs (targets x workers)")
    ap.add_argument("--fixed-grids", action="store_true",
                    help=f"use --grids for every config instead of growing the arena "
                         f"with T ({TRAINED['grid_size']} + 5 per extra target)")
    ap.add_argument("--grids", default=str(TRAINED["grid_size"]))
    ap.add_argument("--conditions", default="random/shared,random/segregated,balanced/segregated",
                    help="<assignment>/<comm>: random|balanced / shared|segregated")
    ap.add_argument("--episodes", type=int, default=100, help="episodes per env seed")
    ap.add_argument("--eval-seeds", type=int, default=5, help="env seeds per cell")
    ap.add_argument("--eval-seed0", type=int, default=2000)
    ap.add_argument("--out", default=None, help="directory for per-seed CSV + JSON (optional)")
    a = ap.parse_args()

    configs = [tuple(int(x) for x in c.lower().split("x")) for c in a.configs.split(",") if c]
    ladder = [tuple(c.split("/")) for c in a.conditions.split(",") if c]
    for m, cm in ladder:
        if m not in ("random", "balanced") or cm not in ("shared", "segregated"):
            ap.error(f"bad condition {m}/{cm}")
    grids = [int(g) for g in a.grids.split(",") if g]
    seeds = [a.eval_seed0 + i for i in range(a.eval_seeds)]
    path = a.checkpoint or P.default_checkpoint("sdec_mat", "aerial_ordering", a.seed, False)
    pol = P.load_policy("sdec_mat", path, None, a.step, a.n_head)
    print(f"Aerial Ordering multi-target -- {os.path.relpath(pol.path, P.REPO)}")
    print(f"{a.eval_seeds} env seeds x {a.episodes} episodes per cell, greedy")

    def cells():
        for t, n in configs:
            for g in ([scaled_grid(t)] if not a.fixed_grids else grids):
                for m, cm in ladder:
                    yield t, n, g, m, cm

    rec, rows = {}, []
    for key in cells():
        t, n, g, m, cm = key
        per_seed = [run_cell(pol, t, n, g, m, cm, s, a.episodes) for s in seeds]
        rows += [dict(n_targets=t, n_workers=n, grid=g, assign=m, comm=cm, seed=s, **c)
                 for s, c in zip(seeds, per_seed)]
        agg = {}
        for k in per_seed[0]:
            v = np.array([c[k] for c in per_seed], float)
            v = v[~np.isnan(v)]
            agg[k] = float(v.mean()) if v.size else float("nan")
            agg[k + "_std"] = float(v.std()) if v.size else float("nan")
        rec[key] = agg

    head = (f"{'T x N':<8}{'grid':>5}{'assign':>10}{'comm':>12}"
            f"{'simultaneous':>17}{'per-target':>13}{'return':>10}")
    print("\n" + head + "\n" + "-" * len(head))
    prev = None
    for key in cells():
        r, (t, n, g, m, cm) = rec[key], key
        if prev not in (None, (t, n)):
            print()
        prev = (t, n)
        print(f"{f'{t} x {n}':<8}{g:>5}{m:>10}{cm:>12}"
              f"{'%.3f +- %.3f' % (r['simultaneous'], r['simultaneous_std']):>17}"
              f"{r['per_target']:>13.3f}{r['return_mean']:>+10.2f}")

    grid_of = {(t, n): (scaled_grid(t) if not a.fixed_grids else grids[0]) for t, n in configs}
    print("\nEach row adds one restriction to the one above (simultaneous success, gain):")
    print(f"{'condition':<46}" + "".join(f"{f'{t} x {n} @ {grid_of[(t, n)]}':>18}"
                                          for t, n in configs))
    for i, (m, cm) in enumerate(ladder):
        line = ""
        for t, n in configs:
            v = rec[(t, n, grid_of[(t, n)], m, cm)]["simultaneous"]
            if i == 0:
                line += f"{v:>18.3f}"
            else:
                d = v - rec[(t, n, grid_of[(t, n)], *ladder[i - 1])]["simultaneous"]
                line += f"{f'{v:.3f} ({d:+.3f})':>18}"
        print(f"{LADDER_LABEL.get((m, cm), f'{m}/{cm}'):<46}{line}")

    print("\nEpisodes in which a team committed while its decode chain held another team")
    print("(mean count per env seed, and the episode-level success of those vs the rest):")
    head = (f"{'T x N':<8}{'grid':>5}{'assign':>10}{'comm':>12}{'mixed eps':>11}"
            f"{'success':>10}{'clean eps':>11}{'success':>10}")
    print(head + "\n" + "-" * len(head))
    for key in cells():
        r, (t, n, g, m, cm) = rec[key], key
        ms = "-" if np.isnan(r["mixed_episode_success"]) else f"{r['mixed_episode_success']:.3f}"
        cs = "-" if np.isnan(r["clean_episode_success"]) else f"{r['clean_episode_success']:.3f}"
        print(f"{f'{t} x {n}':<8}{g:>5}{m:>10}{cm:>12}{r['mixed_episodes']:>11.1f}"
              f"{ms:>10}{r['clean_episodes']:>11.1f}{cs:>10}")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "aerial_multitarget.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(a.out, "aerial_multitarget.json"), "w") as f:
            json.dump({"checkpoint": pol.path, "env_cfg": ENV_CFG, "episodes": a.episodes,
                       "env_seeds": seeds,
                       "cells": [dict(n_targets=t, n_workers=n, grid=g, assign=m, comm=cm, **v)
                                 for (t, n, g, m, cm), v in rec.items()]}, f, indent=2)
        print(f"\nwrote {a.out}/aerial_multitarget.csv and .json")


if __name__ == "__main__":
    main()
