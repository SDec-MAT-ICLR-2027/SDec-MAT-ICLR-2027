#!/usr/bin/env python3
"""Zero-shot generalisation on Pickup Consensus: change who is present, keep the weights.

Trained policies are frozen and evaluated greedily under perturbed teams:

  base   the unperturbed 5-agent team
  T1     agent loss: 1 of the 5 agents drops out; the 4 survivors play a real
         4-agent episode (K in [2,3], obs = [K/4, 4/5])
  T1b    agent loss: 2 drop out; 3 survivors, K = 2
  T2     team split: 10 agents divided at random into two teams of 5 that
         cannot hear each other, each with its own crate and its own K.
         Success is JOINT (both teams exactly right); per-team success is
         reported too.
  sizes  (SDec-MAT only) the same weights on teams of 3..10 agents

How identity enters depends on the method:

  sdec_mat     no ids. A lost agent simply is not there; a split is one forward
               pass over all 10 agents whose communication graph has two
               components, so the teams are interleaved through the decode
               chain and only the graph keeps them apart.
  id methods   (mat_dec, mappo, commnet, tarmac, qmix, informarl, trained with
               agent ids) survivors keep their ORIGINAL one-hot ids, so 4
               survivors hold a random 4 of the 5 ids; for the split, each of
               the 5 ids is issued twice and shuffled across the 10 agents. A
               team that cannot hear the other team is run as its own pass.

    python scripts/generalization/pickup_consensus.py --method sdec_mat --seeds 42 43 44
    python scripts/generalization/pickup_consensus.py --method mappo \\
        --checkpoint results/epymarl/models/mappo_aid_seed42_pickup-consensus-v0_<time>

`--seeds` finds the runs run_experiment.py trains; `--checkpoint` takes any
run directory, step directory or weights file. With several checkpoints the
table reports the mean +- std across them. A checkpoint whose base success is
below --min-base-success never solved the task and is left out: only a solved
policy can be asked whether its solution generalises.
"""
import argparse
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policies as P  # noqa: E402

import mac  # noqa: E402  (policies puts the repo root on sys.path)
from mac.envs import PickupConsensusEnv  # noqa: E402

CFG = mac.PAPER["pickup_consensus"]
N, K_MIN, K_MAX, N_MAX, T_LIMIT = (CFG["n_agents"], CFG["k_min"], CFG["k_max"],
                                   CFG["n_max"], CFG["time_limit"])
CONDITIONS = ("base", "T1", "T1b", "T2")
LABEL = {"base": "base (unperturbed)", "T1": "T1  agent loss 1/5",
         "T1b": "T1b agent loss 2/5", "T2": "T2  team split 5+5"}
N_LIVE = {"base": N, "T1": N - 1, "T1b": N - 2}
ID_POOL = np.repeat(np.arange(N), 2)          # T2: each of the 5 ids issued twice


def make_env(seed, n_agents, k_max=None, n_max=N_MAX):
    """K is capped at team size - 1 at every size, as in the benchmark."""
    with contextlib.redirect_stdout(io.StringIO()):
        return PickupConsensusEnv(
            n_agents=n_agents, k_min=K_MIN,
            k_max=min(K_MAX, n_agents - 1) if k_max is None else k_max,
            n_max=n_max, time_limit=T_LIMIT, seed=seed)


def id_rows(pol, slots):
    """One-hot id rows for agents holding `slots` (empty if trained without ids)."""
    return np.eye(N, dtype=np.float32)[np.asarray(slots)][:, :pol.id_width]


def blank(n):
    return dict(ret=np.zeros(n), succ=np.zeros(n), k=np.zeros(n, dtype=int))


def run_team(pol, env, episodes, rng):
    """base / T1 / T1b / sizes: one fully connected team of env.n_agents."""
    n = env.n_agents
    out = blank(episodes)
    for ep in range(episodes):
        # Which of the original ids survive. Drawn for every method, so every
        # method sees the same episodes.
        slots = np.sort(rng.choice(N, size=min(n, N), replace=False))
        obs, _ = env.reset()
        ret, done, info = 0.0, False, {}
        h = pol.init_hidden(n) if pol.uses_ids else None
        ids = id_rows(pol, slots) if pol.uses_ids else None
        while not done:
            obs = np.asarray(obs, np.float32)
            if pol.uses_ids:
                a, h = pol.act(obs, ids, h)
            else:
                a = pol.act(obs, np.ones((n, n), np.float32))
            obs, r, term, trunc, info = env.step(a.tolist())
            ret += r
            done = term or trunc
        out["ret"][ep], out["succ"][ep], out["k"][ep] = ret, info["success"], info["K"]
    return out


def run_split(pol, envs, episodes, rng):
    """T2: 10 agents, two teams of 5 that cannot hear each other."""
    g = len(envs)
    out = blank(episodes)
    out["team_succ"], out["team_k"] = np.zeros((episodes, g)), np.zeros((episodes, g), int)
    for ep in range(episodes):
        perm = rng.permutation(g * N)
        members = [perm[i * N:(i + 1) * N] for i in range(g)]
        rets, infos = np.zeros(g), [{} for _ in range(g)]
        if pol.uses_ids:
            for i, env in enumerate(envs):            # each team its own pass
                ids = id_rows(pol, ID_POOL[members[i]])
                obs, _ = env.reset()
                h, done = pol.init_hidden(N), False
                while not done:
                    a, h = pol.act(np.asarray(obs, np.float32), ids, h)
                    obs, r, term, trunc, infos[i] = env.step(a.tolist())
                    rets[i] += r
                    done = term or trunc
        else:                                          # one pass over all 10 agents
            adj = np.zeros((g * N, g * N), np.float32)
            for m in members:
                adj[np.ix_(m, m)] = 1.0
            team_obs = [np.asarray(env.reset()[0], np.float32) for env in envs]
            done = [False] * g
            while not all(done):
                obs = np.zeros((g * N, team_obs[0].shape[1]), np.float32)
                for i, m in enumerate(members):
                    obs[m] = team_obs[i]
                a = pol.act(obs, adj)
                for i, env in enumerate(envs):
                    if done[i]:
                        continue                       # resolved team: tokens stay, env stops
                    o, r, term, trunc, infos[i] = env.step(a[members[i]].tolist())
                    team_obs[i] = np.asarray(o, np.float32)
                    rets[i] += r
                    done[i] = term or trunc
        out["ret"][ep] = rets.mean()
        out["team_succ"][ep] = [inf["success"] for inf in infos]
        out["team_k"][ep] = [inf["K"] for inf in infos]
        out["succ"][ep] = float(all(s >= 1.0 for s in out["team_succ"][ep]))
    return out


def per_k(succ, ks):
    return {k: float(succ[ks == k].mean()) for k in range(K_MIN, K_MAX + 1) if (ks == k).any()}


def evaluate(pol, cond, episodes, env_seed, pert_seed):
    rng = np.random.default_rng(pert_seed)
    if cond == "T2":
        envs = [make_env(env_seed + 1000 * i, N) for i in range(2)]
        o = run_split(pol, envs, episodes, rng)
        return dict(ret=o["ret"].mean(), succ=o["succ"].mean(),
                    team=o["team_succ"].mean(),
                    per_k=per_k(o["team_succ"].ravel(), o["team_k"].ravel()))
    o = run_team(pol, make_env(env_seed, N_LIVE[cond]), episodes, rng)
    return dict(ret=o["ret"].mean(), succ=o["succ"].mean(), per_k=per_k(o["succ"], o["k"]))


def mean_std(xs):
    xs = np.asarray(xs, float)
    return xs.mean(), (xs.std(ddof=1) if xs.size > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=P.METHODS)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", nargs="+", help="run dir, step dir or weights file")
    src.add_argument("--seeds", type=int, nargs="+",
                     help="seeds of runs trained with run_experiment.py")
    ap.add_argument("--agent-id", choices=("on", "off"), default=None,
                    help="with --seeds: which run to load (default: on, except sdec_mat)")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: newest)")
    ap.add_argument("--n-head", type=int, default=1, help="attention heads (MAT models)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--min-base-success", type=float, default=0.99)
    ap.add_argument("--sizes", default="3,4,6,7,8,9,10",
                    help="team sizes for the sdec_mat size probe ('' to skip)")
    ap.add_argument("--size-episodes", type=int, default=500)
    a = ap.parse_args()

    conds = [c for c in a.conditions.split(",") if c]
    for c in conds:
        if c not in CONDITIONS:
            ap.error(f"unknown condition {c!r}; choose from {CONDITIONS}")
    if a.checkpoint:
        paths = a.checkpoint
    else:
        aid = (a.method != "sdec_mat") if a.agent_id is None else a.agent_id == "on"
        paths = [P.default_checkpoint(a.method, "pickup_consensus", s, aid) for s in a.seeds]

    obs_dim = make_env(0, N).obs_dim
    pols = [P.load_policy(a.method, p, obs_dim, a.step, a.n_head) for p in paths]
    for pol in pols:
        if pol.uses_ids and pol.id_width not in (0, N):
            raise SystemExit(f"{pol.path}: one-hot width {pol.id_width}, expected {N}")

    # The env's K draws and the perturbation draws come from independent
    # streams, so K is never correlated with which agent is lost.
    env_ss, pert_ss = np.random.SeedSequence(a.eval_seed).spawn(2)
    env_seed, pert_seed = (int(s.generate_state(1)[0]) for s in (env_ss, pert_ss))

    print(f"Pickup Consensus generalisation -- {a.method}, {a.episodes} episodes per condition")
    kept = []
    for pol in pols:
        base = evaluate(pol, "base", a.episodes, env_seed, pert_seed)
        ok = base["succ"] >= a.min_base_success
        print(f"  {'' if ok else '[excluded: base success < %.2f] ' % a.min_base_success}"
              f"{os.path.relpath(pol.path, P.REPO)}   base success {base['succ']:.3f}")
        if ok:
            kept.append((pol, base))
    if not kept:
        raise SystemExit("no checkpoint solved the task; nothing to test")

    rows = {}
    for cond in conds:
        rows[cond] = [b if cond == "base" else evaluate(p, cond, a.episodes, env_seed, pert_seed)
                      for p, b in kept]

    ks = list(range(K_MIN, K_MAX + 1))
    head = (f"{'condition':<22}{'return':>16}{'success':>16}"
            + "".join(f"{'K=' + str(k):>8}" for k in ks))
    print("\n" + head + "\n" + "-" * len(head))
    for cond in conds:
        rs = rows[cond]
        rm, rsd = mean_std([r["ret"] for r in rs])
        sm, ssd = mean_std([r["succ"] for r in rs])
        cells = ""
        for k in ks:
            v = [r["per_k"][k] for r in rs if k in r["per_k"]]
            cells += f"{np.mean(v):>8.3f}" if v else f"{'-':>8}"
        print(f"{LABEL[cond]:<22}{f'{rm:+.2f} +- {rsd:.2f}':>16}"
              f"{f'{sm:.3f} +- {ssd:.3f}':>16}{cells}")
    if "T2" in rows:
        tm, _ = mean_std([r["team"] for r in rows["T2"]])
        print(f"\nT2: success = both teams exactly right; per-team success {tm:.3f}; "
              f"per-K columns are per team.")
    print(f"mean +- std across {len(kept)} checkpoint(s). '-' = K not sampled "
          f"(K is capped at team size - 1).")

    sizes = [int(s) for s in a.sizes.split(",") if s.strip()]
    if sizes and not kept[0][0].uses_ids:
        print(f"\nSame weights, different team size ({a.size_episodes} episodes each; "
              f"trained on {N}):")
        print(f"{'agents':<8}{'K range':<10}{'return':>16}{'success':>16}")
        for n in sizes:
            res = [run_team(p, make_env(env_seed, n, k_max=n - 1, n_max=max(N_MAX, n)),
                            a.size_episodes, np.random.default_rng(pert_seed))
                   for p, _ in kept]
            rm, rsd = mean_std([o["ret"].mean() for o in res])
            sm, ssd = mean_std([o["succ"].mean() for o in res])
            print(f"{n:<8}{f'[{K_MIN},{n - 1}]':<10}{f'{rm:+.2f} +- {rsd:.2f}':>16}"
                  f"{f'{sm:.3f} +- {ssd:.3f}':>16}")
        print(f"Above {N_MAX} agents the team-size field of the observation saturates at "
              f"1.0, so the team size is no longer legible from it.")


if __name__ == "__main__":
    main()
