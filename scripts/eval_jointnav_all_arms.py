"""Post-hoc greedy/sampled evaluation of EVERY joint-nav arm, including InforMARL.

`mat/probe_penalty_eval.py` only understands MAT checkpoints (`transformer_*.pt`
through `TransformerPolicy`). InforMARL stores `actor.pt` / `critic.pt` for a
completely different network, so it was excluded from every eval table so far.
This script evaluates both, over the SAME fixed eval seeds, and prints one table.

Fairness of the comparison rests on two facts, both checked in code:
  * Both wrappers seed with `np.random.seed(s)` before `reset()`
    (`multiagent/environment.py:158`, `joint_nav_env.py:269`), and both build the
    world with the same `navigation_joint_graph.random_scenario`, so seed s gives
    byte-identical agent and target layouts in either codebase.
  * The episode tallies read out of `info` (`joint_captures`, `undershoots`,
    `overkills`, `capture_actions`) are produced by the scenario, not the wrapper.
    MAT stops the episode as soon as the board is clear while InforMARL always
    runs the full horizon, but post-clear steps earn nothing and change no tally.

Usage
    python scripts/eval_jointnav_all_arms.py --task e4 --prefix e4fair
    python scripts/eval_jointnav_all_arms.py --task e2 --prefix e2fair --episodes 60
    python scripts/eval_jointnav_all_arms.py --task e4 --prefix e4fair --only ctsde,dec
"""
import argparse
import glob
import io
import os
import re
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAT_ROOT = os.path.join(REPO, "Multi-Agent-Transformer_ctsde")
INF_ROOT = os.path.join(REPO, "InforMARL")
for p in (MAT_ROOT, INF_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

N_AGENTS, N_TARGETS, K_REQ, FIRE = 6, 3, 3, 5
SEED0 = 50000

# task -> (fail_rew, k_max)
TASKS = {
    "e2":   (-1.0, None),
    "e2p":  (-5.0, None),
    "e4":   (-5.0, 3),
    "e4m1": (-1.0, 3),
}
# arm -> (kind for the MAT wrapper, results sub-directory)
MAT_ARMS = {
    "ctsde": ("ctsde", "mat_comm_soft"),
    "dec":   ("dec",   "mat_comm_dec"),
    "mappo": ("mappo", "mappo"),
}


def wandb_dir(ckpt_path):
    """The wandb `files/` directory a checkpoint lives in."""
    return os.path.dirname(os.path.abspath(ckpt_path))


def recorded_argv(ckpt_path):
    """The exact command line this run was launched with, from wandb's config.yaml.

    Architecture flags differ between run generations -- the 1M InforMARL lane
    used `layer_N 1` while the e2fair/e4fair lanes use 2 -- so a hardcoded
    namespace builds the wrong network and `load_state_dict` fails with missing
    keys. Re-parsing the recorded argv reproduces each run's own namespace.
    Returns None when the file is absent or has no argv (then the caller falls
    back to its constructed default).
    """
    cfg = os.path.join(wandb_dir(ckpt_path), "config.yaml")
    if not os.path.isfile(cfg):
        return None
    try:
        import yaml
        with io.open(cfg, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        entries = doc.get("_wandb", {}).get("value", {}).get("e", {})
        for payload in entries.values():
            argv = payload.get("args")
            if argv:
                return [str(x) for x in argv]
    except Exception:
        return None
    return None


def wandb_values(ckpt_path):
    """Flat {key: value} of a run's wandb config (top-level `k: {value: v}`)."""
    cfg = os.path.join(wandb_dir(ckpt_path), "config.yaml")
    if not os.path.isfile(cfg):
        return {}
    try:
        import yaml
        with io.open(cfg, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
    except Exception:
        return {}
    out = {}
    for k, v in doc.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and "value" in v:
            out[k] = v["value"]
    return out


# --------------------------------------------------------------------------- MAT
def mat_checkpoints(prefix, arm_dir):
    """{experiment name: newest transformer_*.pt} under one arm directory."""
    root = os.path.join(MAT_ROOT, "mat", "scripts", "results", "joint_nav",
                        "joint_nav_6a3t_k3", arm_dir)
    out = {}
    for exp in sorted(glob.glob(os.path.join(root, f"{prefix}_*"))):
        files = glob.glob(os.path.join(exp, "wandb", "run-*", "files",
                                       "transformer_*.pt"))
        if not files:
            continue
        best = max(files, key=lambda p: int(
            re.search(r"transformer_(\d+)\.pt", p).group(1)))
        step = int(re.search(r"transformer_(\d+)\.pt", best).group(1))
        out[os.path.basename(exp)] = (best, step)
    return out


def mat_args(fail_rew, k_max, kind):
    from mat.eval_joint_nav import build_args
    a = build_args("mappo" if kind == "mappo" else "plain")
    a.obs_mode, a.comm_mode = "radius", "agent_radius"
    a.sense_radius, a.world_size, a.min_target_sep = 0.6, 2.0, 1.25
    a.shuffle_targets, a.target_order = 1, "distance"
    a.fail_rew = fail_rew
    a.k_req = K_REQ
    a.k_max = k_max
    if kind == "mappo":
        a.encode_state = True          # global-state critic, as trained
    else:
        a.critic_global_state = 1
        a.decoder_self_only = 1 if kind == "dec" else 0
    return a


def eval_mat(path, fail_rew, k_max, kind, greedy, device, n_eps):
    from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
    from mat.envs.joint_nav.joint_nav_env import JointNavMATEnv

    a = mat_args(fail_rew, k_max, kind)
    env = JointNavMATEnv(a)
    pol = TransformerPolicy(a, env.observation_space[0],
                            env.share_observation_space[0],
                            env.action_space[0], N_AGENTS, device=device)
    pol.restore(path)
    pol.eval()
    rnn = np.zeros((N_AGENTS, 1, 64), dtype=np.float32)
    mask = np.ones((N_AGENTS, 1), dtype=np.float32)

    rows = []
    for ep in range(n_eps):
        env.seed(SEED0 + ep)
        torch.manual_seed(ep)
        obs, share, avail = env.reset()
        ret, clear_t, info = 0.0, None, None
        for t in range(env.episode_limit):
            with torch.no_grad():
                out = pol.get_actions(np.asarray(share), np.asarray(obs), rnn, rnn,
                                      mask, np.asarray(avail), deterministic=greedy)
            acts = out[1].detach().cpu().numpy().reshape(-1)
            obs, share, rew, _, info, avail = env.step(acts)
            ret += float(np.asarray(rew).reshape(-1)[0])
            if info[0].get("won") and clear_t is None:
                clear_t = t + 1
        i0 = info[0]
        rows.append((i0["joint_captures"], float(i0["won"]), i0["undershoots"],
                     i0.get("overkills", 0), i0["capture_actions"], ret, clear_t))
    return rows


# --------------------------------------------------------------------- InforMARL
def informarl_checkpoints(prefix):
    root = os.path.join(INF_ROOT, "onpolicy", "results", "GraphMPE",
                        "navigation_joint_graph", "mappo")
    out = {}
    for exp in sorted(glob.glob(os.path.join(root, f"{prefix}_*"))):
        hits = glob.glob(os.path.join(exp, "**", "actor.pt"), recursive=True)
        if hits:
            # newest, in case a seed was resumed
            out[os.path.basename(exp)] = (max(hits, key=os.path.getmtime), -1)
    return out


def informarl_args(fail_rew, k_max, ckpt_path=None):
    """Rebuild the training namespace from InforMARL's own parser.

    When the run's wandb config records its argv, that is replayed verbatim so
    the network matches the checkpoint exactly; the list below is only the
    fallback for a run with no recorded config. Architecture flags differ
    between run generations (the 1M lane used layer_N 1, e2fair/e4fair use 2),
    and a mismatch fails load_state_dict with missing keys.
    """
    from onpolicy.config import get_config, graph_config
    from onpolicy.scripts.train_mpe import parse_args as mpe_parse

    argv = [
        "--env_name", "GraphMPE", "--algorithm_name", "mappo",
        "--scenario_name", "navigation_joint_graph",
        "--num_agents", str(N_AGENTS), "--num_landmarks", str(N_TARGETS),
        "--num_targets", str(N_TARGETS), "--k_req", str(K_REQ),
        "--fail_rew", str(fail_rew),
        "--obs_mode", "radius", "--comm_mode", "agent_radius",
        "--sense_radius", "0.6", "--max_edge_dist", "0.6",
        "--world_size", "2.0", "--episode_length", "50",
        "--min_target_sep", "1.25", "--shuffle_targets", "1",
        "--target_order", "distance",
        "--capture_radius", "0.3", "--group_radius", "0.3",
        "--capture_rew", "10.0", "--assembly_rew", "0.05",
        "--dist_rew_scale", "1.0", "--potential_shaping", "1",
        "--shaping_gamma", "1.0", "--num_obstacles", "0",
        "--collision_rew", "0.0", "--max_speed", "2.0",
        "--min_dist_thresh", "0.1", "--use_dones", "False",
        "--layer_N", "2", "--gnn_layer_N", "2", "--hidden_size", "64",
        "--use_recurrent_policy",          # store_false: passing it sets False
        "--n_rollout_threads", "1", "--n_training_threads", "1",
    ]
    if k_max is not None:
        argv += ["--k_max", str(k_max)]
    rec = recorded_argv(ckpt_path) if ckpt_path else None
    if rec:
        argv = rec
    # train_mpe.parse_args returns (namespace, parser), not just the namespace,
    # and train_mpe.main() then layers graph_config on top for env_name=GraphMPE
    # (that is where --graph_feat_type, --gnn_layer_N and the aggregation flags
    # live). Both steps are needed or the namespace is missing graph fields.
    a, parser = mpe_parse(argv, get_config())
    a, _ = graph_config(argv, parser)
    a.cuda = False
    return a


def eval_informarl(path, fail_rew, k_max, greedy, device, n_eps):
    from multiagent.MPE_env import GraphMPEEnv
    from onpolicy.algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy

    a = informarl_args(fail_rew, k_max, ckpt_path=path)
    env = GraphMPEEnv(a)
    share_space = (env.share_observation_space[0] if a.use_centralized_V
                   else env.observation_space[0])
    pol = GR_MAPPOPolicy(a, env.observation_space[0], share_space,
                         env.node_observation_space[0],
                         env.edge_observation_space[0],
                         env.action_space[0], device=device)
    pol.actor.load_state_dict(torch.load(path, map_location=device))
    pol.actor.eval()

    n_actions = env.action_space[0].n
    rnn = np.zeros((N_AGENTS, a.recurrent_N, a.hidden_size), dtype=np.float32)
    mask = np.ones((N_AGENTS, 1), dtype=np.float32)

    rows = []
    for ep in range(n_eps):
        env.seed(SEED0 + ep)
        torch.manual_seed(ep)
        obs, agent_id, node_obs, adj = env.reset()
        ret, clear_t, info = 0.0, None, None
        rnn_ep = rnn.copy()
        for t in range(a.episode_length):
            with torch.no_grad():
                act, rnn_ep = pol.act(np.asarray(obs), np.asarray(node_obs),
                                      np.asarray(adj), np.asarray(agent_id),
                                      rnn_ep, mask, deterministic=greedy)
            acts = act.detach().cpu().numpy().reshape(-1).astype(int)
            # MultiAgentGraphEnv has discrete_action_input=False, so it reads a
            # one-hot vector per agent (environment.py:261), exactly like the
            # MAT wrapper builds internally. Passing raw ints indexes a scalar.
            onehot = np.eye(n_actions, dtype=np.float32)[acts]
            obs, agent_id, node_obs, adj, rew, _, info = env.step(list(onehot))
            rnn_ep = rnn_ep.detach().cpu().numpy() if torch.is_tensor(rnn_ep) else rnn_ep
            ret += float(np.asarray(rew).reshape(-1)[0])
            if info[0].get("all_captured") and clear_t is None:
                clear_t = t + 1
        i0 = info[0]
        rows.append((i0["joint_captures"], float(i0["all_captured"]),
                     i0["undershoots"], i0.get("overkills", 0),
                     i0["capture_actions"], ret, clear_t))
    return rows


# --------------------------------------------------------------------------- main
def summarise(rows):
    a = np.array([[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows], dtype=float)
    tcl = [r[6] for r in rows if r[6] is not None]
    return (a[:, 0].mean(), a[:, 1].mean(), np.mean(tcl) if tcl else float("nan"),
            a[:, 2].mean(), a[:, 3].mean(), a[:, 4].mean(), a[:, 5].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--prefix", required=True,
                    help="experiment-name prefix, e.g. e4fair")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--only", default="", help="comma list: ctsde,dec,mappo,informarl")
    ap.add_argument("--decode", default="both", choices=["greedy", "sampled", "both"])
    args = ap.parse_args()

    fail_rew, k_max = TASKS[args.task]
    want = [s.strip() for s in args.only.split(",") if s.strip()] or \
           ["ctsde", "dec", "mappo", "informarl"]
    device = torch.device("cpu")
    decodes = [True, False] if args.decode == "both" else [args.decode == "greedy"]

    jobs = []
    for arm in want:
        if arm in MAT_ARMS:
            kind, arm_dir = MAT_ARMS[arm]
            for exp, (path, step) in mat_checkpoints(args.prefix, arm_dir).items():
                jobs.append((arm, exp, path, step, kind))
        elif arm == "informarl":
            for exp, (path, step) in informarl_checkpoints(args.prefix).items():
                jobs.append((arm, exp, path, step, None))
    if not jobs:
        print(f"no checkpoints found for prefix {args.prefix!r}")
        return 1

    cols = ("caps", "win", "t_clear", "unders", "overk", "fires", "return")
    print(f"task={args.task}  fail_rew={fail_rew}  k_max={k_max}  "
          f"episodes={args.episodes}  seeds {SEED0}..{SEED0 + args.episodes - 1}")
    print("%-34s %6s %-8s " % ("run", "ckpt", "decode")
          + " ".join("%8s" % c for c in cols))
    print("-" * 122)

    per_arm = {}
    for arm, exp, path, step, kind in jobs:
        for greedy in decodes:
            try:
                if arm == "informarl":
                    rows = eval_informarl(path, fail_rew, k_max, greedy, device,
                                          args.episodes)
                else:
                    rows = eval_mat(path, fail_rew, k_max, kind, greedy, device,
                                    args.episodes)
            except Exception as e:      # one bad checkpoint must not kill the table
                print("%-34s %6s %-8s  FAILED: %s"
                      % (exp[:34], step if step >= 0 else "last",
                         "greedy" if greedy else "sampled", e))
                if os.environ.get("EVAL_TRACE"):
                    import traceback
                    traceback.print_exc()
                continue
            res = summarise(rows)
            print("%-34s %6s %-8s " % (exp[:34], step if step >= 0 else "last",
                                       "greedy" if greedy else "sampled")
                  + " ".join("%8.3f" % v for v in res), flush=True)
            per_arm.setdefault((arm, greedy), []).append(res)

    print("\n" + "=" * 122)
    print("PER-ARM MEAN OVER SEEDS")
    print("%-34s %6s %-8s " % ("arm", "n", "decode")
          + " ".join("%8s" % c for c in cols))
    print("-" * 122)
    for (arm, greedy), lst in sorted(per_arm.items()):
        m = np.nanmean(np.array(lst, dtype=float), axis=0)
        print("%-34s %6d %-8s " % (arm, len(lst), "greedy" if greedy else "sampled")
              + " ".join("%8.3f" % v for v in m))
    return 0


if __name__ == "__main__":
    sys.exit(main())
