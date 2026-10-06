"""Deployed evaluation of the E2 penalty variants (no exploration override).

60 fixed eval seeds, sampled and greedy decoding, E2 settings (radius R=0.6,
2x2, sep 1.25, shuffle + distance order). The env's miss penalty is set per
model to the one it was trained with. Models (last available checkpoint):
  -5 optimistic (OptiMAPPO, leaky 0)   o5_E2_ctsde_const_opt
  -5 SIL (unfiltered, stopped)         s5_E2_ctsde_const_sil
  -5 const eps (stopped)               c5_E2_ctsde_const
  -1 const eps (reference, solved)     rad_ws2_..._opw_shuf_dist_s1 (8hle0zgb)

Columns: caps, win, t_clear (won eps), unders, fires, return (with the
model's own penalty), in_rng, READY/ep, f_rdy, f_nr, f_out (fire rates when
ready / in range not ready / out of range), joint (P(>=3 fire | READY)).

  python -m mat.probe_penalty_eval
"""
import glob
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.envs.joint_nav.joint_nav_env import JointNavMATEnv
from mat.eval_joint_nav import build_args

RES = os.path.join(os.path.dirname(__file__), "scripts", "results", "joint_nav",
                   "joint_nav_6a3t_k3", "mat_comm_soft")
MODELS = [
    ("-5 optimistic", "o5_E2_ctsde_const_opt", -5.0),
    ("-5 SIL (unfilt.)", "s5_E2_ctsde_const_sil", -5.0),
    ("-5 const eps", "c5_E2_ctsde_const", -5.0),
    ("-1 const eps (ref)", "rad_ws2_R0.6_sep1.25_gcrit_radgrp_opw_shuf_dist_s1", -1.0),
]
RES_ROOT = os.path.dirname(RES)   # .../joint_nav_6a3t_k3 (one folder per arm)
if os.environ.get("EVAL_SET") == "bench":
    # E2-p (-5) benchmark: identical recipe, 2M steps, seed 1. 4th field = arm kind.
    MODELS = [
        ("CT-SDE +opt", "o5_E2_opt00_2m", -5.0, "ctsde"),
        ("DE control +opt", "b5_E2_dec_opt_2m", -5.0, "dec"),
        ("MAPPO +opt", "b5_E2_mappo_opt_2m", -5.0, "mappo"),
        ("CT-SDE no-opt", "b5_E2_ctsde_noopt_2m", -5.0, "ctsde"),
    ]
if os.environ.get("EVAL_SET") == "e4":
    # E4 (exact-k, k_max 3): subset commit q 0.5 + arm-consistent weights + opt.
    MODELS = [
        ("CT-SDE E4 4M", "e4_ctsde_sub05_armw_opt_4m", -5.0, "ctsde"),
        ("DE control E4 4M", "e4_dec_sub05_armw_opt_4m", -5.0, "dec"),
        ("CT-SDE E4 2M", "e4_ctsde_sub05_armw_opt_2m", -5.0, "ctsde"),
    ]
if os.environ.get("EVAL_SET") == "e4m1":
    # exact-k at miss penalty -1, no optimism, 1.5M.
    MODELS = [
        ("CT-SDE E4 -1 1.5M", "e4m1_ctsde_sub05_armw_1p5m", -1.0, "ctsde"),
        ("DE control E4 -1 1.5M", "e4m1_dec_sub05_armw_1p5m", -1.0, "dec"),
    ]
if os.environ.get("EVAL_SET") == "fair":
    # E2-p, CT-SDE + optimism: old (marginal) vs arm-consistent behaviour weight.
    MODELS = [
        ("armw (conditional)", "f5_E2_ctsde_armw_opt_2m", -5.0, "ctsde"),
        ("old w (marginal)", "o5_E2_opt00_2m", -5.0, "ctsde"),
    ]
if os.environ.get("EVAL_SET") == "opt2m":
    MODELS = [
        ("-5 opt leaky0 2M", "o5_E2_opt00_2m", -5.0),
        ("-5 opt leaky0.3 2M", "o5_E2_opt03_2m", -5.0),
        ("-5 opt leaky0 1M", "o5_E2_ctsde_const_opt", -5.0),
        ("-1 const eps (ref)", "rad_ws2_R0.6_sep1.25_gcrit_radgrp_opw_shuf_dist_s1", -1.0),
    ]
N, M, K, FIRE = 6, 3, 3, 5
N_EPS = 60


def last_ckpt(folder):
    files = glob.glob(os.path.join(RES_ROOT, "*", folder, "wandb", "run-*", "files",
                                   "transformer_*.pt"))
    best = max(files, key=lambda p: int(re.search(r"transformer_(\d+)\.pt", p).group(1)))
    return best, int(re.search(r"transformer_(\d+)\.pt", best).group(1))


def env_args(fail_rew, kind="ctsde"):
    a = build_args("mappo" if kind == "mappo" else "plain")
    a.obs_mode, a.comm_mode = "radius", "agent_radius"
    a.sense_radius, a.world_size, a.min_target_sep = 0.6, 2.0, 1.25
    a.shuffle_targets, a.target_order = 1, "distance"
    a.fail_rew = fail_rew
    if os.environ.get("EVAL_SET") in ("e4", "e4m1"):
        a.k_max = 3                                # exact-k, as trained
    if kind == "mappo":
        a.encode_state = True                      # global-state critic, as trained
    else:
        a.critic_global_state = 1
        a.decoder_self_only = 1 if kind == "dec" else 0
    return a


def evaluate(path, fail_rew, greedy, device, kind="ctsde"):
    a = env_args(fail_rew, kind)
    env = JointNavMATEnv(a)
    sc, w = env._scenario, env._world
    pol = TransformerPolicy(a, env.observation_space[0], env.share_observation_space[0],
                            env.action_space[0], N, device=device)
    pol.restore(path)
    pol.eval()
    rnn = np.zeros((N, 1, 64), dtype=np.float32)
    mask = np.ones((N, 1), dtype=np.float32)
    caps, won, tcl, und, fires, rets, ovr = [], [], [], [], [], [], []
    inr_n = tot = ready = 0
    fr, fn, fo, jt = [0, 0], [0, 0], [0, 0], [0, 0]
    for ep in range(N_EPS):
        env.seed(50000 + ep)
        torch.manual_seed(ep)
        obs, share, avail = env.reset()
        ret, clear_t = 0.0, None
        for t in range(env.episode_limit):
            live = [j for j in range(M) if w.target_active[j]]
            inr = {j: np.nonzero(sc._agents_in_range(w, j))[0] for j in live}
            in_range = np.zeros(N, dtype=bool)
            rdy = np.zeros(N, dtype=bool)
            with torch.no_grad():
                out = pol.get_actions(np.asarray(share), np.asarray(obs), rnn, rnn,
                                      mask, np.asarray(avail), deterministic=greedy)
            acts = out[1].detach().cpu().numpy().reshape(-1)
            for j, s in inr.items():
                in_range[s] = True
                if len(s) >= K:
                    rdy[s] = True
                    ready += 1
                    jt[0] += int((acts[s] == FIRE).sum() >= K)
                    jt[1] += 1
            if live:
                for i in range(N):
                    tot += 1
                    inr_n += int(in_range[i])
                    box = fr if rdy[i] else (fn if in_range[i] else fo)
                    box[0] += int(acts[i] == FIRE)
                    box[1] += 1
            obs, share, rew, _, info, avail = env.step(acts)
            ret += float(np.asarray(rew).reshape(-1)[0])
            if info[0]["won"] and clear_t is None:
                clear_t = t + 1
        i0 = info[0]
        caps.append(i0["joint_captures"])
        won.append(float(i0["won"]))
        und.append(i0["undershoots"])
        ovr.append(i0.get("overkills", 0))
        fires.append(i0["capture_actions"])
        rets.append(ret)
        if clear_t is not None:
            tcl.append(clear_t)
    r = lambda x: x[0] / max(x[1], 1)
    return (np.mean(caps), np.mean(won), np.mean(tcl) if tcl else float("nan"),
            np.mean(und), np.mean(ovr), np.mean(fires), np.mean(rets), inr_n / max(tot, 1),
            ready / N_EPS, r(fr), r(fn), r(fo), r(jt))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cols = ("caps", "win", "t_clear", "unders", "overk", "fires", "return", "in_rng", "READY/ep",
            "f_rdy", "f_nr", "f_out", "joint")
    print("%-20s %5s %-7s " % ("model", "ckpt", "decode") + " ".join("%8s" % c for c in cols))
    print("-" * 140)
    for entry in MODELS:
        name, folder, fr = entry[:3]
        kind = entry[3] if len(entry) > 3 else "ctsde"
        path, ck = last_ckpt(folder)
        for greedy in (False, True):
            res = evaluate(path, fr, greedy, device, kind)
            print("%-20s %5d %-7s " % (name, ck, "greedy" if greedy else "sampled")
                  + " ".join("%8.3f" % v for v in res), flush=True)


if __name__ == "__main__":
    main()
