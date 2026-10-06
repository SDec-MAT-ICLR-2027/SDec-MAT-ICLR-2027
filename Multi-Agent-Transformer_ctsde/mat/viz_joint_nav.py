"""Render a joint_nav checkpoint as an animation (Windows-native matplotlib).

Rolls episodes out of a saved run and draws the world: agents, targets with
their capture rings, the comm graph, and the CAPTURE actions fired each step.

  python viz_joint_nav.py <run_dir> [--episodes 3] [--seed 50000]
                          [--ckpt transformer_624.pt] [--sampled] [--show]
                          [--out out.gif] [--mp4]

<run_dir> is the experiment directory holding wandb/run-*/files (config.yaml
plus the transformer_*.pt checkpoints), e.g.
  scripts/results/joint_nav/joint_nav_6a3t_k3/mat_comm_soft/s1_E1_ctsde_decay
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.config import get_config
from mat.envs.joint_nav.joint_nav_env import JointNavMATEnv

CAPTURE_ACTION_EXTRA = 1  # the scenario's CAPTURE id in agent.action.extra


# --------------------------------------------------------------- run loading
def find_files_dir(run_dir):
    """The wandb files/ dir inside an experiment dir (or run_dir itself)."""
    if os.path.isfile(os.path.join(run_dir, "config.yaml")):
        return run_dir
    all_dirs = sorted(glob.glob(os.path.join(run_dir, "wandb", "run-*", "files")))
    cands = [c for c in all_dirs if os.path.isfile(os.path.join(c, "config.yaml"))]
    if cands:
        return cands[-1]
    # A run stopped before wandb wrote config.yaml: use the files/ dir holding
    # the most checkpoints; main() then needs --set overrides for the env.
    with_ckpts = [c for c in all_dirs if glob.glob(os.path.join(c, "transformer_*.pt"))]
    if not with_ckpts:
        raise SystemExit("no wandb files/ dir with config.yaml or checkpoints under %r"
                         % run_dir)
    return max(with_ckpts, key=lambda c: len(glob.glob(os.path.join(c, "transformer_*.pt"))))


def latest_ckpt(files_dir, name=None):
    if name:
        p = name if os.path.isabs(name) else os.path.join(files_dir, name)
        if not os.path.isfile(p):
            raise SystemExit("checkpoint not found: %r" % p)
        return p
    pts = glob.glob(os.path.join(files_dir, "transformer_*.pt"))
    if not pts:
        raise SystemExit("no transformer_*.pt in %r" % files_dir)

    def it(p):
        m = re.search(r"transformer_(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1

    return max(pts, key=it)


def load_args(files_dir):
    """Parser defaults, overridden by the run's resolved wandb config."""
    base = get_config().parse_args([])
    raw = yaml.safe_load(open(os.path.join(files_dir, "config.yaml"),
                              encoding="utf-8"))
    cfg = {}
    for k, v in raw.items():
        if k.startswith("_") or k == "wandb_version":
            continue
        cfg[k] = v.get("value") if isinstance(v, dict) and "value" in v else v

    a = SimpleNamespace(**vars(base))
    for k, v in cfg.items():
        # wandb yaml can stringify scientific-notation floats ("1e-05")
        d = getattr(base, k, None)
        if isinstance(v, str) and isinstance(d, (int, float)) \
                and not isinstance(d, bool):
            try:
                v = type(d)(float(v))
            except ValueError:
                pass
        setattr(a, k, v)
    # single-threaded rollout, no training-side plumbing
    a.n_rollout_threads = 1
    a.n_eval_rollout_threads = 1
    a.n_training_threads = 1
    a.cuda = bool(getattr(a, "cuda", True)) and torch.cuda.is_available()
    return a, cfg


# -------------------------------------------------------------------- rollout
def rollout(env, pol, args, seed, greedy):
    """One episode; returns a per-step list of world snapshots."""
    env.seed(seed)
    torch.manual_seed(seed)
    n = env.n_agents
    hidden = int(getattr(args, "hidden_size", 64))
    rnn = np.zeros((n, 1, hidden), dtype=np.float32)
    mask = np.ones((n, 1), dtype=np.float32)

    obs, share, avail = env.reset()
    world = env._world

    def snapshot(step, capturing, rew, ret, info):
        return dict(
            step=step,
            agents=np.array([a.state.p_pos.copy() for a in world.agents]),
            targets=np.array([l.state.p_pos.copy() for l in world.landmarks]),
            active=np.array(world.target_active, dtype=bool).copy(),
            adj=np.array(env.get_comm_mask(), dtype=float).copy(),
            capturing=capturing,
            rew=rew,
            ret=ret,
            caps=int(info["joint_captures"]) if info else 0,
            unders=int(info["undershoots"]) if info else 0,
            won=bool(info["won"]) if info else False,
        )

    frames = [snapshot(0, np.zeros(n, bool), 0.0, 0.0, None)]
    ret = 0.0
    for t in range(env.episode_limit):
        with torch.no_grad():
            out = pol.get_actions(np.asarray(share), np.asarray(obs), rnn, rnn,
                                  mask, np.asarray(avail), deterministic=greedy)
        acts = out[1].detach().cpu().numpy().reshape(-1).astype(int)
        obs, share, rew, dones, info, avail = env.step(acts)
        capturing = np.array(
            [getattr(a.action, "extra", 0) == CAPTURE_ACTION_EXTRA
             for a in world.agents], dtype=bool)
        r = float(rew[0][0])
        ret += r
        frames.append(snapshot(t + 1, capturing, r, ret, info[0]))
        if bool(np.all(dones)):
            break
    return frames


# -------------------------------------------------------------------- drawing
AGENT_CMAP = plt.get_cmap("tab10")


def draw_episode(ax, frames, args, title_prefix):
    """Create the artists once; return an update(i) closure."""
    n = frames[0]["agents"].shape[0]
    m = frames[0]["targets"].shape[0]
    cap_r = float(args.capture_radius)
    # which radius actually forms the groups depends on comm_mode:
    # agent_radius pairs agents within sense_radius, target_clique pairs
    # agents co-located within group_radius of the SAME live target.
    comm_mode = str(getattr(args, "comm_mode", "target_clique"))
    if comm_mode == "agent_radius":
        grp_r = float(getattr(args, "sense_radius", 1.0))
    else:
        grp_r = float(getattr(args, "group_radius", 2 * cap_r))
    # MAPPO has no comm channel: the env still computes an adjacency, but the
    # policy never reads it, so drawing those edges would imply structure the
    # agents do not have.
    has_comm = str(getattr(args, "algorithm_name", "")) in (
        "mat_comm_soft", "mat_comm_dec", "mat_comm")

    lim = 1.25 * max(1.0, float(getattr(args, "world_size", 2.0)) / 2.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.75")

    # comm edges, drawn under everything
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edges = [ax.plot([], [], "-", color="0.55", lw=1.0, alpha=0.0, zorder=1)[0]
             for _ in pairs]

    rings = [Circle((0, 0), cap_r, fill=False, ls="--", lw=1.4,
                    ec="#2e8b57", alpha=0.9, zorder=2) for _ in range(m)]
    cores = [Circle((0, 0), 0.045, fc="#2e8b57", ec="none", zorder=3)
             for _ in range(m)]
    for c in rings + cores:
        ax.add_patch(c)
    tlabels = [ax.text(0, 0, "T%d" % t, fontsize=8, color="#20603c",
                       ha="center", va="center", zorder=4) for t in range(m)]

    # each agent's sensing disc: in obs_mode=radius this is literally what it
    # can see, and under comm_mode=agent_radius also who it groups with
    sense_r = float(getattr(args, "sense_radius", 0.0) or 0.0)
    show_sense = str(getattr(args, "obs_mode", "shared")) == "radius" \
        or comm_mode == "agent_radius"
    discs = [Circle((0, 0), sense_r, fill=False, ec=AGENT_CMAP(i % 10), lw=0.7,
                    ls=(0, (2, 3)), alpha=0.22 if show_sense else 0.0, zorder=1)
             for i in range(n)]
    for c in discs:
        ax.add_patch(c)

    bodies = [Circle((0, 0), 0.05, fc=AGENT_CMAP(i % 10), ec="white", lw=1.0,
                     zorder=6) for i in range(n)]
    halos = [Circle((0, 0), 0.085, fill=False, ec="#d62728", lw=2.2,
                    alpha=0.0, zorder=5) for _ in range(n)]
    for c in halos + bodies:
        ax.add_patch(c)
    alabels = [ax.text(0, 0, str(i), fontsize=7, color="white", ha="center",
                       va="center", zorder=7) for i in range(n)]
    trails = [ax.plot([], [], "-", color=AGENT_CMAP(i % 10), lw=1.0,
                      alpha=0.35, zorder=2)[0] for i in range(n)]

    title = ax.set_title("", fontsize=9, family="monospace")
    banner = ax.text(0, lim * 0.86, "", fontsize=13, color="#d62728",
                     ha="center", va="center", weight="bold", alpha=0.0,
                     zorder=8)

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#2e8b57", label="target"),
        Line2D([], [], ls="--", color="#2e8b57",
               label="capture radius %.2f" % cap_r),
        Line2D([], [], marker="o", ls="", mfc="none", mec="#d62728", mew=2,
               label="CAPTURE action"),
        Line2D([], [], ls=(0, (2, 3)), color="0.4",
               label="sense radius %.2f (%s)" % (sense_r, args.obs_mode)),
        Line2D([], [], color="0.55" if has_comm else "none",
               label=("comm group: %s r=%.2f" % (comm_mode, grp_r)) if has_comm
               else "no comm channel (%s)" % args.algorithm_name),
    ], loc="upper left", fontsize=7, framealpha=0.85)

    hist = [np.empty((0, 2)) for _ in range(n)]

    def update(i):
        f = frames[i]
        if i == 0:
            for k in range(n):
                hist[k] = np.empty((0, 2))
        for k in range(n):
            hist[k] = np.vstack([hist[k][-14:], f["agents"][k][None, :]])
            trails[k].set_data(hist[k][:, 0], hist[k][:, 1])
            bodies[k].center = tuple(f["agents"][k])
            discs[k].center = tuple(f["agents"][k])
            halos[k].center = tuple(f["agents"][k])
            halos[k].set_alpha(1.0 if f["capturing"][k] else 0.0)
            alabels[k].set_position(tuple(f["agents"][k]))

        for t in range(m):
            p = tuple(f["targets"][t])
            rings[t].center = p
            cores[t].center = p
            tlabels[t].set_position(p)
            live = bool(f["active"][t])
            col = "#2e8b57" if live else "0.72"
            rings[t].set_edgecolor(col)
            rings[t].set_linestyle("--" if live else ":")
            cores[t].set_facecolor(col)
            tlabels[t].set_color("#20603c" if live else "0.45")

        adj = f["adj"]
        for e, (i0, j0) in zip(edges, pairs):
            if has_comm and (adj[i0, j0] > 0 or adj[j0, i0] > 0):
                e.set_data([f["agents"][i0][0], f["agents"][j0][0]],
                           [f["agents"][i0][1], f["agents"][j0][1]])
                e.set_alpha(0.55)
            else:
                e.set_alpha(0.0)

        got = f["caps"] - (frames[i - 1]["caps"] if i > 0 else 0)
        banner.set_alpha(1.0 if got > 0 else 0.0)
        banner.set_text("CAPTURE x%d" % got if got > 0 else "")

        title.set_text(
            "%s\nt=%2d/%d  r=%+6.2f  R=%+7.2f  captures %d/%d  "
            "undershoots %2d  capture_acts %d"
            % (title_prefix, f["step"], len(frames) - 1, f["rew"], f["ret"],
               f["caps"], m, f["unders"], int(f["capturing"].sum())))
        return []

    return update


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=50000,
                    help="first eval seed (eval envs use 50000+i)")
    ap.add_argument("--ckpt", default=None, help="transformer_*.pt (default: latest)")
    ap.add_argument("--sampled", action="store_true",
                    help="sample actions instead of argmax")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--hold", type=int, default=6,
                    help="frames held on the final step of each episode")
    ap.add_argument("--out", default=None, help="output path (.gif)")
    ap.add_argument("--mp4", action="store_true", help="also write .mp4 (needs ffmpeg)")
    ap.add_argument("--show", action="store_true", help="open a live window")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override/define a run arg; required for runs without a "
                         "saved config.yaml (e.g. --set n_agents=9 --set obs_mode=radius)")
    ap.add_argument("--arm", default="plain", choices=["plain", "dec", "mappo"],
                    help="arm, only used when the run has no config.yaml")
    cli = ap.parse_args()

    files_dir = find_files_dir(cli.run_dir)
    ckpt = latest_ckpt(files_dir, cli.ckpt)
    if os.path.isfile(os.path.join(files_dir, "config.yaml")):
        a, cfg = load_args(files_dir)
    else:
        if not cli.set:
            raise SystemExit("run has no config.yaml; pass its env/arm args with --set")
        from mat.eval_joint_nav import build_args
        a = build_args("mappo" if cli.arm == "mappo" else "plain")
        if cli.arm == "dec":
            a.decoder_self_only = 1
        a.cuda = torch.cuda.is_available()
        cfg = {"experiment_name": os.path.basename(cli.run_dir.rstrip("\\/")),
               "arm_name": {"plain": "mat_comm_soft", "dec": "mat_comm_dec",
                            "mappo": "mappo"}[cli.arm],
               "algorithm_name": a.algorithm_name}
        print("[viz] no config.yaml -- using eval defaults + --set overrides")
    for kv in cli.set:
        k, v = kv.split("=", 1)
        old = getattr(a, k, None)
        if isinstance(old, bool) or v in ("True", "False"):
            v = v == "True"
        elif isinstance(old, int) and not isinstance(old, bool):
            v = int(v)
        elif isinstance(old, float):
            v = float(v)
        else:
            try:
                v = int(v) if v.lstrip("-").isdigit() else float(v)
            except ValueError:
                pass
        setattr(a, k, v)
    device = torch.device("cuda:0" if a.cuda else "cpu")

    env = JointNavMATEnv(a)
    pol = TransformerPolicy(a, env.observation_space[0],
                            env.share_observation_space[0],
                            env.action_space[0], env.n_agents, device=device)
    pol.restore(ckpt)
    pol.eval()

    mode = "sampled" if cli.sampled else "greedy"
    exp = str(cfg.get("experiment_name", os.path.basename(cli.run_dir.rstrip("\\/"))))
    arm = str(cfg.get("arm_name", cfg.get("algorithm_name", "?")))
    print("[viz] run     %s" % exp)
    print("[viz] arm     %s (%s)" % (arm, cfg.get("algorithm_name")))
    print("[viz] ckpt    %s" % os.path.basename(ckpt))
    print("[viz] device  %s   policy %s" % (device, mode))

    episodes = []
    for e in range(cli.episodes):
        seed = cli.seed + e
        fr = rollout(env, pol, a, seed, not cli.sampled)
        last = fr[-1]
        print("[viz] ep seed=%d  steps=%2d  captures=%d/%d  undershoots=%d  "
              "return=%+.2f  won=%s"
              % (seed, last["step"], last["caps"], a.num_targets,
                 last["unders"], last["ret"], last["won"]))
        episodes.append((seed, fr))

    # one flat timeline over all episodes, holding on each final frame
    timeline = []
    for ei, (_, fr) in enumerate(episodes):
        timeline += [(ei, i) for i in range(len(fr))]
        timeline += [(ei, len(fr) - 1)] * cli.hold

    if not cli.show:
        matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(6.4, 6.6), dpi=110)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.04)

    state = {"ep": None, "update": None}

    def frame(idx):
        ei, i = timeline[idx]
        seed, fr = episodes[ei]
        if state["ep"] != ei:
            ax.clear()
            state["ep"] = ei
            state["update"] = draw_episode(
                ax, fr, a, "%s | %s | %s | ep %d (seed %d)"
                % (exp, arm, mode, ei + 1, seed))
        return state["update"](i)

    anim = animation.FuncAnimation(fig, frame, frames=len(timeline),
                                   interval=1000 // max(1, cli.fps),
                                   blit=False, repeat=True)

    out = cli.out or os.path.join(
        HERE, "viz_%s_%s_%s.gif" % (exp, os.path.basename(ckpt)[:-3], mode))
    anim.save(out, writer=animation.PillowWriter(fps=cli.fps))
    print("[viz] wrote %s" % out)
    if cli.mp4:
        mp4 = os.path.splitext(out)[0] + ".mp4"
        try:
            anim.save(mp4, writer=animation.FFMpegWriter(fps=cli.fps, bitrate=2400))
            print("[viz] wrote %s" % mp4)
        except Exception as exc:  # ffmpeg missing
            print("[viz] mp4 skipped: %s" % exc)
    if cli.show:
        plt.show()


if __name__ == "__main__":
    main()
