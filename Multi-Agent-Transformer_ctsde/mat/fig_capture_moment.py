"""One-panel PDF of a joint capture: 6 agents, 3 targets, exact-k (E4).

Searches eval episodes for the step where a joint capture actually resolves
(joint_captures increments) and draws that single frame. No title.

The sense radius is NOT drawn (six overlapping 0.6 discs bury the capture
geometry), so it is also absent from the legend.

Sizing. Font size on the printed page is the figure's font size times
(printed width / figure width), so a 7pt legend in a 5in figure included at
half a 5.5in text block prints at under 4pt. This script therefore builds the
figure AT its intended printed width (--width, inches) and sets fonts in true
points, and saves without bbox_inches so the PDF page stays exactly that size.
Include it with \\includegraphics[width=<--width>in]{...} -- or an equivalent
fraction of \linewidth -- and the --fontsize value is what lands on the page.

Usage:
  python fig_capture_moment.py <run_dir> --out fig.pdf [--episodes 40]
                               [--width 3.3] [--fontsize 8.5]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from viz_joint_nav import (find_files_dir, latest_ckpt, load_args, rollout,
                           AGENT_CMAP)
from mat.envs.joint_nav.joint_nav_env import JointNavMATEnv
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy


def find_capture_frame(frames):
    """Index of the frame where a capture resolved, or None."""
    for i in range(1, len(frames)):
        if frames[i]["caps"] > frames[i - 1]["caps"]:
            return i
    return None


# Match the learning-curve figures: Computer Modern serif, embedded TrueType.
PAPER_RC = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
}

AGENT_FC = "#3b6ea5"   # one neutral colour for every agent, so the only
                       # green is targets and the only red is CAPTURE


def draw(ax, f, a, capture_target, active, fs=8.5):
    n = f["agents"].shape[0]
    m = f["targets"].shape[0]
    cap_r = float(a.capture_radius)
    lim = 1.25 * max(1.0, float(getattr(a, "world_size", 2.0)) / 2.0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.75")

    # comm edges (1-hop agent-radius graph the policy actually conditions on)
    adj = f["adj"]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0.5:
                ax.plot(*zip(f["agents"][i], f["agents"][j]), "-",
                        color="0.55", lw=1.0, zorder=1)

    for t in range(m):
        p = f["targets"][t]
        live = active[t]
        col = "#2e8b57" if live else "0.6"
        ax.add_patch(Circle(p, cap_r, fill=False, ls="--", lw=1.6, ec=col,
                            alpha=0.95, zorder=2))
        ax.add_patch(Circle(p, 0.045, fc=col, ec="none", zorder=3))
        ax.text(p[0], p[1] - cap_r - 0.07, "T%d" % t, fontsize=fs,
                color="#20603c" if live else "0.5", ha="center", va="top",
                zorder=4)

    for i in range(n):
        p = f["agents"][i]
        if f["capturing"][i]:
            ax.add_patch(Circle(p, 0.085, fill=False, ec="#d62728", lw=2.2,
                                zorder=5))
        ax.add_patch(Circle(p, 0.05, fc=AGENT_FC, ec="white", lw=1.0,
                            zorder=6))
        ax.text(p[0], p[1], str(i), fontsize=fs * 0.8, color="white", ha="center",
                va="center", zorder=7)

    return [
        Line2D([], [], marker="o", ls="", color=AGENT_FC, markersize=8,
               label="agent"),
        Line2D([], [], marker="o", ls="", color="#2e8b57", markersize=8,
               label="target"),
        Line2D([], [], ls="--", color="#2e8b57", lw=1.6,
               label=r"capture radius, $r_{\mathrm{cap}}=%.1f$" % cap_r),
        Line2D([], [], marker="o", ls="", mfc="none", mec="#d62728", mew=2,
               markersize=8, label="CAPTURE action"),
        Line2D([], [], color="0.55", lw=1.6,
               label=r"comm link, $r_{\mathrm{com}}=%.1f$"
               % float(getattr(a, "sense_radius", 0.6))),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="capture_moment.pdf")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=50000)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--width", type=float, default=3.3,
                    help="intended PRINTED width in inches; the PDF page is "
                         "built at exactly this size so fonts are true-to-point")
    ap.add_argument("--fontsize", type=float, default=8.5,
                    help="printed point size of the legend and labels")
    cli = ap.parse_args()

    files_dir = find_files_dir(cli.run_dir)
    ckpt = latest_ckpt(files_dir, cli.ckpt)
    a, cfg = load_args(files_dir)
    device = torch.device("cpu")
    a.cuda = False

    env = JointNavMATEnv(a)
    pol = TransformerPolicy(a, env.observation_space[0],
                            env.share_observation_space[0],
                            env.action_space[0], env.n_agents, device=device)
    pol.restore(ckpt)
    pol.eval()
    print("[fig] run %s  ckpt %s" % (cfg.get("experiment_name"),
                                     os.path.basename(ckpt)))
    print("[fig] n_agents=%d num_targets=%d k_req=%s k_max=%s fail_rew=%s"
          % (a.n_agents, a.num_targets, a.k_req,
             getattr(a, "k_max", None), a.fail_rew))

    for e in range(cli.episodes):
        seed = cli.seed + e
        fr = rollout(env, pol, a, seed, True)
        idx = find_capture_frame(fr)
        if idx is None:
            continue
        f = fr[idx]
        n_fire = int(f["capturing"].sum())
        # which target resolved: the one that went inactive this step
        went = np.nonzero(fr[idx - 1]["active"] & ~f["active"])[0]
        tgt = int(went[0]) if len(went) else -1
        print("[fig] seed=%d step=%d firing=%d target=T%d"
              % (seed, f["step"], n_fire, tgt))
        if n_fire != int(a.k_req):
            continue          # want the clean exact-k picture
        # Printed point size = figure point size x (printed width / figure
        # width). Rather than guess the column width, the figure is BUILT at
        # its intended printed width, so --fontsize is literally what lands on
        # the page and no scaling happens. Include with
        # \includegraphics[width=<--width>in]{...}.
        W = float(cli.width)
        with matplotlib.rc_context(PAPER_RC):
            fig, ax = plt.subplots(figsize=(W, W * 1.24))
            handles = draw(ax, f, a, tgt, fr[idx - 1]["active"],
                           fs=cli.fontsize)
            # loc="outside lower center" needs matplotlib >= 3.7 (this env has
            # 3.5), so the strip is reserved by hand; same result on any version.
            # 5 entries over 2 columns = 3 rows; 3 columns overflows the
            # page width at this font size and clips the last label
            fig.subplots_adjust(left=0.015, right=0.985, top=0.985,
                                bottom=0.205)
            # Five entries do not divide into two columns, and matplotlib
            # fills column-major, so a single 2-column legend strands the odd
            # one bottom-left. Split it: a 2x2 block, then the capture-radius
            # entry centred on its own row underneath.
            grid = [handles[0], handles[1], handles[3], handles[4]]
            fig.legend(handles=grid, loc="lower center",
                       bbox_to_anchor=(0.5, 0.052), ncol=2, frameon=False,
                       fontsize=cli.fontsize, handlelength=1.5,
                       columnspacing=1.0, handletextpad=0.5,
                       borderaxespad=0.0)
            fig.legend(handles=[handles[2]], loc="lower center",
                       bbox_to_anchor=(0.5, 0.0), ncol=1, frameon=False,
                       fontsize=cli.fontsize, handlelength=1.5,
                       handletextpad=0.5, borderaxespad=0.0)
            # no bbox_inches: the page must stay exactly (W, W*1.18) inches or
            # the printed point size stops matching --fontsize
            fig.savefig(cli.out, format="pdf")
            fig.savefig(os.path.splitext(cli.out)[0] + ".png", dpi=300)
        print("[fig] page %.2f x %.2f in, text %.1f pt; include at width=%.2fin"
              % (W, W * 1.24, cli.fontsize, W))
        print("[fig] wrote %s (seed %d, step %d, %d agents firing on T%d)"
              % (cli.out, seed, f["step"], n_fire, tgt))
        return 0
    print("[fig] no clean exact-k capture frame found in %d episodes"
          % cli.episodes)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
