"""Joint-Action Navigation camera-ready figures, in the house style.

Styling is taken from results/viz/plot_results.ipynb so these figures sit in
the same family as the MAC ones: same palette, same dash patterns, same
Computer Modern serif, same legend placement, and the same band semantics
(mean with a MIN/MAX-over-seeds band, not a standard deviation). The tables
report mean +/- 1 std of the final value, which is the usual table convention
-- the two intervals are deliberately different and the caption should say so.

Method names follow the notebook's vocabulary:
    SDec-MAT  = mat_comm_soft  (CT-SDE, the AR decoder over 1-hop neighbours)
    MAT-Dec   = mat_comm_dec   (same encoder + obs sharing, no action conditioning)
    MAPPO     = mappo --dec_actor
    InforMARL = graph MAPPO

Metrics: success rate, joint captures, undershoots, overkills, and false
captures (= undershoots + overkills, the unsuccessful in-range capture attempts
per episode; the two are disjoint because _resolve_captures tests overkill,
then the capture threshold, then undershoot).

On the at-least-k tasks (E2, E2-p) there is no k_max, so an overkill cannot
occur by construction: the overkill figure would be a flat zero and the false
captures figure would duplicate undershoots. Both are detected and skipped
rather than drawn as empty panels.

Usage
    python scripts/plot_jointnav_paper.py --task e2
    python scripts/plot_jointnav_paper.py --task e4 --variant sub
"""
import argparse
import os
import shutil
import sys
import warnings
from collections import defaultdict

import numpy as np

# The W&B project holding the runs. Set both in the environment.
ENTITY = os.environ.get("WANDB_ENTITY")
PROJECT = os.environ.get("WANDB_PROJECT")
if not ENTITY or not PROJECT:
    raise SystemExit("set WANDB_ENTITY and WANDB_PROJECT to the project holding the runs")

# task -> (wandb experiment-name prefix, has a k_max so overkills are possible)
TASKS = {
    "e2":   ("e2fair",   False),   # at-least-3, miss -1, 2M
    "e2p":  ("e2pfair",  False),   # at-least-3, miss -5, 2M
    "e4":   ("e4fair",   True),    # exact-3,   miss -5, 4M
    "e4m1": ("e4m1v2",   True),    # exact-3,   miss -1, 4M, behav_w_floor 0.05
    "e4m1f0": ("e4m1f0", True),    # as e4m1 but behav_w_floor 0 (ablation)
}

SMOOTH = 5       # rolling-mean window on the resampled grid (notebook default)
N_GRID = 300     # common x-grid every seed is resampled onto

# ---- palette and line styles, copied from plot_results.ipynb ---------------
PALETTE = {
    "MAPPO":     "#3A86FF",   # azure
    "SDec-MAT":  "#FF006E",   # rose
    "MAT-Dec":   "#7209B7",   # deep violet
    "InforMARL": "#00E5FF",   # neon cyan - always drawn with the dark outline
}
INFORMARL_OUTLINE = "#0B3C49"

# Dash patterns spelled out rather than "--"/"-.": matplotlib scales dash
# patterns by line width, so at lw=2 the built-in "-." runs a cycle longer than
# the legend handle and its swatch would render as a plain line.
PAPER_LS = {
    "MAPPO":     (0, (2.5, 1.5)),
    "InforMARL": (0, (3, 2)),
}
PAPER_BASELINE_LW = 2.0
PAPER_MAT_LW = 3.2
PAPER_MATDEC_LW = 2.4
PAPER_INFORMARL_LW = 1.8      # +1.2pt outline = 3.0pt overall
PAPER_BAND_ALPHA = 0.15
PAPER_FIGSIZE = (7.0, 5.0)

# arm key -> (label, colour, linestyle, linewidth, outline)
ARMS = [
    ("ctsde",     "SDec-MAT",  PALETTE["SDec-MAT"],  "-", PAPER_MAT_LW,        None),
    ("dec",       "MAT-Dec",   PALETTE["MAT-Dec"],   "-", PAPER_MATDEC_LW,     None),
    ("mappo",     "MAPPO",     PALETTE["MAPPO"],     PAPER_LS["MAPPO"],
                                                     PAPER_BASELINE_LW,        None),
    ("informarl", "InforMARL", PALETTE["InforMARL"], PAPER_LS["InforMARL"],
                                                     PAPER_INFORMARL_LW, INFORMARL_OUTLINE),
]

METRICS = [
    ("win_rate",            "Success Rate",                   "success_rate", (-0.03, 1.03)),
    ("joint_captures_mean", "Joint Captures per Episode",     "joint_captures", None),
    ("undershoots_mean",    "Undershoots per Episode",        "undershoots",   None),
    ("overkills_mean",      "Overkills per Episode",          "overkills",     None),
    ("false_captures",      "False Captures per Episode",     "false_captures", None),
]
RAW_KEYS = ["win_rate", "joint_captures_mean", "undershoots_mean", "overkills_mean"]

SUB_DROP = {"ctsde": "_s3_", "dec": "_s3_"}
SUB_ADD = {
    "ctsde": "mat_comm_soft_e4_ctsde_sub05_armw_opt_4m_k3_cr10_fr-5_radius_ar0.05_seed1",
    "dec":   "mat_comm_dec_e4_dec_sub05_armw_opt_4m_k3_cr10_fr-5_radius_ar0.05_seed1",
}

PAPER_USETEX = bool(shutil.which("latex") and shutil.which("dvipng"))
_pt = lambda s: s
PAPER_RC = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "text.usetex": PAPER_USETEX,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
    "font.size": _pt(13), "axes.labelsize": _pt(15),
    "xtick.labelsize": _pt(13), "ytick.labelsize": _pt(13),
    "legend.fontsize": _pt(13.0),
    "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": "#999999",
    "grid.linestyle": "--", "grid.linewidth": 0.6, "grid.alpha": 0.3,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4.0, "ytick.major.size": 4.0,
    "xtick.top": False, "ytick.right": False,
}


def arm_of(name, SUB_ADD=SUB_ADD):
    for a, sub in SUB_ADD.items():
        if name == sub:
            return a
    for a, *_ in ARMS:
        if f"_{a}_" in name:
            return a
    return None


def fetch(api, prefix, variant):
    everything = list(api.runs(f"{ENTITY}/{PROJECT}"))
    runs = [r for r in everything if prefix in r.name and arm_of(r.name)]
    if variant == "sub":
        def keep(r):
            tag = SUB_DROP.get(arm_of(r.name))
            return tag is None or tag not in r.name
        runs = [r for r in runs if keep(r)]
        wanted = set(SUB_ADD.values())
        runs += [r for r in everything if r.name in wanted]
    out = defaultdict(list)
    for r in sorted(runs, key=lambda r: r.name):
        present = [k for k in RAW_KEYS if k in r.summary]
        if not present:
            print(f"  ! {r.name}: no E4 metrics, skipped")
            continue
        steps, cols = [], defaultdict(list)
        for row in r.history(keys=present, samples=100000, pandas=False):
            s = row.get("_step")
            if s is None:
                continue
            steps.append(float(s))
            for k in present:
                cols[k].append(float(row[k]) if row.get(k) is not None else np.nan)
        if not steps:
            continue
        o = np.argsort(steps)
        steps = np.asarray(steps)[o]
        series = {k: np.asarray(v)[o] for k, v in cols.items()}
        if "overkills_mean" in series:
            series["false_captures"] = (series["undershoots_mean"]
                                        + series["overkills_mean"])
        else:
            # at-least-k task: an overkill cannot occur, so a failed in-range
            # attempt is exactly an undershoot
            series["false_captures"] = series["undershoots_mean"]
        out[arm_of(r.name)].append((steps, series))
        print(f"  {r.name}")
    return out


def aggregate(seeds, metric, grid, smooth=SMOOTH):
    """Mean with a MIN/MAX-over-seeds band, as the notebook's aggregate() does."""
    import pandas as pd
    rows = []
    for steps, series in seeds:
        if metric not in series:
            continue
        y = series[metric]
        ok = ~np.isnan(y)
        if ok.sum() < 2:
            continue
        rows.append(np.interp(grid, steps[ok], y[ok]))
    if not rows:
        return None
    Y = np.stack(rows)
    if smooth > 1:
        Y = pd.DataFrame(Y.T).rolling(smooth, min_periods=1).mean().to_numpy().T
    return Y.mean(axis=0), Y.min(axis=0), Y.max(axis=0), Y


def legend_row_major(handles, labels, ncol):
    """Permute so the legend READS in the order passed (matplotlib fills columns)."""
    n = len(labels)
    nrows = -(-n // ncol)
    order = [row * ncol + col
             for col in range(ncol) for row in range(nrows)
             if row * ncol + col < n]
    return [handles[i] for i in order], [labels[i] for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--variant", choices=["all", "sub"], default="all",
                    help="sub: E4 only -- seed 3 of SDec-MAT/MAT-Dec replaced "
                         "by the earlier same-config runs")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--png", action="store_true", help="also write PNG previews")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    warnings.filterwarnings("ignore", message="Glyph .* missing")

    import wandb
    api = wandb.Api()
    prefix, has_overkill = TASKS[args.task]
    if args.variant == "sub" and args.task != "e4":
        print("--variant sub is defined for --task e4 only")
        return 1
    print(f"fetching {prefix} runs (task={args.task}, variant={args.variant})...")
    data = fetch(api, prefix, args.variant)
    if not data:
        print("no runs found")
        return 1
    print()
    for a, label, *_ in ARMS:
        print(f"  {label:<10s} {len(data.get(a, []))} seed(s)")

    hi = min(max(s[-1] for s, _ in seeds) for seeds in data.values() if seeds)
    grid = np.linspace(0.0, hi, N_GRID)
    xmax = hi / 1e6
    pad = 0.015 * xmax
    # a 4M run stops at 3,993,600 steps, so the "4" tick is a hair past xmax but
    # still inside xlim -- keep it, otherwise the axis appears to end at 3
    xticks = [t for t in (0, 1, 2, 3, 4, 5) if t <= xmax + pad]

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # results/ is laid out by what each directory MEANS, not by the W&B prefix:
    # `e4m1f0` is the reported E4-m1 (behaviour-weight floor 0) and `e4m1` is
    # the floor-0.05 ablation, so the two swap names on the way out.
    OUTDIR = {"e2": "e2", "e2p": "e2p", "e4": "e4",
              "e4m1f0": "e4m1", "e4m1": "e4m1_floor005_ablation"}
    tag = OUTDIR.get(args.task, args.task)
    if args.variant == "sub":
        tag += "_sub"
    outdir = args.outdir or os.path.join(root, "results", tag, "figures")
    os.makedirs(outdir, exist_ok=True)

    table = {}
    skip = set()
    if not has_overkill:
        # no k_max on this task: overkills are impossible and false captures
        # would be an exact duplicate of undershoots
        skip = {"overkills_mean", "false_captures"}
        print("  (at-least-k task: skipping overkills and false captures -- "
              "no k_max, so neither is defined)")
    for metric, ylabel, stem, ylim in METRICS:
        if metric in skip:
            continue
        with mpl.rc_context(PAPER_RC):
            # The notebook uses layout="constrained" + loc="outside lower center",
            # which needs matplotlib >= 3.7; this env has 3.5.x. Reserving the
            # bottom strip by hand gives the same result on any version: the
            # legend sits INSIDE the page, so saving without bbox_inches keeps
            # the PDF exactly PAPER_FIGSIZE and the fonts come out true size.
            fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
            fig.subplots_adjust(left=0.115, right=0.985, top=0.975, bottom=0.215)
            for arm, label, colour, ls, lw, outline in ARMS:
                seeds = data.get(arm)
                if not seeds:
                    continue
                agg = aggregate(seeds, metric, grid)
                if agg is None:
                    continue
                mean, lo, hig, Y = agg
                if metric == "win_rate":
                    lo, hig = np.clip(lo, 0, 1), np.clip(hig, 0, 1)
                effects = ([pe.withStroke(linewidth=lw + 1.2, foreground=outline),
                            pe.Normal()] if outline else [])
                x = grid / 1e6
                ax.plot(x, mean, label=label, color=colour, ls=ls, lw=lw,
                        zorder=3, path_effects=effects)
                ax.fill_between(x, lo, hig, color=colour,
                                alpha=PAPER_BAND_ALPHA, lw=0, zorder=2)
                finals = [float(np.mean(v[-max(1, len(v) // 10):]))
                          for v in (s[metric][~np.isnan(s[metric])] for _, s in seeds)
                          if v.size]
                table.setdefault(metric, {})[arm] = (
                    float(np.mean(finals)), float(np.std(finals)), len(finals))

            # no title: the paper caption carries it
            ax.set_xlabel(r"Environment Steps ($\times 10^{6}$)")
            ax.set_ylabel(ylabel)
            if ylim:
                ax.set_ylim(*ylim)
            else:
                ax.set_ylim(bottom=0)
            ax.set_xlim(-pad, xmax + pad)
            ax.set_xticks(xticks)
            ax.set_axisbelow(True)

            handles, labels = ax.get_legend_handles_labels()
            handles, labels = legend_row_major(handles, labels, 4)
            fig.legend(handles, labels, loc="lower center",
                       bbox_to_anchor=(0.5, 0.005), ncol=4,
                       frameon=False, handlelength=1.45, columnspacing=0.35,
                       handletextpad=0.18, borderaxespad=0.0)

            fig.savefig(os.path.join(outdir, f"{args.task}_{stem}.pdf"), format="pdf")
            if args.png:
                fig.savefig(os.path.join(outdir, f"{args.task}_{stem}.png"), dpi=200)
            plt.close(fig)
        print(f"  wrote {args.task}_{stem}.pdf")

    labels = {a: l for a, l, *_ in ARMS}
    order = [a for a, *_ in ARMS]
    lines = [f"{args.task}: final value, mean +/- std over seeds",
             "(each seed contributes the mean of its last 10% of logged points;",
             " the figure bands are min/max over seeds, per the notebook's style)", ""]
    shown = [m for m in METRICS if m[0] not in skip]
    head = "%-12s" % "method" + "".join("%28s" % m[1] for m in shown)
    lines += [head, "-" * len(head)]
    for a in order:
        row = "%-12s" % labels[a]
        for metric, *_ in METRICS:
            if metric in skip:
                continue
            e = table.get(metric, {}).get(a)
            row += "%28s" % ("--" if e is None else "%.3f +/- %.3f (n=%d)" % e)
        lines.append(row)
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(outdir, f"{args.task}_table.txt"), "w") as fh:
        fh.write(txt + "\n")

    tex = [r"\begin{tabular}{l" + "c" * len(shown) + "}", r"\toprule",
           "Method & " + " & ".join(m[1] for m in shown) + r" \\", r"\midrule"]
    for a in order:
        cells = []
        for metric, *_ in shown:
            e = table.get(metric, {}).get(a)
            cells.append("--" if e is None else r"$%.2f \pm %.2f$" % (e[0], e[1]))
        nm = r"\textbf{%s}" % labels[a] if a == "ctsde" else labels[a]
        tex.append(nm + " & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(outdir, f"{args.task}_table.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")

    print(f"\nusetex={PAPER_USETEX}"
          f"{'' if PAPER_USETEX else '  (no latex/dvipng -- bundled Computer Modern)'}")
    print("figures in", outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
