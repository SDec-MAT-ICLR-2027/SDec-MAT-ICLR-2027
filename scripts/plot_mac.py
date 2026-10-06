#!/usr/bin/env python3
"""Paper figures for the MAC benchmark (Pickup Consensus + the three MAC tasks).

    python scripts/plot_mac.py                         # every figure
    python scripts/plot_mac.py --task team_partition   # one benchmark
    python scripts/plot_mac.py --no-refs               # skip the scripted levels

Reads the TensorBoard logs the three training stacks write when launched
through run_experiment.py (run names are matched, timestamps globbed):

  SDec-MAT, MAT-Dec   Multi-Agent-Transformer_ctsde/mat/scripts/results/
                        <Env>/<scenario>/<algorithm>/<arm>_s<seed>[_aid]/runN/logs/<tag>/
  EPyMARL baselines   results/epymarl/tb_logs/<alg>[_aid]_seed<seed>_<gym id>_<time>/
  InforMARL           results/informarl/tb_logs/informarl[_aid]_seed<seed>_<gym id>_<time>/

Each curve is the mean over seeds with a min/max band, resampled onto a common
grid that stops at the shortest seed and smoothed with a 5-point rolling mean.
A missing run is reported and its curve skipped, so partial results still plot.
PDFs go to results/figures/mac/.
"""
import argparse
import contextlib
import io
import math
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
if tuple(int(x) for x in mpl.__version__.split(".")[:2]) < (3, 7):
    sys.exit(f"plot_mac.py needs matplotlib >= 3.7 (found {mpl.__version__}): the "
             "figures use constrained layout's 'outside' legend placement. "
             "pip install 'matplotlib>=3.7,<3.8' (Python 3.8) in a plotting env.")
import matplotlib.patheffects as patheffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mac  # noqa: E402
from configs import mac as MAC  # noqa: E402

MAT_RESULTS = ROOT / "Multi-Agent-Transformer_ctsde" / "mat" / "scripts" / "results"
EPYMARL_TB = ROOT / "results" / "epymarl" / "tb_logs"
INFORMARL_TB = ROOT / "results" / "informarl" / "tb_logs"
OUT_DIR = ROOT / "results" / "figures" / "mac"

SMOOTH = 5      # rolling-mean window over the resampled grid
N_GRID = 300    # common x-grid every seed is resampled onto

# Success is logged under a different name in every benchmark. EPyMARL and
# InforMARL prefix it with test_, the MAT fork with eval_.
SUCCESS_KEY = {
    "pickup_consensus": "success",
    "aerial_ordering": "targets_lifted",       # = success rate with one target
    "charging_rendezvous": "all_satisfied",
    "team_partition": "is_success",
}
# MAT's scenario directory (the train scripts' --scenario_name default).
MAT_SCENARIO = {
    "pickup_consensus": "pickup_consensus",
    "aerial_ordering": "aerial_ordering",
    "charging_rendezvous": "charging_rendezvous",
    "team_partition": "team_partition",
}
RETURN_YLIM = {
    "pickup_consensus": (-10.0, 10.0),
    "aerial_ordering": (-25.0, 22.0),
    "charging_rendezvous": (-185.0, 180.0),
    "team_partition": (-32.0, 62.0),
}
SUCC_YLIM = (-0.03, 1.05)
X_AXIS = {   # (ticks, limits), in millions of steps
    1: ((0.0, 0.2, 0.4, 0.6, 0.8, 1.0), (-0.015, 1.015)),
    2: ((0.0, 0.5, 1.0, 1.5, 2.0), (-0.03, 2.03)),
}

# ----------------------------------------------------------------------------
# Style -- one family across every figure
# ----------------------------------------------------------------------------
PALETTE = {
    "SDec-MAT": "#FF006E", "MAT-Dec": "#7209B7", "MAPPO": "#3A86FF",
    "CommNet": "#FF9E00", "TarMAC": "#00BF63", "QMIX": "#8C564B",
    "InforMARL": "#00E5FF",
}
INFORMARL_OUTLINE = "#0B3C49"     # neon cyan is unreadable on white without it
REF_UPPER_COLOR, REF_LOWER_COLOR = "#111111", "#D62828"
# Dash patterns spelled out so the legend swatch shows the pattern.
LINESTYLE = {
    "SDec-MAT": "-", "MAT-Dec": "-",
    "MAPPO": (0, (2.5, 1.5)), "CommNet": (0, (4, 1.2, 1, 1.2)),
    "TarMAC": (0, (2.5, 1, 0.8, 1, 0.8, 1)), "QMIX": (0, (5, 1.7)),
    "InforMARL": (0, (3, 2)),
}
# SDec-MAT thickest and drawn first, so curves that saturate at the same value
# stay visible as fringes.
LINEWIDTH = {"SDec-MAT": 3.2, "MAT-Dec": 2.4, "MAPPO": 2.0, "CommNet": 2.0,
             "TarMAC": 2.0, "QMIX": 2.0, "InforMARL": 1.8}
BAND_ALPHA = 0.15
FIGSIZE = (7.0, 5.3)

PAPER_RC = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "text.usetex": bool(shutil.which("latex") and shutil.which("dvipng")),
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
    "font.size": 15, "axes.labelsize": 17, "xtick.labelsize": 15,
    "ytick.labelsize": 15, "legend.fontsize": 16,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.linewidth": 0.9, "axes.grid": True,
    "grid.color": "#999999", "grid.linestyle": "--", "grid.linewidth": 0.6,
    "grid.alpha": 0.3, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4.0, "ytick.major.size": 4.0,
}
# Multi-panel figures: type sized for a 9in canvas, line widths scaled with it.
PANEL_SCALE = 9.0 / 7.0
PANEL_RC = {k: round(v * PANEL_SCALE, 2) for k, v in {
    "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 13,
    "figure.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 17}.items()}
XLABEL = r"Environment Steps ($\times 10^{6}$)"


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def read_scalar(path, tag=None):
    """(steps, values) of one scalar from one TensorBoard directory."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    with contextlib.redirect_stderr(io.StringIO()):
        ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
        ea.Reload()
    tags = ea.Tags()["scalars"]
    tag = tag or (tags[0] if len(tags) == 1 else None)
    if tag not in tags:
        raise KeyError(f"{path}: no scalar {tag!r} (has {tags})")
    ev = ea.Scalars(tag)
    return (np.array([e.step for e in ev], float), np.array([e.value for e in ev], float))


def _latest(hits, what):
    hits = sorted(hits)
    if len(hits) > 1:
        warnings.warn(f"{len(hits)} runs match {what}; using the newest, {hits[-1].name}")
    return hits[-1] if hits else None


def find_runs(task, arm, cond, metric, seeds):
    """[(tensorboard dir, tag)] for every seed of (task, arm, id condition)
    that exists on disk. `metric` is "return" or "success"."""
    stack = MAC.ARMS[arm]["stack"]
    suffix = "_aid" if cond == "on" else ""
    gym_id = mac.GYM_IDS[task]
    key = "return" if metric == "return" else SUCCESS_KEY[task]
    out = []
    for s in seeds:
        if stack == "mat":
            algo = MAC.MAT_ARM_FLAGS[arm]["algorithm_name"]
            exp = (MAT_RESULTS / MAC.TASKS[task]["mat_env_name"] / MAT_SCENARIO[task]
                   / algo / f"{arm}_s{s}{suffix}")
            run = _latest([p for p in exp.glob("run*") if p.is_dir()], exp)
            tag = f"eval_{key}_mean"
            path = run / "logs" / tag / tag if run else None
            hit = (path, None) if path is not None and path.exists() else None
        else:
            root, name = ((EPYMARL_TB, f"{MAC.ARMS[arm]['config']}{suffix}")
                          if stack == "epymarl" else (INFORMARL_TB, f"informarl{suffix}"))
            run = _latest(root.glob(f"{name}_seed{s}_{gym_id}_*"), f"{name} seed {s}")
            hit = (run, f"test_{key}_mean") if run else None
        if hit is None:
            print(f"  [missing] {task} {arm} id-{cond} seed {s}")
        else:
            out.append(hit)
    return out


def aggregate(runs):
    """Mean and min/max over seeds on a grid that stops at the shortest seed."""
    xs, ys = zip(*(read_scalar(p, tag) for p, tag in runs))
    grid = np.linspace(0.0, min(x.max() for x in xs), N_GRID)
    Y = np.stack([np.interp(grid, x, y) for x, y in zip(xs, ys)])
    if SMOOTH > 1:
        Y = pd.DataFrame(Y.T).rolling(SMOOTH, min_periods=1).mean().to_numpy().T
    return grid, Y.mean(0), Y.min(0), Y.max(0)


# ----------------------------------------------------------------------------
# Series and reference levels
# ----------------------------------------------------------------------------
def method_series(task, metric, cond, id_suffix=True):
    """Curves for one figure: SDec-MAT (never given an id), then every other
    method in the given agent-id condition, if the paper reports it."""
    id_label = {"off": " w/o ID", "on": " w/ ID"}[cond] if id_suffix else ""
    out = []
    for arm in MAC.ARMS:
        if arm == "sdec_mat":
            arm_cond, label = "off", "SDec-MAT"
        elif cond in MAC.id_conditions_for(task, arm):
            arm_cond, label = cond, MAC.ARMS[arm]["label"] + id_label
        else:
            continue
        runs = find_runs(task, arm, arm_cond, metric, MAC.seeds_for(task, arm))
        if runs:
            name = MAC.ARMS[arm]["label"]
            out.append(dict(label=label, runs=runs, color=PALETTE[name],
                            ls=LINESTYLE[name], lw=LINEWIDTH[name],
                            outline=INFORMARL_OUTLINE if name == "InforMARL" else None))
    return out


_LEVELS = {}


def return_levels(task):
    """The two scripted levels the paper draws on return figures."""
    if task not in _LEVELS:
        from mac.reference_policies import drawn_levels, measure_levels
        print(f"  measuring reference levels for {task} ...")
        _LEVELS[task] = (measure_levels(task), drawn_levels(task))
    levels, (upper, lower) = _LEVELS[task]
    label = {"Perfect": r"Perfect ($K$ PICKUP)", "PICKUP w.p. K/N": r"PICKUP w.p. $K/N$"}
    return [dict(value=levels[upper]["return_mean"], label=label.get(upper, upper),
                 color=REF_UPPER_COLOR, ls=":"),
            dict(value=levels[lower]["return_mean"], label=label.get(lower, lower),
                 color=REF_LOWER_COLOR, ls=":")]


def pickup_shared_ceiling(label=r"Shared policy, $p = K/N$"):
    """Best success of a shared policy without ids: P(Binomial(N, K/N) = K),
    averaged over K (analytic, single attempt)."""
    cfg = mac.PAPER["pickup_consensus"]
    n = cfg["n_agents"]
    ks = range(cfg["k_min"], cfg["k_max"] + 1)
    p = np.mean([math.comb(n, k) * (k / n) ** k * (1 - k / n) ** (n - k) for k in ks])
    return [dict(value=float(p), label=label, color=REF_LOWER_COLOR, ls="--")]


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def _shown(path):
    """`path` relative to the repo when it is inside it, else as given."""
    try:
        return path.resolve().relative_to(ROOT)
    except ValueError:
        return path


# ----------------------------------------------------------------------------
def _outline(s, extra=1.2):
    if not s.get("outline"):
        return []
    return [patheffects.withStroke(linewidth=s["lw"] + extra, foreground=s["outline"]),
            patheffects.Normal()]


def _row_major(handles, labels, ncol):
    """Reorder so the legend READS left-to-right in the order given."""
    nrows = -(-len(labels) // ncol)
    order = [r * ncol + c for c in range(ncol) for r in range(nrows)
             if r * ncol + c < len(labels)]
    return [handles[i] for i in order], [labels[i] for i in order]


def _draw(ax, series, refs, lw_scale=1.0, outline_extra=1.2):
    for s in series:
        grid, mean, lo, hi = aggregate(s["runs"])
        x = grid / 1e6
        ax.plot(x, mean, label=s["label"], color=s["color"], ls=s["ls"],
                lw=s["lw"] * lw_scale, zorder=3,
                path_effects=_outline(dict(s, lw=s["lw"] * lw_scale), outline_extra))
        ax.fill_between(x, lo, hi, color=s["color"], alpha=BAND_ALPHA, lw=0, zorder=2)
    for r in refs:
        ax.axhline(r["value"], ls=r["ls"], lw=2.0 * lw_scale, color=r["color"],
                   alpha=0.9, zorder=1, label=r["label"])
    ax.set_axisbelow(True)


def single_figure(task, metric, cond, out):
    series = method_series(task, metric, cond)
    if not series:
        print(f"  [skip] {out.name}: no runs found")
        return
    refs = return_levels(task) if metric == "return" else (
        pickup_shared_ceiling() if task == "pickup_consensus" else [])
    xticks, xlim = X_AXIS[MAC.TASKS[task]["steps"] // 1_000_000]
    with mpl.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
        _draw(ax, series, refs)
        ax.set_xlabel(XLABEL)
        ax.set_ylabel("Mean Episodic Return" if metric == "return" else "Success Rate")
        ax.set_ylim(*(RETURN_YLIM[task] if metric == "return" else SUCC_YLIM))
        ax.set_xlim(*xlim)
        ax.set_xticks(list(xticks))
        h, lab = _row_major(*ax.get_legend_handles_labels(), 3)
        fig.legend(h, lab, loc="outside lower center", ncol=3, frameon=False,
                   handlelength=1.45, columnspacing=0.35, handletextpad=0.18,
                   borderaxespad=0.0)
        fig.savefig(out, format="pdf")
        plt.close(fig)
    print(f"  wrote {_shown(out)}")


def panel_figure(panels, out, ylabel, ylim, yticks, legend_ncol=4,
                 figsize=(10.5, 4.6)):
    """Several panels sharing y, one legend built from the first panel."""
    if not all(p["series"] for p in panels):
        print(f"  [skip] {out.name}: a panel has no runs")
        return
    with mpl.rc_context({**PAPER_RC, **PANEL_RC}):
        fig, axes = plt.subplots(1, len(panels), figsize=figsize,
                                 layout="constrained", sharey=True)
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.06, hspace=0.0)
        axes = np.atleast_1d(axes)
        for ax, p in zip(axes, panels):
            _draw(ax, p["series"], p.get("refs", []), lw_scale=PANEL_SCALE,
                  outline_extra=1.2 * PANEL_SCALE)
            ax.set_title(p["title"])
            ax.set_ylim(*ylim)
            ax.set_xlim(*p["xlim"])
            ax.set_xticks(list(p["xticks"]))
            ax.set_yticks(list(yticks))
        mid = axes[(len(axes) - 1) // 2]
        mid.set_xlabel(XLABEL)
        axes[0].set_ylabel(ylabel)
        h, lab = _row_major(*axes[0].get_legend_handles_labels(), legend_ncol)
        fig.legend(h, lab, loc="outside lower center", ncol=legend_ncol,
                   frameon=False, handlelength=1.45, columnspacing=0.35,
                   handletextpad=0.18, borderaxespad=0.0)
        if len(axes) % 2 == 0:       # even count: centre the x label on the gap
            fig.canvas.draw()
            fig.set_layout_engine("none")
            left, right = mid.get_position(), axes[len(axes) // 2].get_position()
            mid.xaxis.label.set_x(((left.x1 + right.x0) / 2 - left.x0) / left.width)
        fig.savefig(out, format="pdf")
        plt.close(fig)
    print(f"  wrote {_shown(out)}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(MAC.TASKS), nargs="+", default=list(MAC.TASKS))
    ap.add_argument("--no-refs", action="store_true",
                    help="skip the scripted reference levels on return figures")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.no_refs:
        global return_levels
        return_levels = lambda task: []  # noqa: E731

    for task in a.task:
        print(f"\n{MAC.TASKS[task]['title']}")
        conds = sorted({c for arm in MAC.ARMS if arm != "sdec_mat"
                        for c in MAC.id_conditions_for(task, arm)}, reverse=True)
        for metric in ("return", "success"):
            for cond in conds:
                tag = "with_id" if cond == "on" else "without_id"
                single_figure(task, metric, cond, a.out / f"{task}_{metric}_{tag}.pdf")

    if "pickup_consensus" in a.task:
        print("\nPickup Consensus, both conditions")
        xt, xl = X_AXIS[1]
        refs = pickup_shared_ceiling(label=r"$p = K/N$")
        panel_figure(
            [dict(title="(a) CTDE without Agent IDs", xticks=xt, xlim=xl, refs=refs,
                  series=method_series("pickup_consensus", "success", "off", False)),
             dict(title="(b) CTDE with Static Agent IDs", xticks=xt, xlim=xl, refs=refs,
                  series=method_series("pickup_consensus", "success", "on", False))],
            a.out / "pickup_consensus_success.pdf", "Success Rate", SUCC_YLIM,
            (0.0, 0.5, 1.0))

    three = ["aerial_ordering", "charging_rendezvous", "team_partition"]
    if all(t in a.task for t in three):
        print("\nMAC, three panels (agent ids on)")
        # Panel titles carry the team composition, as in the paper.
        code = {"aerial_ordering": "5w1s1t", "charging_rendezvous": "6n1s2c",
                "team_partition": "9n6a3r"}
        # Five ticks per panel, so the 1M- and 2M-step panels look alike.
        ticks = {1: (0.0, 0.25, 0.5, 0.75, 1.0), 2: (0.0, 0.5, 1.0, 1.5, 2.0)}
        panels = []
        for t in three:
            n = MAC.TASKS[t]["steps"] // 1_000_000
            panels.append(dict(title=f"{MAC.TASKS[t]['title']}\n({code[t]})",
                               xticks=ticks[n], xlim=X_AXIS[n][1],
                               series=method_series(t, "success", "on", False)))
        panel_figure(panels, a.out / "mac_success_with_id.pdf", "Success Rate",
                     SUCC_YLIM, (0.0, 0.25, 0.5, 0.75, 1.0))


if __name__ == "__main__":
    main()
