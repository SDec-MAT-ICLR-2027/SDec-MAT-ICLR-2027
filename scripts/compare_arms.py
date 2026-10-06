"""
Final-performance comparison across arms, reading whatever metric name each
codebase used. Prints one table per task.

InforMARL logs `joint_nav/*`; the MAT fork logs `*_mean`. This script maps both
onto one schema so the comparison does not depend on the rename replay.
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import wandb

# The W&B project holding the runs. Set both in the environment.
ENTITY = os.environ.get("WANDB_ENTITY")
PROJECT = os.environ.get("WANDB_PROJECT")
if not ENTITY or not PROJECT:
    raise SystemExit("set WANDB_ENTITY and WANDB_PROJECT to the project holding the runs")

# canonical -> the names either codebase might have used
ALIASES = {
    "captures": ["joint_captures_mean", "joint_nav/joint_captures"],
    "win": ["win_rate", "joint_nav/win_rate"],
    "overkills": ["overkills_mean", "joint_nav/overkills"],
    "undershoots": ["undershoots_mean", "joint_nav/undershoots"],
    "cap_acts": ["capture_actions_mean", "joint_nav/capture_actions"],
    "return": ["return_mean", "average_episode_rewards"],
}


def task_of(name):
    """Bucket a run name into the task it belongs to."""
    n = name.lower()
    if "e4m1" in n or ("k3" in n and "fr-1" in n and "e4" in n):
        return "E4-m1 (exact-k, miss -1)"
    if "e4" in n:
        return "E4 (exact-k, miss -5)"
    if "e2p" in n or "fr-5" in n:
        return "E2-p (radius, miss -5)"
    if "e3" in n or "9a4t" in n:
        return "E3 (9a/4t)"
    return "E2 (radius, miss -1)"


def tail_mean(run, frac=0.1):
    """Mean of each metric over the last `frac` of logged points."""
    # Scope the fetch to these columns; an unscoped scan_history() pulls every
    # per-agent column and takes minutes per run.
    # `history(keys=...)` intersects: a key absent from this run drops every
    # row. Each codebase logs only its own spelling, so ask for what is there.
    wanted = {a: c for c, al in ALIASES.items() for a in al if a in run.summary}
    series = defaultdict(list)
    if not wanted:
        return {}
    for row in run.history(keys=list(wanted), samples=1000000, pandas=False):
        for raw, canon in wanted.items():
            v = row.get(raw)
            if v is not None:
                series[canon].append(v)
    out = {}
    for canon, vals in series.items():
        if not vals:
            continue
        k = max(1, int(len(vals) * frac))
        out[canon] = float(np.nanmean(vals[-k:]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contains", default="", help="only runs whose name contains this")
    ap.add_argument("--frac", type=float, default=0.1,
                    help="fraction of the tail to average (default 0.1)")
    args = ap.parse_args()

    api = wandb.Api()
    runs = [r for r in api.runs(f"{ENTITY}/{PROJECT}")
            if r.state == "finished"
            and args.contains.lower() in r.name.lower()
            and "matnames" not in r.tags]

    by_task = defaultdict(list)
    for r in runs:
        stats = tail_mean(r, args.frac)
        if "captures" not in stats:
            continue          # not a joint-nav run
        by_task[task_of(r.name)].append((r.name, stats))

    cols = ["captures", "win", "overkills", "undershoots", "cap_acts", "return"]
    for task in sorted(by_task):
        print(f"\n=== {task} ===")
        print("%-62s %s" % ("run", "  ".join("%9s" % c for c in cols)))
        for name, s in sorted(by_task[task], key=lambda t: -t[1].get("captures", 0)):
            cells = "  ".join(
                ("%9.3f" % s[c]) if c in s else "%9s" % "-" for c in cols)
            print("%-62s %s" % (name[:62], cells))
    print("\n(mean over the last %.0f%% of logged points)" % (args.frac * 100))


if __name__ == "__main__":
    main()
