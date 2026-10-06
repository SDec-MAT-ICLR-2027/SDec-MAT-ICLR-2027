"""Audit whether a set of wandb runs is a fair comparison.

Fetches each run's full config and prints, for every key, the value in each run,
flagging any key that is not identical across all of them. Keys that are
*expected* to differ (the arm identity itself, and derived paths) are listed
separately so the real discrepancies stand out.

Usage:
    python scripts/audit_run_fairness.py <run name> <run name> ...
    python scripts/audit_run_fairness.py --contains E2_
"""
import argparse
import os
import sys
from collections import OrderedDict

import wandb

# The W&B project holding the runs. Set both in the environment.
ENTITY = os.environ.get("WANDB_ENTITY")
PROJECT = os.environ.get("WANDB_PROJECT")
if not ENTITY or not PROJECT:
    raise SystemExit("set WANDB_ENTITY and WANDB_PROJECT to the project holding the runs")

# Keys that legitimately differ between arms or are bookkeeping, not settings.
EXPECTED_DIFF = {
    "algorithm_name", "experiment_name", "run_dir", "model_dir", "save_dir",
    "user_name", "wandb_name", "group", "name", "notes", "tags",
    "hostname", "gpu", "device", "num_env_steps",  # horizon is per-task, checked separately
    "dec_actor", "share_actor",                    # arm-defining
    "decoder_self_only", "comm_mask_kind", "encoder_critic_attn",
    "replayed_from", "cuda_deterministic", "n_training_threads",
}

# Keys that MUST match for the comparison to be fair, grouped for readability.
CRITICAL = OrderedDict([
    ("task / env", [
        "n_agents", "num_agents", "num_targets", "k_req", "k_max", "obs_mode",
        "comm_mode", "sense_radius", "obs_radius", "obs_abs_pos", "world_size",
        "episode_length", "min_target_sep", "shuffle_targets", "target_order",
        "capture_radius", "group_radius", "capture_rew", "fail_rew",
        "assembly_rew", "dist_rew_scale", "potential_shaping", "shaping_gamma",
        "step_cost", "collision_rew", "num_obstacles", "max_speed",
        "obs_ready_count", "map_name", "scenario_name", "env_name",
    ]),
    ("switches", [
        "group_commit_eps", "commit_member_prob", "commit_groups",
        # The exploration SCHEDULE. Missing these is how the const-vs-decay
        # mismatch between the E2 arms went unnoticed: eps itself was 0.2
        # everywhere, but eps_final=-1 means constant and eps_final=0 with
        # decay_frac=0.5 means annealed to zero over the first half.
        "group_commit_eps_final", "group_commit_decay_frac",
        "explore_group_capture_eps", "group_shared_noise", "latent_commit_eps",
        "offpolicy_weight", "behav_w_mode", "behav_w_floor",
        "optimistic_adv", "opt_leaky_rate", "dual_clip",
    ]),
    ("optimisation", [
        "lr", "critic_lr", "opti_eps", "weight_decay", "entropy_coef",
        "value_loss_coef", "ppo_epoch", "num_mini_batch", "clip_param",
        "gamma", "gae_lambda", "max_grad_norm", "huber_delta",
        "use_gae", "use_valuenorm", "use_popart", "use_huber_loss",
        "use_clipped_value_loss", "use_max_grad_norm",
        "use_value_active_masks", "use_policy_active_masks",
        "use_linear_lr_decay", "n_rollout_threads", "seed",
    ]),
    ("network", [
        "n_block", "n_embd", "n_head", "use_obs_skip", "group_token",
        "group_latent", "hidden_size", "layer_N", "use_ReLU",
        "use_recurrent_policy", "use_naive_recurrent_policy", "data_chunk_length",
        "actor_graph_aggr", "critic_graph_aggr", "max_edge_dist",
    ]),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="exact run names")
    ap.add_argument("--contains", default=None, help="match runs by substring instead")
    args = ap.parse_args()

    api = wandb.Api()
    all_runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    if args.contains:
        runs = [r for r in all_runs if args.contains.lower() in r.name.lower()]
    else:
        runs = []
        for n in args.names:
            hits = [r for r in all_runs if r.name == n]
            if not hits:
                print(f"!! run not found: {n}")
                continue
            # keep the newest if a name was reused
            runs.append(sorted(hits, key=lambda r: r.created_at)[-1])
    if len(runs) < 2:
        print("need at least 2 runs")
        return 1

    cfgs = [(r.name, dict(r.config), r) for r in runs]
    print(f"{len(cfgs)} runs\n")
    for i, (n, c, r) in enumerate(cfgs):
        steps = c.get("num_env_steps", "?")
        print(f"  [{i}] {n}")
        print(f"      state={r.state} steps={steps} algo={c.get('algorithm_name')}")
    print()

    labels = [f"[{i}]" for i in range(len(cfgs))]
    width = max(len(k) for _, c, _ in cfgs for k in c) + 2

    problems = []
    for group, keys in CRITICAL.items():
        rows = []
        for k in keys:
            present = [k in c for _, c, _ in cfgs]
            if not any(present):
                continue
            vals = [c.get(k, "<ABSENT>") for _, c, _ in cfgs]
            same = all(v == vals[0] for v in vals)
            if not same:
                rows.append((k, vals))
                problems.append((group, k, vals))
        if rows:
            print(f"=== MISMATCH in {group} ===")
            for k, vals in rows:
                print("  %-*s %s" % (width, k,
                      "  |  ".join(f"{l} {v}" for l, v in zip(labels, vals))))
            print()

    # keys present in some runs but not others, outside the vetted lists
    vetted = {k for keys in CRITICAL.values() for k in keys} | EXPECTED_DIFF
    keysets = [set(c) for _, c, _ in cfgs]
    only = []
    for k in sorted(set().union(*keysets)):
        if k in vetted:
            continue
        where = [i for i, ks in enumerate(keysets) if k in ks]
        if len(where) != len(cfgs):
            only.append((k, where))
    if only:
        print("=== keys present in only SOME runs (outside the vetted lists) ===")
        for k, where in only:
            print("  %-*s only in %s" % (width, k, [labels[i] for i in where]))
        print()

    if not problems:
        print("NO MISMATCHES in any critical key. The comparison is fair.")
    else:
        print(f"{len(problems)} critical mismatch(es) -- see above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
