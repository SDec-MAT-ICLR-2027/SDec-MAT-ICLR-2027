"""The four Joint-Action Navigation tasks.

Each entry is the environment rule, the step budget and the three learning
switches. Every method within a task receives exactly this block, which is what
makes the comparison controlled.

Two switches vary by task, and both follow a stated property of the task rather
than a sweep:

  commit_member_prob   0.5 on the exact-k tasks, 1.0 otherwise. With q=1 a whole
                       communication group fires at once, which under an
                       exact-k rule is always an overkill, so the
                       three-fire / three-hold split could never be explored.
  optimistic_adv       on exactly when the miss penalty is -5, where exploration
                       otherwise yields almost uniformly negative advantages.

`group_commit_eps_final = -1` means "no decay, hold eps constant"; `0` means
anneal to zero over `group_commit_decay_frac` of training.
"""

_SWITCHES_OFF = {
    "commit_groups": "comm",
    "group_commit_eps": 0.0,
    "group_commit_eps_final": -1.0,
    "group_commit_decay_frac": 1.0,
    "commit_member_prob": 1.0,
    "offpolicy_weight": 0,
    "behav_w_mode": "arm",
    "behav_w_floor": 0.0,
    "optimistic_adv": 0,
    "opt_leaky_rate": 0.0,
}

TASKS = {
    # at-least-k, weak penalty. Exploration is annealed to zero here.
    "e2": {
        "desc": "at-least-k capture, miss -1",
        "env": {"fail_rew": -1.0},
        "num_env_steps": 2_000_000,
        "seeds": (1, 2, 3),
        "switches": {
            "commit_groups": "comm",
            "group_commit_eps": 0.2,
            "group_commit_eps_final": 0.0,
            "group_commit_decay_frac": 0.5,
            "commit_member_prob": 1.0,
            "offpolicy_weight": 1,
            "behav_w_mode": "arm",
            "behav_w_floor": 0.0,
            "optimistic_adv": 0,
            "opt_leaky_rate": 0.0,
        },
    },
    # at-least-k, harsh penalty.
    "e2p": {
        "desc": "at-least-k capture, miss -5",
        "env": {"fail_rew": -5.0},
        "num_env_steps": 2_000_000,
        "seeds": (1, 2, 3),
        "switches": {
            "commit_groups": "comm",
            "group_commit_eps": 0.2,
            "group_commit_eps_final": -1.0,
            "group_commit_decay_frac": 1.0,
            "commit_member_prob": 1.0,
            "offpolicy_weight": 1,
            "behav_w_mode": "arm",
            "behav_w_floor": 0.0,
            "optimistic_adv": 1,
            "opt_leaky_rate": 0.0,
        },
    },
    # exact-k, harsh penalty: more than k_max simultaneous firers is an overkill.
    "e4": {
        "desc": "exact-k capture, miss -5",
        "env": {"fail_rew": -5.0, "k_max": 3},
        "num_env_steps": 4_000_000,
        "seeds": (1, 2, 3),
        "switches": {
            "commit_groups": "comm",
            "group_commit_eps": 0.2,
            "group_commit_eps_final": -1.0,
            "group_commit_decay_frac": 1.0,
            "commit_member_prob": 0.5,
            "offpolicy_weight": 1,
            "behav_w_mode": "arm",
            "behav_w_floor": 0.0,
            "optimistic_adv": 1,
            "opt_leaky_rate": 0.0,
        },
    },
    # exact-k, weak penalty.
    "e4m1": {
        "desc": "exact-k capture, miss -1",
        "env": {"fail_rew": -1.0, "k_max": 3},
        "num_env_steps": 4_000_000,
        "seeds": (2, 3, 4),
        "switches": {
            "commit_groups": "comm",
            "group_commit_eps": 0.2,
            "group_commit_eps_final": -1.0,
            "group_commit_decay_frac": 1.0,
            "commit_member_prob": 0.5,
            "offpolicy_weight": 1,
            "behav_w_mode": "arm",
            "behav_w_floor": 0.0,
            "optimistic_adv": 0,
            "opt_leaky_rate": 0.0,
        },
    },
    # Ablation: E4-m1 with a non-zero floor on the off-policy weight. Reported
    # separately; the floor turned out to hurt rather than help, which is why
    # the main E4-m1 result above uses floor 0 like every other task.
    "e4m1_floor005": {
        "desc": "exact-k capture, miss -1, behaviour-weight floor 0.05",
        "env": {"fail_rew": -1.0, "k_max": 3},
        "num_env_steps": 4_000_000,
        "seeds": (2, 3, 4),
        "switches": {
            "commit_groups": "comm",
            "group_commit_eps": 0.2,
            "group_commit_eps_final": -1.0,
            "group_commit_decay_frac": 1.0,
            "commit_member_prob": 0.5,
            "offpolicy_weight": 1,
            "behav_w_mode": "arm",
            "behav_w_floor": 0.05,
            "optimistic_adv": 0,
            "opt_leaky_rate": 0.0,
        },
    },
}


def switches_for(task, base=False):
    """The switch block for `task`, or the all-off base configuration.

    `base=True` reproduces the paper's base model: no group-commit exploration,
    no off-policy weighting, no optimistic advantages. These are also the
    defaults in `mat/config.py`, so the base run needs no switch flags at all.
    """
    if base:
        return dict(_SWITCHES_OFF)
    return dict(TASKS[task]["switches"])
