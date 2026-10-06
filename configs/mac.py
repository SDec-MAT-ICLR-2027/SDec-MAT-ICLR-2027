"""MAC benchmark experiments (Pickup Consensus + Aerial Ordering, Charging
Rendezvous, Team Partition).

Every setting here is the one the paper's runs used, recovered from their
launch commands and logged configs. Three code bases train the methods:

  SDec-MAT, MAT-Dec            Multi-Agent-Transformer_ctsde (MAT fork)
  MAPPO, CommNet, TarMAC, QMIX epymarl/ (EPyMARL fork)
  InforMARL                    InforMARL/mac_adapter

All of them build the environment from mac/presets.py, so within a task every
method plays the same game.

Agent ids. Pickup Consensus is reported with and without a one-hot agent id
for the CTDE methods: without one, a shared policy cannot break the symmetry
between identical agents. The three MAC tasks then use ids throughout.
SDec-MAT never receives an id -- its autoregressive decoder breaks the tie.
"""

# ----------------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------------
# steps           environment steps per run
# episode_length  = the env's time_limit; MAT and InforMARL roll out this long
# mat_env_name    MAT's --env_name (also the results sub-directory)
# mat_script      MAT train script
# mat_flags       extra MAT flags for this task, for both transformer arms
# qmix_eps_anneal epsilon_anneal_time for QMIX, per id condition
# id_conditions   {arm: conditions reported in the paper}; "off" / "on"
# seeds           {arm group: seeds reported in the paper}
TASKS = {
    "pickup_consensus": {
        "title": "Pickup Consensus",
        "steps": 1_000_000,
        "episode_length": 10,
        "mat_env_name": "PickupConsensus",
        "mat_script": "train_pickup_consensus.py",
        # The raw-observation LayerNorm erases Pickup Consensus's 2-d obs
        # ([K/N, N/n_max] normalises to the same +-1 pattern for every K).
        "mat_flags": {"no_raw_obs_layernorm": True},
        "qmix_eps_anneal": {"off": 50_000, "on": 50_000},
        "id_conditions": {
            "sdec_mat": ("off",),
            "mat_dec": ("off", "on"),
            "mappo": ("off", "on"), "commnet": ("off", "on"),
            "tarmac": ("off", "on"), "qmix": ("off", "on"),
            "informarl": ("off", "on"),
        },
        "seeds": {"epymarl": (41, 42, 43, 44, 45), "mat": (42, 43, 44),
                  "informarl": (42, 43, 44)},
    },
    "aerial_ordering": {
        "title": "Aerial Ordering",
        "steps": 1_000_000,
        "episode_length": 100,
        "mat_env_name": "Aerial",
        "mat_script": "train_aerial.py",
        "mat_flags": {},
        "qmix_eps_anneal": {"off": 50_000, "on": 500_000},
        "id_conditions": {
            "sdec_mat": ("off",),
            "mat_dec": ("on",),
            "mappo": ("on",), "commnet": ("on",), "tarmac": ("on",), "qmix": ("on",),
            "informarl": ("on",),
        },
        "seeds": {"epymarl": (41, 42, 43, 44, 45), "mat": (42, 43, 44),
                  "informarl": (42, 43, 44)},
    },
    "charging_rendezvous": {
        "title": "Charging Rendezvous",
        "steps": 2_000_000,
        "episode_length": 100,
        "mat_env_name": "Charging",
        "mat_script": "train_charging.py",
        "mat_flags": {},
        "qmix_eps_anneal": {"off": 1_000_000, "on": 1_000_000},
        "id_conditions": {
            "sdec_mat": ("off",),
            "mat_dec": ("on",),
            "mappo": ("on",), "commnet": ("on",), "tarmac": ("on",), "qmix": ("on",),
            "informarl": ("on",),
        },
        "seeds": {"epymarl": (42, 43, 44), "mat": (42, 43, 44),
                  "informarl": (42, 43, 44)},
    },
    "team_partition": {
        "title": "Team Partition",
        "steps": 1_000_000,
        "episode_length": 20,
        "mat_env_name": "TeamPartition",
        "mat_script": "train_team_partition.py",
        "mat_flags": {},
        "qmix_eps_anneal": {"off": 1_000_000, "on": 1_000_000},
        "id_conditions": {
            "sdec_mat": ("off",),
            "mat_dec": ("on",),
            "mappo": ("on",), "commnet": ("on",), "tarmac": ("on",), "qmix": ("on",),
            "informarl": ("on",),
        },
        "seeds": {"epymarl": (42, 43, 44), "mat": (42, 43, 44),
                  "informarl": (42, 43, 44)},
    },
}

# ----------------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------------
# stack: which code base trains it ("mat", "epymarl", "informarl").
ARMS = {
    "sdec_mat": {"label": "SDec-MAT", "stack": "mat"},
    "mat_dec": {"label": "MAT-Dec", "stack": "mat"},
    "mappo": {"label": "MAPPO", "stack": "epymarl", "config": "mappo"},
    "commnet": {"label": "CommNet", "stack": "epymarl", "config": "commnet"},
    "tarmac": {"label": "TarMAC", "stack": "epymarl", "config": "tarmac"},
    "qmix": {"label": "QMIX", "stack": "epymarl", "config": "qmix"},
    "informarl": {"label": "InforMARL", "stack": "informarl"},
}

# --- MAT fork (SDec-MAT, MAT-Dec) ---------------------------------------------
# Shared by both transformer arms. Evaluation SAMPLES from the policy
# (eval_deterministic false): with argmax, agents that see identical
# observations all pick the same action, so a decentralised actor could never
# score on an exact-K task, and the CTDE baselines are evaluated by sampling too.
MAT_COMMON = {
    "n_rollout_threads": 32,
    "lr": 3e-4,
    "ppo_epoch": 4,
    "clip_param": 0.05,
    "entropy_coef": 0.1,
    "n_block": 1,
    "n_embd": 64,
    "n_head": 1,
    "use_eval": True,
    "eval_episodes": 100,
    "n_eval_rollout_threads": 4,
    "eval_deterministic": "false",
    "log_interval": 5,
    "save_interval": 50,
}

# SDec-MAT: the communication-masked encoder plus the autoregressive decoder,
# in its plain form (no group token / latent / commitment, no exploration
# switches).
MAT_ARM_FLAGS = {
    "sdec_mat": {
        "algorithm_name": "mat_comm",
        "comm_mask_kind": "bits",
        "encoder_critic_attn": "soft_group_bias",
        "use_obs_skip": 0,
        "group_token": 0,
        "group_latent": 0,
        "commitment": 0,
        "group_commit_eps": 0,
        "latent_commit_eps": 0,
        "offpolicy_weight": 0,
        "optimistic_adv": 0,
        "critic_global_state": 0,
        "decoder_self_only": 0,
        "decoder_attn_gain": 0.01,
    },
    # MAT-Dec: shared per-agent MLP actor (decentralised execution).
    "mat_dec": {
        "algorithm_name": "mat_dec",
    },
}

# --- EPyMARL (MAPPO, CommNet, TarMAC, QMIX) ------------------------------------
# Everything else comes from epymarl/config/algs/<alg>.yaml and
# epymarl/config/envs/<task>.yaml.
EPYMARL_COMMON = {
    "use_tensorboard": "True",
    "save_model": "True",
    "save_model_interval": 100_000,
}

# --- InforMARL ------------------------------------------------------------------
INFORMARL_COMMON = {
    "n_rollout_threads": 10,
    "data_chunk_length": 10,
    "num_mini_batch": 1,
    "ppo_epoch": 4,
    "lr": 3e-4,
    "critic_lr": 3e-4,
    "clip_param": 0.2,
    "entropy_coef": 0.1,
    "max_grad_norm": 0.5,
    "gamma": 0.99,
    "hidden_size": 128,
}


def seeds_for(task, arm):
    """The seeds the paper reports for (task, arm)."""
    return TASKS[task]["seeds"][ARMS[arm]["stack"]]


def id_conditions_for(task, arm):
    """The agent-id conditions the paper reports for (task, arm)."""
    return TASKS[task]["id_conditions"][arm]
