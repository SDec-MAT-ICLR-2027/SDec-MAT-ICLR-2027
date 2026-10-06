"""Settings shared by every task and every method.

These are identical across all reported runs. Anything that varies by task is
in `tasks.py`; anything that varies by method is in `arms.py`.
"""

# --- environment -----------------------------------------------------------
# The 6-agent / 3-target joint-capture arena. `k_req=3` agents must fire on the
# same target on the same step for it to fall.
ENV = {
    "map_name": "joint_nav_6a3t_k3",
    "n_agents": 6,
    "num_targets": 3,
    "k_req": 3,
    "obs_mode": "radius",          # partial observability, 32-d observation
    "comm_mode": "agent_radius",   # the graph is pure distance, no task knowledge
    "sense_radius": 0.6,           # >= 2 * capture_radius, so a ready group is connected
    "world_size": 2.0,
    "episode_length": 50,
    "min_target_sep": 1.25,
    "shuffle_targets": 1,          # removes the target-id / placement artefact
    "target_order": "distance",    # id-free observation slot order
    "capture_radius": 0.3,
    "group_radius": 0.3,
    "capture_rew": 10.0,
    "assembly_rew": 0.05,
    "dist_rew_scale": 1.0,
    "potential_shaping": 1,        # must stay on; see docs
    "shaping_gamma": 1.0,
}

# InforMARL's parser spells several of these differently and needs a few extra
# flags; everything here has the same value as its ENV counterpart.
ENV_GRAPH_RENAMES = {
    "n_agents": "num_agents",
}
ENV_GRAPH_EXTRA = {
    "env_name": "GraphMPE",
    "scenario_name": "navigation_joint_graph",
    "num_landmarks": 3,
    "max_edge_dist": 0.6,          # tied to sense_radius
    "num_obstacles": 0,
    "collision_rew": 0.0,
    "max_speed": 2.0,
    "min_dist_thresh": 0.1,
    "use_dones": "False",
}
# `map_name` is a MAT-side label with no InforMARL equivalent.
ENV_GRAPH_DROP = {"map_name"}

# --- optimisation ----------------------------------------------------------
# Identical in all 60 reported runs, verified against the logged configs.
PPO = {
    "lr": 5e-4,
    "entropy_coef": 0.01,
    "ppo_epoch": 5,
    "num_mini_batch": 1,
    "clip_param": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "value_loss_coef": 1,
    "max_grad_norm": 10,
    "huber_delta": 10,
    "layer_N": 2,
    "n_rollout_threads": 32,
    "n_training_threads": 1,
    "log_interval": 5,
    "save_interval": 50,
}

# --- network ---------------------------------------------------------------
# The plain CT-SDE form: no head-agent token, no observation skip to the logits.
NET = {
    "use_obs_skip": 0,
    "group_token": 0,
    "group_latent": 0,
    "n_block": 1,
    "n_embd": 64,
    "n_head": 1,
}
