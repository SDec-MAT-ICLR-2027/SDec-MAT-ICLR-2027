"""Environment configurations used in the paper.

These are the exact constructor arguments every method was trained with --
SDec-MAT, MAT-Dec, the EPyMARL baselines (MAPPO, CommNet, TarMAC, QMIX) and
InforMARL all built the env from the same values. Change them and the curves
no longer describe the paper's task.

`time_limit` is also the episode length every trainer was given.
"""

PAPER = {
    # 5 identical agents, K ~ U{2,3,4} must PICKUP in the same step. The obs is
    # (K/n_max, team size/n_max), identical for every agent.
    "pickup_consensus": dict(
        n_agents=5, k_min=2, k_max=4, n_max=5, time_limit=10,
    ),
    # 1 sensor + 5 workers on a 10x10 grid, one target with hidden K ~ U{2,3,4}
    # that the sensor reads only on the target cell; 10% wind drift.
    "aerial_ordering": dict(
        obs_mode="simple", grid_size=10, time_limit=100,
        n_workers=5, n_targets=1, k_min=2, k_max=4,
        comm_radius=0, sense_radius=0, wind_prob=0.1,
    ),
    # 6 drones in two arrival cohorts of 3 (at distances 3 and 6) share a
    # 2-port charging station; low arrival battery, so ports must be rationed.
    "charging_rendezvous": dict(
        n_agents=6, grid_size=10, capacity=2,
        cohort_sizes=(3, 3), cohort_distances=(3, 6),
        init_battery=(0.10, 0.15), battery_sigma=0.0, target_level=0.80,
        drain_rate=0.02, charge_rate=0.20, comm_radius=1, dock_radius=0,
        shaping_weight=0.1, wind_prob=0.0, time_limit=100,
    ),
    # 9 agents regroup at a staging cell in two waves (6 near, 3 far) and then
    # split across 3 zones to match a public quota summing to 6 (3 spares).
    "team_partition": dict(
        n_agents=9, n_zones=3, grid_w=9, grid_h=6, time_limit=20,
        n_spare=3, k_min=1, allow_zero_quota=False, fixed_quota=None,
        n_quota_readers=None, spawn_distance=2, wave_gap=3, near_wave_size=6,
        comm_radius=2, commit_gate=True, commit_quorum=None, commit_window=1,
        gate_deadline=None, public_regrouped=False, expose_own_wave=False,
        all_reward=10.0, zone_reward=10.0, wrong_penalty=20.0,
        miss_penalty=25.0, dodge_penalty=0.2, step_cost=0.1,
        step_cost_far=5.0, lambda_coef=0.0, fixed_distances=None,
        shaping_weight=1.0, interference_prob=0.0,
        transit_interference_prob=0.0, drift_at_staging=False,
        symmetry_assert=True,
    ),
}


def paper_kwargs(name, **overrides):
    """A copy of the paper config for `name`, with `overrides` applied."""
    if name not in PAPER:
        raise KeyError(f"unknown MAC env {name!r}; known: {sorted(PAPER)}")
    unknown = set(overrides) - set(PAPER[name]) - _RENDER_KWARGS
    if unknown:
        raise KeyError(f"{name}: unknown env arguments {sorted(unknown)}")
    return {**PAPER[name], **overrides}


# Rendering options the drone envs read out of **kwargs. Accepted as overrides
# so a renderer can be configured without touching the task parameters.
_RENDER_KWARGS = {
    "render_theme", "render_save_dir", "render_dpi", "render_pause",
    "render_prefix", "render_note", "render_trail", "render_terrain",
    "render_terrain_contrast", "render_terrain_seed",
}
