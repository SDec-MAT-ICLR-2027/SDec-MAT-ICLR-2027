"""
Wrapper-level tests for joint-capture navigation.

Checks the seams between the InforMARL scenario and the MAT vec-env API:
  1. obs / share_obs dimensions for both arms and both obs modes
  2. the comm tail of share_obs decodes back to the target-clique adjacency
  3. grouping semantics: same-target agents grouped, distant agents singleton
  4. obs_mode="shared" really does expose other agents and targets
  5. the joint-capture outcome survives the wrapper (capture + undershoot)

Run:  python -m mat.test_joint_nav      (from Multi-Agent-Transformer_ctsde/)
"""
import argparse
import sys

import numpy as np

from mat.envs.joint_nav import JointNavMATEnv

CAPTURE = 5


def make_args(**over):
    args = argparse.Namespace(
        algorithm_name="mat_comm_soft",
        n_agents=6,
        num_targets=3,
        k_req=3,
        episode_length=50,
        capture_radius=0.3,
        capture_rew=10.0,
        fail_rew=-1.0,
        assembly_rew=0.0,
        dist_rew_scale=0.0,
        collision_rew=0.0,
        world_size=2.0,
        num_obstacles=0,
        max_speed=2.0,
        obs_mode="local",
        obs_ready_count=False,
    )
    for k, v in over.items():
        setattr(args, k, v)
    return args


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(cond)


def place(env, i, pos):
    env._world.agents[i].state.p_pos = np.array(pos, dtype=np.float64)
    env._world.agents[i].state.p_vel = np.zeros(2)


def main():
    ok = True

    print("== 1. spaces ==")
    e_local = JointNavMATEnv(make_args(obs_mode="local"))
    ok &= check("local obs dim 8", e_local.observation_space[0][0] == 8,
                f"got {e_local.observation_space[0][0]}")
    ok &= check("action space 6", e_local.action_space[0].n == 6,
                f"got {e_local.action_space[0].n}")

    n, m = 6, 3
    want_shared = 8 + 2 * (n - 1) + 3 * m
    e = JointNavMATEnv(make_args(obs_mode="shared"))
    ok &= check(f"shared obs dim {want_shared}",
                e.observation_space[0][0] == want_shared,
                f"got {e.observation_space[0][0]}")
    ok &= check(
        "share_obs = n*obs + n^2 comm tail",
        e.share_observation_space[0][0] == n * want_shared + n * n,
        f"got {e.share_observation_space[0][0]}",
    )

    e_mappo = JointNavMATEnv(make_args(algorithm_name="mappo", obs_mode="shared"))
    ok &= check(
        "mappo share_obs has NO comm tail",
        e_mappo.share_observation_space[0][0] == n * want_shared,
        f"got {e_mappo.share_observation_space[0][0]}",
    )

    print("\n== 2. comm tail decodes to the target-clique adjacency ==")
    from mat.algorithms.utils.mat_comm_groups import comm_from_share_obs

    obs, share_obs, avail = e.reset()
    ok &= check("obs batch is [n, dim]", obs.shape == (n, want_shared), f"{obs.shape}")
    ok &= check("avail all ones", np.all(avail == 1.0))
    import torch

    # comm_from_share_obs wants a torch [B, N, D] batch of tiled share_obs
    decoded = comm_from_share_obs(
        torch.from_numpy(share_obs).unsqueeze(0), n, "pairwise"
    )
    decoded = decoded.reshape(n, n).cpu().numpy()
    direct = e.get_comm_mask()
    ok &= check(
        "decoded tail == get_comm_mask()",
        np.array_equal(decoded > 0.5, direct > 0.5),
        f"decoded\n{decoded}\ndirect\n{direct}",
    )

    print("\n== 3. grouping semantics ==")
    e.reset()
    t0 = e._world.landmarks[0].state.p_pos
    t1 = e._world.landmarks[1].state.p_pos
    place(e, 0, t0 + np.array([0.01, 0.0]))
    place(e, 1, t0 + np.array([0.02, 0.0]))
    place(e, 2, t0 + np.array([0.03, 0.0]))
    place(e, 3, t1 + np.array([0.01, 0.0]))
    place(e, 4, [50.0, 50.0])
    place(e, 5, [-50.0, -50.0])
    adj = e.get_comm_mask()
    ok &= check("agents 0,1,2 mutually grouped",
                adj[0, 1] and adj[1, 2] and adj[0, 2])
    ok &= check("agent 3 not grouped with the target-0 clique",
                not adj[0, 3] and not adj[3, 0])
    ok &= check("distant agent 4 is a singleton", adj[4].sum() == 1,
                f"degree {adj[4].sum()}")
    ok &= check("diagonal on for every agent", np.all(np.diag(adj) == 1))
    ok &= check("adjacency symmetric", np.array_equal(adj, adj.T))

    print("\n== 3b. group_radius > capture_radius: grouped but cannot capture ==")
    e.reset()
    t0 = e._world.landmarks[0].state.p_pos
    mid = 0.5 * (e._scenario.capture_radius + e._scenario.group_radius)
    place(e, 0, t0 + np.array([0.01, 0.0]))          # inside capture radius
    place(e, 1, t0 + np.array([0.02, 0.0]))          # inside capture radius
    place(e, 2, t0 + np.array([mid, 0.0]))           # grouped, NOT capturable
    for i in (3, 4, 5):
        place(e, i, [50.0, 50.0])
    adj = e.get_comm_mask()
    ok &= check("agent 2 is grouped with 0 and 1", adj[0, 2] and adj[1, 2])
    ok &= check(
        "agent 2 cannot capture from there",
        not e._scenario._agents_in_range(e._world, 0)[2],
    )
    _, _, rewards, _, info, _ = e.step([CAPTURE, CAPTURE, CAPTURE, 0, 0, 0])
    ok &= check(
        "3 grouped but only 2 in range => undershoot, no capture",
        info[0]["joint_captures"] == 0 and info[0]["undershoots"] == 1,
        f"caps={info[0]['joint_captures']} unders={info[0]['undershoots']}",
    )

    print("\n== 3c. groups are non-trivial at reset ==")
    degs = []
    for _ in range(200):
        e.reset()
        degs.append(e.get_comm_mask().sum(axis=1).mean())
    mean_deg = float(np.mean(degs))
    ok &= check(
        "mean clique degree at reset > 1.0 (groups actually form)",
        mean_deg > 1.0,
        f"mean degree {mean_deg:.3f}",
    )

    print("\n== 4. obs_mode=shared exposes others ==")
    obs_shared = np.asarray(
        [e._scenario.observation(a, e._world) for a in e._world.agents],
        dtype=np.float32,
    )
    # agent 0 should see agent 4 at a large relative offset
    a0 = e._world.agents[0].state.p_pos
    a4 = e._world.agents[4].state.p_pos
    rel_expected = a4 - a0
    block = obs_shared[0][8 : 8 + 2 * (n - 1)].reshape(n - 1, 2)
    ok &= check(
        "agent 0's obs contains agent 4's relative position",
        np.any(np.all(np.isclose(block, rel_expected, atol=1e-5), axis=1)),
        f"looking for {rel_expected.round(2)} in\n{block.round(2)}",
    )
    tgt_block = obs_shared[0][8 + 2 * (n - 1):8 + 2 * (n - 1) + 3 * m].reshape(m, 3)
    ok &= check(
        "all three targets flagged active at reset-ish state",
        np.allclose(tgt_block[:, 2], 1.0),
        f"active flags {tgt_block[:, 2]}",
    )

    print("\n== 5. joint capture survives the wrapper ==")
    e.reset()
    t0 = e._world.landmarks[0].state.p_pos
    for i in range(3):
        place(e, i, t0 + np.array([0.01 * i, 0.0]))
    for i in (3, 4, 5):
        place(e, i, [50.0, 50.0])
    acts = [CAPTURE if i < 3 else 0 for i in range(n)]
    _, _, rewards, _, info, _ = e.step(acts)
    ok &= check("capture registered", info[0]["joint_captures"] == 1,
                f"got {info[0]['joint_captures']}")
    ok &= check("reward tiled to all agents",
                np.allclose(rewards, rewards[0]), f"{rewards.reshape(-1)}")
    ok &= check("team reward == capture_rew",
                np.isclose(rewards[0][0], 10.0), f"got {rewards[0][0]}")

    e.reset()
    t0 = e._world.landmarks[0].state.p_pos
    for i in range(2):
        place(e, i, t0 + np.array([0.01 * i, 0.0]))
    for i in (2, 3, 4, 5):
        place(e, i, [50.0, 50.0])
    acts = [CAPTURE if i < 2 else 0 for i in range(n)]
    _, _, rewards, _, info, _ = e.step(acts)
    ok &= check("undershoot registered", info[0]["undershoots"] == 1)
    ok &= check("team reward == fail_rew",
                np.isclose(rewards[0][0], -1.0), f"got {rewards[0][0]}")

    print("\n== 6. k_req > n_agents is refused ==")
    try:
        JointNavMATEnv(make_args(k_req=99))
        ok &= check("raises on impossible k_req", False)
    except ValueError:
        ok &= check("raises on impossible k_req", True)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
