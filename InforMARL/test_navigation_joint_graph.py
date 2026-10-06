"""
Smoke test for the joint-capture navigation scenario.

Checks the parts that are easy to get silently wrong:
  1. action space gained exactly one CAPTURE action
  2. a target falls iff >= k_req in-range agents capture on the SAME step
  3. fewer than k_req capturers is taxed, not ignored
  4. out-of-range capture attempts do not count
  5. the per-agent rewards SUM to the intended team reward
  6. a random policy does not trivially solve the task

Run:  python test_navigation_joint_graph.py
"""
import argparse
import sys, os

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multiagent.MPE_env import GraphMPEEnv

N_MOVE = 5  # no-op, -x, +x, -y, +y
CAPTURE_IDX = 5


def make_args(**overrides):
    args = argparse.Namespace(
        scenario_name="navigation_joint_graph",
        world_size=2.0,
        num_agents=4,
        num_scripted_agents=0,
        num_obstacles=0,
        collaborative=True,
        max_speed=2.0,
        collision_rew=0.0,
        use_dones=False,
        episode_length=25,
        graph_feat_type="relative",
        max_edge_dist=1.0,
        min_dist_thresh=0.1,
        # joint-capture params
        num_targets=2,
        k_req=3,
        capture_radius=0.3,
        capture_rew=10.0,
        fail_rew=-1.0,
        assembly_rew=0.0,  # off, so reward arithmetic is exact in the tests
        dist_rew_scale=0.0,  # off, so reward arithmetic is exact in the tests
        obs_ready_count=False,
        potential_shaping=1,
        shaping_gamma=0.99,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def onehot(idx, n):
    a = np.zeros(n, dtype=np.float32)
    a[idx] = 1.0
    return a


def place(env, agent_idx, pos):
    env.world.agents[agent_idx].state.p_pos = np.array(pos, dtype=np.float64)
    env.world.agents[agent_idx].state.p_vel = np.zeros(2)


def step_with(env, capture_agents):
    """Step where `capture_agents` emit CAPTURE and everyone else no-ops."""
    n_act = env.action_space[0].n
    acts = [
        onehot(CAPTURE_IDX if i in capture_agents else 0, n_act)
        for i in range(env.n)
    ]
    return env.step(acts)


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(cond)


def main():
    ok = True
    args = make_args()
    env = GraphMPEEnv(args)

    print("== 1. spaces ==")
    n_act = env.action_space[0].n
    ok &= check("action space is Discrete(6)", n_act == N_MOVE + 1, f"got {n_act}")
    obs, agent_id, node_obs, adj = env.reset()
    ok &= check("reset returns one obs per agent", len(obs) == args.num_agents)
    ok &= check("obs dim is 8", len(obs[0]) == 8, f"got {len(obs[0])}")
    ok &= check(
        "node feature has the extra `active` flag",
        node_obs[0].shape[1] == 8,
        f"got {node_obs[0].shape}",
    )

    args_rc = make_args(obs_ready_count=True)
    env_rc = GraphMPEEnv(args_rc)
    o_rc, _, _, _ = env_rc.reset()
    ok &= check(
        "obs_ready_count adds exactly one feature",
        len(o_rc[0]) == 9,
        f"got {len(o_rc[0])}",
    )

    print("\n== 1b. targets never share a capture radius ==")
    worst = np.inf
    for _ in range(200):
        env.reset()
        pos = [lm.state.p_pos for lm in env.world.landmarks]
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                worst = min(worst, np.sqrt(np.sum(np.square(pos[i] - pos[j]))))
    ok &= check(
        "min target separation >= 2*capture_radius over 200 resets",
        worst >= 2 * args.capture_radius - 1e-9,
        f"worst gap {worst:.3f}, need {2 * args.capture_radius:.3f}",
    )

    print("\n== 2. exactly k_req in-range capturers fells a target ==")
    env.reset()
    tgt = env.world.landmarks[0].state.p_pos
    for i in range(3):
        place(env, i, tgt + np.array([0.01 * i, 0.0]))
    place(env, 3, tgt + np.array([5.0, 5.0]))  # far away
    before = int(env.world.target_active.sum())
    _, _, _, _, reward_n, _, info_n = step_with(env, {0, 1, 2})
    after = int(env.world.target_active.sum())
    ok &= check("target captured", after == before - 1, f"{before} -> {after}")
    ok &= check(
        "joint_captures incremented", info_n[0]["joint_captures"] == 1,
        f"got {info_n[0]['joint_captures']}",
    )
    ok &= check(
        "team reward == capture_rew",
        np.isclose(reward_n[0][0], args.capture_rew),
        f"got {reward_n[0][0]}",
    )

    print("\n== 3. k_req-1 capturers is an undershoot, and is taxed ==")
    env.reset()
    tgt = env.world.landmarks[0].state.p_pos
    for i in range(2):
        place(env, i, tgt + np.array([0.01 * i, 0.0]))
    for i in (2, 3):
        place(env, i, tgt + np.array([5.0, 5.0]))
    before = int(env.world.target_active.sum())
    _, _, _, _, reward_n, _, info_n = step_with(env, {0, 1})
    ok &= check(
        "target NOT captured",
        int(env.world.target_active.sum()) == before,
        f"{before} -> {int(env.world.target_active.sum())}",
    )
    ok &= check("undershoot recorded", info_n[0]["undershoots"] == 1)
    ok &= check(
        "team reward == fail_rew",
        np.isclose(reward_n[0][0], args.fail_rew),
        f"got {reward_n[0][0]}",
    )

    print("\n== 4. capture from out of range does not count ==")
    env.reset()
    tgt = env.world.landmarks[0].state.p_pos
    place(env, 0, tgt + np.array([0.01, 0.0]))
    place(env, 1, tgt + np.array([0.02, 0.0]))
    for i in (2, 3):
        place(env, i, tgt + np.array([5.0, 5.0]))
    before = int(env.world.target_active.sum())
    _, _, _, _, reward_n, _, info_n = step_with(env, {0, 1, 2, 3})
    ok &= check(
        "far capturers ignored, target survives",
        int(env.world.target_active.sum()) == before,
    )
    ok &= check(
        "still only an undershoot",
        info_n[0]["undershoots"] == 1,
        f"got {info_n[0]['undershoots']}",
    )

    print("\n== 5. no capture action => no event reward ==")
    env.reset()
    tgt = env.world.landmarks[0].state.p_pos
    for i in range(4):
        place(env, i, tgt + np.array([0.01 * i, 0.0]))
    _, _, _, _, reward_n, _, info_n = step_with(env, set())
    ok &= check(
        "assembled but silent => 0 reward",
        np.isclose(reward_n[0][0], 0.0),
        f"got {reward_n[0][0]}",
    )
    ok &= check("no undershoot logged", info_n[0]["undershoots"] == 0)

    print("\n== 6. k_req sweep under a random policy ==")
    for k in (1, 2, 3, 4):
        a = make_args(k_req=k, num_targets=2, dist_rew_scale=0.0, assembly_rew=0.0)
        e = GraphMPEEnv(a)
        rng = np.random.RandomState(0)
        caps, unders = [], []
        for _ in range(30):
            e.reset()
            for _ in range(a.episode_length):
                acts = [onehot(rng.randint(n_act), n_act) for _ in range(e.n)]
                _, _, _, _, _, _, info_n = e.step(acts)
            caps.append(info_n[0]["joint_captures"])
            unders.append(info_n[0]["undershoots"])
        print(
            f"    k_req={k}: random-policy captures/ep = {np.mean(caps):.2f}, "
            f"undershoots/ep = {np.mean(unders):.2f}"
        )
        if k >= 3:
            ok &= check(
                f"k_req={k} is not trivially solved at random",
                np.mean(caps) < 0.5,
                f"got {np.mean(caps):.2f}",
            )

    print("")
    print("== 6b. potential shaping: camping cannot be farmed ==")
    # Regression guard for the camping pathology. Under the legacy per-step
    # rate, capturing the target you stand on was worth -37 vs camping over an
    # episode, so both arms learned never to capture. Potential shaping
    # telescopes to a path-independent constant, so over a full episode the
    # ONLY thing separating capture from camp is the true task reward.
    # This is an episode-level invariant, not a per-step one.
    def episode_return(mode, capture_rew, potential, horizon=25):
        a = make_args(k_req=3, num_agents=6, num_targets=3,
                      assembly_rew=0.05, dist_rew_scale=1.0,
                      capture_rew=capture_rew, potential_shaping=potential,
                      shaping_gamma=0.99, episode_length=horizon)
        e = GraphMPEEnv(a)
        sc = e.reward_callback.__self__
        np.random.seed(0)
        e.reset()
        t0 = e.world.landmarks[0].state.p_pos
        # ring of radius 0.15: inside capture_radius=0.3, but far enough apart
        # that MPE collision repulsion does not scatter them mid-test
        for i, ag in enumerate(e.world.agents):
            th = 2.0 * np.pi * i / len(e.world.agents)
            ag.state.p_pos = t0 + 0.15 * np.array([np.cos(th), np.sin(th)])
            ag.state.p_vel = np.zeros(2)
        # entities were teleported after reset, so Phi(s0) must be recomputed
        # or the first step banks a spurious one-time potential jump
        e.world._prev_potential = sc.potential(e.world)
        n6 = e.action_space[0].n
        total = 0.0
        for t in range(horizon):
            if mode == "capture" and t == 0:
                acts = [onehot(CAPTURE_IDX, n6) if i < 3 else onehot(0, n6)
                        for i in range(6)]
            else:
                acts = [onehot(0, n6) for _ in range(6)]
            _, _, _, _, rn, _, _ = e.step(acts)
            total += float(rn[0][0])
        return total

    # The regression guard: under potential shaping capturing the target you
    # stand on must BEAT camping on it, and the advantage must not decay as
    # the horizon grows. Under the legacy per-step rate it did the opposite
    # (capture was worth -37 vs camp), which is what taught both arms never
    # to capture. The scripted agents here never walk on to the next target,
    # so the gap sits a bounded amount below capture_rew; a real policy that
    # moves on recovers the rest.
    for cr in (10.0, 50.0):
        d25 = episode_return("capture", cr, 1, 25) - episode_return("camp", cr, 1, 25)
        d50 = episode_return("capture", cr, 1, 50) - episode_return("camp", cr, 1, 50)
        ok &= check(
            f"potential: capturing beats camping at capture_rew={cr:g}",
            d25 > 0 and d50 > 0,
            f"delta@25={d25:+.2f}, delta@50={d50:+.2f}",
        )
        ok &= check(
            f"potential: advantage does not decay with horizon ({cr:g})",
            d50 >= d25 - 0.5,
            f"delta@25={d25:+.2f}, delta@50={d50:+.2f}",
        )

    l_camp = episode_return("camp", 10.0, 0)
    l_cap = episode_return("capture", 10.0, 0)
    ok &= check(
        "legacy path still makes capturing WORSE than camping (pathology kept)",
        (l_cap - l_camp) < 0,
        f"camp {l_camp:.2f}, capture {l_cap:.2f}, delta {l_cap - l_camp:+.2f}",
    )


    print("\n== 7. stock navigation_graph is unchanged ==")
    a = make_args(scenario_name="navigation_graph", num_agents=3)
    a.goal_rew = 5.0
    a.num_obstacles = 0
    e = GraphMPEEnv(a)
    ok &= check(
        "stock scenario still has Discrete(5)",
        e.action_space[0].n == N_MOVE,
        f"got {e.action_space[0].n}",
    )

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
