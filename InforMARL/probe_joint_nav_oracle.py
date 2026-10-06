"""
Is the joint-capture task well-posed?

Runs three scripted policies over a k_req sweep:

  oracle    : move to nearest active target; capture once >= k_req agents are
              in range. Needs the in-range COUNT, i.e. correlated information.
              Establishes the achievable ceiling.
  greedy_de : move to nearest active target; capture whenever in range.
              A purely local policy with no count. This is the behaviour a
              decentralized executor falls into, and it should eat the
              undershoot tax.
  assemble  : move to nearest active target and never capture. Used only to
              read off the assembly shaping, to confirm the dense term gives
              a climbable gradient toward grouping.

If `oracle` clears targets while `greedy_de` does not, the task separates
joint action from local action, which is the whole point of the benchmark.
"""
import argparse
import sys, os

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multiagent.MPE_env import GraphMPEEnv

CAPTURE_IDX = 5
N_ACT = 6


def make_args(**overrides):
    args = argparse.Namespace(
        scenario_name="navigation_joint_graph",
        world_size=2.0,
        num_agents=6,
        num_scripted_agents=0,
        num_obstacles=0,
        collaborative=True,
        max_speed=2.0,
        collision_rew=0.0,
        use_dones=False,
        episode_length=50,
        graph_feat_type="relative",
        max_edge_dist=1.0,
        min_dist_thresh=0.1,
        num_targets=3,
        k_req=3,
        capture_radius=0.3,
        capture_rew=10.0,
        fail_rew=-1.0,
        assembly_rew=0.05,
        dist_rew_scale=1.0,
        obs_ready_count=False,
        potential_shaping=1,
        shaping_gamma=0.99,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def onehot(idx):
    a = np.zeros(N_ACT, dtype=np.float32)
    a[idx] = 1.0
    return a


def move_toward(delta, vel, brake_frac=0.5):
    """
    Bang-bang discrete move along the dominant axis of `delta`.

    NOTE on action indices: the comment in `multiagent/environment.py` reads
    `[None, <-, ->, v, ^]`, but the arithmetic below it is
    `u[0] += action[1] - action[2]`, so index 1 is +x and index 2 is -x
    (likewise 3 is +y, 4 is -y). Trust the arithmetic, not the comment.
    """
    axis = 0 if abs(delta[0]) > abs(delta[1]) else 1
    # brake instead of accelerating when already closing fast on this axis,
    # otherwise bang-bang control overshoots straight through the target
    if abs(delta[axis]) < brake_frac * abs(vel[axis]):
        return 0
    if axis == 0:
        return 1 if delta[0] > 0 else 2
    return 3 if delta[1] > 0 else 4


def scripted_actions(env, scenario, mode):
    world = env.world
    acts = []
    for agent in world.agents:
        t_idx, t_dist = scenario._nearest_active_target(agent, world)
        if t_idx < 0:
            acts.append(onehot(0))
            continue
        in_range = scenario._agents_in_range(world, t_idx)
        n_near = int(in_range.sum())
        delta = world.landmarks[t_idx].state.p_pos - agent.state.p_pos
        vel = agent.state.p_vel

        if mode == "assemble":
            acts.append(onehot(0) if in_range[agent.id] else onehot(move_toward(delta, vel)))
        elif mode == "greedy_de":
            # local only: fire whenever in range, no idea who else is there
            acts.append(
                onehot(CAPTURE_IDX) if in_range[agent.id] else onehot(move_toward(delta, vel))
            )
        elif mode == "oracle":
            if in_range[agent.id] and n_near >= scenario.k_req:
                acts.append(onehot(CAPTURE_IDX))
            elif in_range[agent.id]:
                acts.append(onehot(0))  # wait for the group
            else:
                acts.append(onehot(move_toward(delta, vel)))
        else:
            raise ValueError(mode)
    return acts


def run(mode, k_req, n_eps=30, seed=0):
    args = make_args(k_req=k_req)
    env = GraphMPEEnv(args)
    scenario = env.reward_callback.__self__  # the Scenario instance
    np.random.seed(seed)
    caps, unders, rets, cleared = [], [], [], []
    for _ in range(n_eps):
        env.reset()
        ep_ret = 0.0
        for _ in range(args.episode_length):
            acts = scripted_actions(env, scenario, mode)
            _, _, _, _, reward_n, _, info_n = env.step(acts)
            ep_ret += float(reward_n[0][0])
        caps.append(info_n[0]["joint_captures"])
        unders.append(info_n[0]["undershoots"])
        cleared.append(float(info_n[0]["all_captured"]))
        rets.append(ep_ret)
    return (
        float(np.mean(caps)),
        float(np.mean(unders)),
        float(np.mean(cleared)),
        float(np.mean(rets)),
    )


def main():
    print(f"{'k_req':>6} {'policy':>10} {'captures':>9} {'undershoot':>11} "
          f"{'clear%':>7} {'return':>9}")
    print("-" * 58)
    for k in (1, 2, 3, 4):
        for mode in ("oracle", "greedy_de", "assemble"):
            c, u, cl, r = run(mode, k)
            print(f"{k:>6} {mode:>10} {c:>9.2f} {u:>11.2f} "
                  f"{100 * cl:>6.0f}% {r:>9.1f}")
        print("-" * 58)
    print(
        "\nWell-posed iff: oracle clears at every k, greedy_de degrades as k "
        "grows,\nand assemble shows the dense gradient without ever capturing."
    )


if __name__ == "__main__":
    main()
