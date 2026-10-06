"""Is capturing actually WORSE than camping under the current reward?

The dense navigation term is -dist_to_nearest_ACTIVE_target. Removing the
target you are standing on therefore makes your own next-step reward worse:
your nearest active target jumps from ~0 away to the next one, which is at
least min_target_sep away. This measures whether that penalty exceeds the
capture bonus.

Two scripted policies from an identical assembled state:
  camp     : all agents sit on target 0 forever, never capture
  capture  : all agents capture target 0 on step 1, then walk to the next
"""
import argparse
import sys, os

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multiagent.MPE_env import GraphMPEEnv

CAPTURE_IDX, N_ACT = 5, 6
POTENTIAL = 1
HORIZON = 50


def make_args(capture_rew):
    return argparse.Namespace(
        scenario_name="navigation_joint_graph", world_size=2.0,
        num_agents=6, num_scripted_agents=0, num_obstacles=0,
        collaborative=True, max_speed=2.0, collision_rew=0.0,
        use_dones=False, episode_length=HORIZON,
        graph_feat_type="relative", max_edge_dist=1.0, min_dist_thresh=0.1,
        num_targets=3, k_req=3, capture_radius=0.3, group_radius=0.6,
        capture_rew=capture_rew, fail_rew=-1.0,
        assembly_rew=0.05, dist_rew_scale=1.0, obs_mode="shared",
        obs_ready_count=False,
        potential_shaping=POTENTIAL, shaping_gamma=0.99,
    )


def onehot(i):
    a = np.zeros(N_ACT, dtype=np.float32)
    a[i] = 1.0
    return a


def move_toward(delta, vel, brake=0.5):
    ax = 0 if abs(delta[0]) > abs(delta[1]) else 1
    if abs(delta[ax]) < brake * abs(vel[ax]):
        return 0
    return (1 if delta[0] > 0 else 2) if ax == 0 else (3 if delta[1] > 0 else 4)


def run(mode, capture_rew, seed=0):
    a = make_args(capture_rew)
    e = GraphMPEEnv(a)
    sc = e.reward_callback.__self__
    np.random.seed(seed)
    e.reset()
    # place every agent on target 0, assembled and ready
    t0 = e.world.landmarks[0].state.p_pos
    for i, ag in enumerate(e.world.agents):
        ag.state.p_pos = t0 + np.array([0.01 * (i - 2.5), 0.0])
        ag.state.p_vel = np.zeros(2)

    total = 0.0
    for t in range(HORIZON):
        acts = []
        for ag in e.world.agents:
            ti, td = sc._nearest_active_target(ag, e.world)
            if ti < 0:
                acts.append(onehot(0)); continue
            in_cap = sc._agents_in_range(e.world, ti)[ag.id]
            if mode == "capture" and t == 0:
                acts.append(onehot(CAPTURE_IDX))
            elif in_cap:
                acts.append(onehot(0))
            else:
                d = e.world.landmarks[ti].state.p_pos - ag.state.p_pos
                acts.append(onehot(move_toward(d, ag.state.p_vel)))
        _, _, _, _, rn, _, info = e.step(acts)
        total += float(rn[0][0])
    return total, info[0]["joint_captures"]


def main():
    print(f"{'capture_rew':>12} {'camp return':>12} {'capture return':>15} "
          f"{'caps':>5} {'delta':>9}")
    print("-" * 60)
    global POTENTIAL
    for mode_name, pot in (("legacy per-step rate", 0), ("potential-based", 1)):
        POTENTIAL = pot
        print(f"  -- {mode_name} (potential_shaping={pot}) --")
        for cr in (10.0, 50.0, 200.0):
            c_ret, _ = run("camp", cr)
            k_ret, caps = run("capture", cr)
            print(f"{cr:>12.0f} {c_ret:>12.1f} {k_ret:>15.1f} {caps:>5.0f} "
                  f"{k_ret - c_ret:>+9.1f}")
    print("\ndelta < 0 means capturing the target you are standing on is")
    print("WORSE than camping on it, i.e. the shaping punishes task progress.")


if __name__ == "__main__":
    main()
