"""Pick a communication radius that does NOT leave agents fully connected.

With one radius R for sensing and communication, a large R collapses the
setting back into full observability and one big group -- the thing the
radius env is meant to avoid. This measures, per candidate R, how connected
the agent-to-agent graph actually is over whole episodes, under two policies:

  random     uniform actions -- agents stay spread out
  gathering  scripted: move to the nearest live target, hold in capture
             range, fire once >= k_req are there. Trained agents converge on
             targets like this, which is exactly when full connectivity
             would creep in, so this is the one that decides R.

Per R and policy:
  mean degree              neighbours per agent (excluding itself)
  P(fully connected)       share of steps where all N agents form ONE
                           connected component (anyone can reach anyone,
                           possibly via relays)
  P(complete)              share of steps where every pair is directly
                           within R (one giant 1-hop group)
  largest component        mean size of the biggest connected component
  isolated                 share of agent-steps with no neighbour
  READY connected          share of READY events (>= k_req at a target)
                           whose agents are all mutually within R. Must be
                           1.0 or the joint capture is not coordinable.

Also runs basic checks on the radius observation.

  python probe_radius_connectivity.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multiagent.MPE_env import GraphMPEEnv

CAPTURE, N_ACT = 5, 6
# (world_size, R, min_target_sep). sep > R + 2*capture_radius guarantees two
# groups sitting on different targets can never see each other.
CONFIGS = (
    (2.0, 0.6, 0.6),
    (2.0, 0.6, 1.25),
    (3.0, 0.6, 0.6),
    (3.0, 0.6, 1.25),
    (3.0, 0.8, 1.45),
)
N_EPS = 40
T = 50


def make_args(R):
    return argparse.Namespace(
        scenario_name="navigation_joint_graph", world_size=2.0, num_agents=6,
        num_scripted_agents=0, num_obstacles=0, collaborative=True,
        max_speed=2.0, collision_rew=0.0, use_dones=False, episode_length=T,
        graph_feat_type="relative", max_edge_dist=1.0, min_dist_thresh=0.1,
        num_targets=3, k_req=3, capture_radius=0.3, group_radius=0.3,
        capture_rew=10.0, fail_rew=-1.0, assembly_rew=0.05,
        dist_rew_scale=1.0, potential_shaping=1, shaping_gamma=1.0,
        obs_ready_count=False, obs_mode="radius", sense_radius=R,
        comm_mode="agent_radius",
    )


def onehot(i):
    a = np.zeros(N_ACT, dtype=np.float32)
    a[i] = 1.0
    return a


def move_toward(delta, vel, brake=0.5):
    # index 1 is +x, 2 is -x, 3 is +y, 4 is -y (trust the arithmetic in
    # environment.py, not its comment)
    ax = 0 if abs(delta[0]) > abs(delta[1]) else 1
    if abs(delta[ax]) < brake * abs(vel[ax]):
        return 0
    return (1 if delta[0] > 0 else 2) if ax == 0 else (3 if delta[1] > 0 else 4)


def gathering_actions(env, sc):
    w = env.world
    acts = []
    for ag in w.agents:
        ti, _ = sc._nearest_active_target(ag, w)
        if ti < 0:
            acts.append(onehot(0))
            continue
        inr = sc._agents_in_range(w, ti)
        if inr[ag.id] and inr.sum() >= sc.k_req:
            acts.append(onehot(CAPTURE))
        elif inr[ag.id]:
            acts.append(onehot(0))
        else:
            d = w.landmarks[ti].state.p_pos - ag.state.p_pos
            acts.append(onehot(move_toward(d, ag.state.p_vel)))
    return acts


def split_actions(env, sc):
    """Realistic trained-like play: agents 0-2 work target 0, agents 3-5
    target 1 (k_req=3 -> two groups of 3), then any live target."""
    w = env.world
    acts = []
    for ag in w.agents:
        pref = ag.id // 3
        if w.target_active[pref]:
            ti = pref
        else:
            ti, _ = sc._nearest_active_target(ag, w)
        if ti < 0:
            acts.append(onehot(0))
            continue
        inr = sc._agents_in_range(w, ti)
        if inr[ag.id] and inr.sum() >= sc.k_req:
            acts.append(onehot(CAPTURE))
        elif inr[ag.id]:
            acts.append(onehot(0))
        else:
            d = w.landmarks[ti].state.p_pos - ag.state.p_pos
            acts.append(onehot(move_toward(d, ag.state.p_vel)))
    return acts


def components(adj):
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    sizes = []
    for s in range(n):
        if seen[s]:
            continue
        stack, c = [s], 0
        seen[s] = True
        while stack:
            u = stack.pop()
            c += 1
            for v in np.nonzero(adj[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        sizes.append(c)
    return sizes


def checks():
    ok = True
    a = make_args(0.6)
    env = GraphMPEEnv(a)
    sc = env.reward_callback.__self__
    obs, _, _, _ = env.reset()
    want = 4 + 3 * 5 + 4 * 3 + 1
    ok &= len(obs[0]) == want
    print("  [%s] radius obs dim %d (want %d)" % ("PASS" if len(obs[0]) == want else "FAIL",
                                                  len(obs[0]), want))
    adj = sc.agent_radius_adjacency(env.world)
    ok &= bool((adj == adj.T).all() and (np.diag(adj) == 1).all())
    print("  [%s] adjacency symmetric, diagonal on" % ("PASS" if ok else "FAIL"))
    # visible flag in agent 0's obs must equal adjacency row 0 (others, id order)
    o = obs[0]
    vis = [o[4 + 3 * k + 2] for k in range(5)]
    others = [j for j in range(6) if j != 0]
    match = all(int(vis[k]) == int(adj[0, j]) for k, j in enumerate(others))
    ok &= match
    print("  [%s] visible flags == adjacency (sensing radius = comm radius)"
          % ("PASS" if match else "FAIL"))
    hidden_zero = all(
        (o[4 + 3 * k:4 + 3 * k + 2] == 0).all() for k in range(5) if vis[k] == 0)
    ok &= hidden_zero
    print("  [%s] hidden agents read as zeros" % ("PASS" if hidden_zero else "FAIL"))
    try:
        make_args_bad = make_args(0.5)
        GraphMPEEnv(make_args_bad)
        print("  [FAIL] R < 2*capture_radius should be refused")
        ok = False
    except ValueError:
        print("  [PASS] R < 2*capture_radius is refused")
    return ok


def main():
    print("== checks ==")
    ok = checks()
    print("\n== connectivity over %d episodes x %d steps, 6 agents, 3 targets, "
          "2x2 board ==" % (N_EPS, T))
    print("  P(full|2+) = P(fully connected) restricted to steps with >= 2 "
          "live targets (excludes the endgame where all converge on the last)")
    print("%4s %4s %5s %-10s %7s %9s %10s %9s %9s %8s %9s %6s" % (
        "ws", "R", "sep", "policy", "degree", "P(fully)", "P(full|2+)",
        "P(compl)", "largest", "isolated", "READYconn", "caps"))
    print("-" * 104)
    for ws, R, sep in CONFIGS:
        for pol in ("random", "gathering", "split"):
            a = make_args(R)
            a.world_size = ws
            a.min_target_sep = sep
            env = GraphMPEEnv(a)
            sc = env.reward_callback.__self__
            rng = np.random.RandomState(0)
            np.random.seed(0)
            deg, fully, compl, largest, iso, full2 = [], [], [], [], [], []
            ready_n = ready_c = 0
            caps = []
            for _ in range(N_EPS):
                env.reset()
                for _t in range(T):
                    adj = sc.agent_radius_adjacency(env.world)
                    d = adj.sum(1) - 1
                    deg.append(d.mean())
                    iso.append((d == 0).mean())
                    sz = components(adj)
                    fully.append(float(max(sz) == 6))
                    if int(np.sum(env.world.target_active)) >= 2:
                        full2.append(float(max(sz) == 6))
                    compl.append(float(adj.all()))
                    largest.append(max(sz))
                    for tg in range(sc.num_targets):
                        if not env.world.target_active[tg]:
                            continue
                        inr = np.nonzero(sc._agents_in_range(env.world, tg))[0]
                        if len(inr) >= sc.k_req:
                            ready_n += 1
                            ready_c += int(all(adj[i, j] for i in inr for j in inr))
                    if pol == "random":
                        acts = [onehot(rng.randint(N_ACT)) for _ in range(6)]
                    elif pol == "gathering":
                        acts = gathering_actions(env, sc)
                    else:
                        acts = split_actions(env, sc)
                    _, _, _, _, _, _, info = env.step(acts)
                caps.append(info[0]["joint_captures"])
            print("%4.1f %4.1f %5.2f %-10s %7.2f %9.3f %10.3f %9.3f %9.2f %8.3f %9s %6.2f" % (
                ws, R, sep, pol, np.mean(deg), np.mean(fully),
                np.mean(full2) if full2 else float("nan"), np.mean(compl),
                np.mean(largest), np.mean(iso),
                ("%.3f" % (ready_c / ready_n)) if ready_n else "  -",
                np.mean(caps)))
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    print("Pick R where P(fully) stays low under GATHERING while READY conn "
          "stays 1.0.")


if __name__ == "__main__":
    main()
