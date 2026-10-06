"""MAPPO vs CT-SDE in the radius env (E2): same partial observability?

  1. the env built for MAPPO and the env built for CT-SDE (same seed, same
     random actions) give IDENTICAL per-agent observations at every step;
     agents/targets outside R read as zero in both
  2. MAPPO has NO observation sharing: changing another agent's obs leaves
     agent i's action log-probs exactly unchanged (its actor reads its own
     radius obs only)
  3. CT-SDE DOES share within R: changing a within-R neighbour's obs moves
     agent i's action log-probs (and an out-of-R agent's obs does not)

Run:  python -m mat.test_mappo_same_obs
"""
import sys

import numpy as np
import torch

from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.envs.joint_nav import JointNavMATEnv
from mat.test_decoder_self_only import args as base_args, check

N, TB = 6, 4 + 3 * 5


def e2_args(algo):
    a = base_args(0)
    a.algorithm_name = algo
    a.obs_mode, a.comm_mode, a.sense_radius = "radius", "agent_radius", 0.6
    a.world_size, a.min_target_sep = 2.0, 1.25
    a.shuffle_targets, a.target_order = 1, "distance"
    if algo == "mappo":
        a.dec_actor, a.share_actor, a.encode_state = True, True, True
    else:
        a.critic_global_state = 1
    return a


def policy(algo, env, a):
    torch.manual_seed(0)
    pol = TransformerPolicy(a, env.observation_space[0], env.share_observation_space[0],
                            env.action_space[0], N)
    with torch.no_grad():
        for p in getattr(pol, "transformer", pol).parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.3)
    return pol


def logp(pol, share, obs, acts):
    z = np.zeros((N, 1, 64), dtype=np.float32)
    m = np.ones((N, 1), dtype=np.float32)
    av = np.ones((N, 6), dtype=np.float32)
    with torch.no_grad():
        out = pol.evaluate_actions(share, obs, z, z, acts, m, av)
    return out[1].numpy().reshape(-1)


def main():
    ok = True
    print("== 1. identical observations")
    am, ac = e2_args("mappo"), e2_args("mat_comm")
    em, ec = JointNavMATEnv(am), JointNavMATEnv(ac)
    ok &= check("same obs dim", em.observation_space[0] == ec.observation_space[0],
                "%s vs %s" % (em.observation_space[0], ec.observation_space[0]))
    # Both envs draw from the GLOBAL numpy RNG, so they must be rolled out one
    # after the other (not interleaved), with the same seed and actions.
    act_seq = np.random.RandomState(0).randint(6, size=(20, 50, N))

    def rollout(env):
        obs_all = []
        for s in range(20):
            env.seed(s)
            o, _, _ = env.reset()
            for t in range(50):
                obs_all.append(np.array(o))
                o, _, _, _, _, _ = env.step(act_seq[s, t])
        return obs_all

    obs_m, obs_c = rollout(em), rollout(ec)
    steps = len(obs_m)
    same = sum(int(np.array_equal(a, b)) for a, b in zip(obs_m, obs_c))
    zero_ok = 0
    for om in obs_m:
        hid_ok = True
        for i in range(N):
            for k in range(N - 1):                       # other agents
                if om[i, 4 + 3 * k + 2] == 0 and np.any(om[i, 4 + 3 * k:6 + 3 * k] != 0):
                    hid_ok = False
            for t in range(3):                           # targets
                r = om[i, TB + 4 * t:TB + 4 * t + 4]
                if r[3] == 0 and np.any(r != 0):
                    hid_ok = False
        zero_ok += int(hid_ok)
    ok &= check("MAPPO obs == CT-SDE obs at every step", same == steps, "%d/%d steps" % (same, steps))
    ok &= check("out-of-R agents/targets read as zero", zero_ok == steps, "%d/%d steps" % (zero_ok, steps))

    # a state with a within-R pair (i, j) and an agent k outside R of i
    for s in range(200):
        ec.seed(s)
        oc, shc, _ = ec.reset()
        adj = ec.get_comm_mask()
        pair = [(i, j) for i in range(N) for j in range(N) if i != j and adj[i, j]]
        if pair:
            i, j = pair[0]
            far = [k for k in range(N) if k != i and not adj[i, k]]
            if far:
                k = far[0]
                break
    em.seed(s)
    om, shm, _ = em.reset()
    acts = np.random.RandomState(1).randint(6, size=(N, 1))

    print("== 2. MAPPO: no observation sharing (agent %d, other agent %d within R)" % (i, j))
    pm = policy("mappo", em, am)
    base = logp(pm, shm, om, acts)
    om2 = om.copy()
    om2[j, 2] += 0.5
    d_other = np.abs(logp(pm, shm, om2, acts)[i] - base[i])
    om3 = om.copy()
    om3[i, 2] += 0.5
    d_own = np.abs(logp(pm, shm, om3, acts)[i] - base[i])
    ok &= check("changing a neighbour's obs leaves agent i unchanged", d_other == 0.0, "%.2e" % d_other)
    ok &= check("changing agent i's own obs moves it (sanity)", d_own > 1e-6, "%.2e" % d_own)

    print("== 3. CT-SDE: shares obs within R only")
    pc = policy("mat_comm", ec, ac)
    base = logp(pc, shc, oc, acts)
    oc2 = oc.copy()
    oc2[j, 2] += 0.5
    d_nb = np.abs(logp(pc, shc, oc2, acts)[i] - base[i])
    oc3 = oc.copy()
    oc3[k, 2] += 0.5
    d_far = np.abs(logp(pc, shc, oc3, acts)[i] - base[i])
    ok &= check("within-R neighbour's obs moves agent i", d_nb > 1e-6, "%.2e" % d_nb)
    ok &= check("out-of-R agent's obs does not", d_far == 0.0, "%.2e" % d_far)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
