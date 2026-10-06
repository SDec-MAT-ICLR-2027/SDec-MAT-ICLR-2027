"""MAPPO arm with a global-state critic (encode_state=1), radius env.

  1. the critic depends on the global state: perturbing share_obs changes
     the values (vanilla MAT zeroed the state, so it never could)
  2. no leak: every action log-prob is exactly unchanged (the dec_actor MLP
     reads only the agent's own radius obs)
  3. the actor does depend on the agent's own obs (sanity)
  4. encode_state=0 keeps the vanilla behaviour: values ignore share_obs

Run:  python -m mat.test_mappo_global_critic
"""
import sys

import numpy as np
import torch

from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.envs.joint_nav import JointNavMATEnv
from mat.test_decoder_self_only import args as base_args, check

N = 6


def build(encode_state):
    torch.manual_seed(0)
    a = base_args(0)
    a.algorithm_name = "mappo"
    a.dec_actor, a.share_actor = True, True
    a.encode_state = encode_state
    a.world_size, a.min_target_sep, a.sense_radius = 2.0, 1.25, 0.6
    env = JointNavMATEnv(a)
    pol = TransformerPolicy(a, env.observation_space[0],
                            env.share_observation_space[0],
                            env.action_space[0], N)
    with torch.no_grad():
        for p in getattr(pol, "transformer", pol).parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.3)
    env.seed(3)
    obs, share, _ = env.reset()
    return pol, obs, share


def ev(pol, share, obs, acts):
    z = np.zeros((N, 1, 64), dtype=np.float32)
    m = np.ones((N, 1), dtype=np.float32)
    av = np.ones((N, 6), dtype=np.float32)
    with torch.no_grad():
        out = pol.evaluate_actions(share, obs, z, z, acts, m, av)
    return out[0].cpu().numpy().reshape(-1), out[1].cpu().numpy().reshape(-1)


def main():
    ok = True
    acts = np.random.RandomState(0).randint(6, size=(N, 1))
    rs = np.random.RandomState(1)

    pol, obs, share = build(1)
    v0, lp0 = ev(pol, share, obs, acts)
    sh2 = share + rs.normal(0, 0.5, share.shape).astype(np.float32)
    v1, lp1 = ev(pol, sh2, obs, acts)
    print("== encode_state=1 (global-state critic)")
    ok &= check("values change with the global state",
                np.abs(v1 - v0).max() > 1e-4, "%.2e" % np.abs(v1 - v0).max())
    ok &= check("action log-probs unchanged (no leak)",
                np.abs(lp1 - lp0).max() == 0.0, "%.2e" % np.abs(lp1 - lp0).max())
    ob2 = obs.copy()
    ob2[:, 2] += 0.5
    _, lp2 = ev(pol, share, ob2, acts)
    ok &= check("actor depends on its own obs", np.abs(lp2 - lp0).max() > 1e-6,
                "%.2e" % np.abs(lp2 - lp0).max())

    pol, obs, share = build(0)
    v0, _ = ev(pol, share, obs, acts)
    v1, _ = ev(pol, share + 1.0, obs, acts)
    print("== encode_state=0 (vanilla)")
    ok &= check("values ignore share_obs (state zeroed, as before)",
                np.abs(v1 - v0).max() == 0.0, "%.2e" % np.abs(v1 - v0).max())

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
