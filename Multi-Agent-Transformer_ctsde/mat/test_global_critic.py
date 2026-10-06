"""
Centralized critic, local actor.

  1. shared mode: share_obs is exactly the concatenated obs (old runs unchanged)
  2. radius mode: share_obs carries targets NO agent can see (global state)
  3. mat_comm + critic_global_state: perturbing the global-state part of
     share_obs changes the VALUES but leaves every action log-prob exactly
     unchanged (no leak into execution), for CT-SDE and the DE control

Run:  python -m mat.test_global_critic
"""
import sys

import numpy as np
import torch

from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.envs.joint_nav import JointNavMATEnv
from mat.test_decoder_self_only import args as base_args, check


def env_args(obs_mode, algo="mat_comm", **kw):
    a = base_args(0)
    a.algorithm_name = algo
    a.obs_mode = obs_mode
    a.world_size, a.min_target_sep, a.sense_radius = 3.0, 1.25, 0.6
    a.critic_global_state = 1
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def main():
    ok = True
    print("== share_obs content ==")
    e = JointNavMATEnv(env_args("shared", "mappo"))
    e.seed(0)
    obs, share, _ = e.reset()
    ok &= check("shared mode: share_obs == concatenated obs",
                np.array_equal(share[0], obs.reshape(-1)))

    e = JointNavMATEnv(env_args("radius", "mappo"))
    hidden = 0
    for s in range(50):
        e.seed(s)
        obs, share, _ = e.reset()
        sc, w = e._scenario, e._world
        P = np.array([a.state.p_pos for a in w.agents])
        for t, lm in enumerate(w.landmarks[:sc.num_targets]):
            if np.linalg.norm(P - lm.state.p_pos, axis=1).min() > sc.sense_radius:
                hidden += 1
                # agent 0's full-view chunk: target t rel pos at 8+2(N-1)+3t
                off = 8 + 2 * 5 + 3 * t
                want = lm.state.p_pos - w.agents[0].state.p_pos
                ok &= bool(np.allclose(share[0, off:off + 2], want, atol=1e-5))
    ok &= check("radius mode: share_obs holds targets no agent sees",
                hidden > 0, "%d hidden targets checked" % hidden)

    for self_only, name in ((0, "CT-SDE"), (1, "DE control")):
        print("== %s: no leak of global state into actions ==" % name)
        torch.manual_seed(0)
        a = env_args("radius", decoder_self_only=self_only)
        env = JointNavMATEnv(a)
        pol = TransformerPolicy(a, env.observation_space[0],
                                env.share_observation_space[0],
                                env.action_space[0], 6)
        with torch.no_grad():
            for p in getattr(pol, "transformer", pol).parameters():
                if p.dim() >= 2:
                    p.normal_(0.0, 0.3)
        env.seed(3)
        obs, share, _ = env.reset()
        n = 6
        z = np.zeros((n, 1, 64), dtype=np.float32)
        m = np.ones((n, 1), dtype=np.float32)
        av = np.ones((n, 6), dtype=np.float32)
        acts = np.random.RandomState(0).randint(6, size=(n, 1))

        def ev(sh):
            with torch.no_grad():
                out = pol.evaluate_actions(sh, obs, z, z, acts, m, av)
            return out[0].numpy().reshape(-1), out[1].numpy().reshape(-1)

        v0, lp0 = ev(share)
        sh2 = share.copy()
        g = n * env._state_obs_dim
        sh2[:, :g] += np.random.RandomState(1).normal(0, 0.5, g).astype(np.float32)
        v1, lp1 = ev(sh2)
        ok &= check("values change with global state",
                    np.abs(v1 - v0).max() > 1e-4, "%.2e" % np.abs(v1 - v0).max())
        ok &= check("action log-probs unchanged", np.abs(lp1 - lp0).max() == 0.0,
                    "%.2e" % np.abs(lp1 - lp0).max())

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
