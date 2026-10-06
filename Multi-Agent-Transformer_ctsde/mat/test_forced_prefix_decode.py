"""Forced-prefix decoding (subset commit, autoregressive actor).

  1. forced positions keep exactly their forced actions
  2. an all-False mask reproduces the plain sampler (same torch seed)
  3. with every agent but j forced, j's sampled action frequencies match
     the teacher-forced pi(. | forced actions)   (CT-SDE and DE control)

Run:  python -m mat.test_forced_prefix_decode
"""
import sys

import numpy as np
import torch

from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.envs.joint_nav import JointNavMATEnv
from mat.test_decoder_self_only import args, find_state

N, NA, B = 6, 6, 4000


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(cond)


def build(self_only):
    torch.manual_seed(0)
    a = args(self_only)
    env = JointNavMATEnv(a)
    pol = TransformerPolicy(a, env.observation_space[0], env.share_observation_space[0],
                            env.action_space[0], N)
    with torch.no_grad():
        for p in pol.transformer.parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.3)
    obs, share, i, j, _ = find_state(env)
    return pol, obs, share, i, j


def act(pol, obs, share, b, **kw):
    z = np.zeros((b * N, 1, 64), dtype=np.float32)
    m = np.ones((b * N, 1), dtype=np.float32)
    av = np.ones((b * N, NA), dtype=np.float32)
    with torch.no_grad():
        out = pol.get_actions(np.tile(share, (b, 1)), np.tile(obs, (b, 1)), z, z, m, av, **kw)
    return out[1].cpu().numpy().reshape(b, N)


def probs_j(pol, obs, share, acts, j):
    z = np.zeros((N, 1, 64), dtype=np.float32)
    m = np.ones((N, 1), dtype=np.float32)
    av = np.ones((N, NA), dtype=np.float32)
    p = []
    for a in range(NA):
        x = acts.copy()
        x[j] = a
        with torch.no_grad():
            out = pol.evaluate_actions(share, obs, z, z, x.reshape(-1, 1), m, av)
        p.append(float(np.exp(out[1].reshape(-1)[j])))
    return np.array(p)


def main():
    ok = True
    for self_only, name in ((0, "CT-SDE"), (1, "DE control")):
        print("== %s ==" % name)
        pol, obs, share, i, j = build(self_only)
        rng = np.random.RandomState(1)
        fa = rng.randint(NA, size=(64, N))
        fm = rng.rand(64, N) < 0.5
        out = act(pol, obs, share, 64, forced_actions=fa, forced_mask=fm)
        ok &= check("forced positions keep their actions", np.array_equal(out[fm], fa[fm]))

        torch.manual_seed(5)
        plain = act(pol, obs, share, 64)
        torch.manual_seed(5)
        none = act(pol, obs, share, 64, forced_actions=np.zeros((64, N), dtype=np.int64),
                   forced_mask=np.zeros((64, N), dtype=bool))
        ok &= check("all-False mask == plain sampler", np.array_equal(plain, none))

        forced = np.full(N, 5, dtype=np.int64)
        forced[i] = 5
        mask = np.ones((B, N), dtype=bool)
        mask[:, j] = False
        out = act(pol, obs, share, B, forced_actions=np.tile(forced, (B, 1)), forced_mask=mask)
        emp = np.bincount(out[:, j], minlength=NA) / B
        ref = probs_j(pol, obs, share, forced, j)
        err = float(np.abs(emp - ref).max())
        ok &= check("agent %d samples from pi(.|forced prefix)" % j, err < 0.03,
                    "max diff %.3f" % err)
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
