"""Decoupled-PPO behaviour weight w = min(1, pi / mu).

mu = (1-eps)*pi + eps/|A| for agents eligible for a group commitment, pi
otherwise. Checks:
  1. non-eligible agents: w == 1 exactly (plain PPO untouched)
  2. eligible, likely action (pi=0.9): w == 1 (on-policy samples keep weight)
  3. eligible, forced action the policy almost never takes (pi=0.001):
     w ~ 0.03 (forced samples barely count)
  4. eligible, pi = 0.005 (a forced fire at a READY target in the audit):
     w ~ 0.13
  5. w in (0, 1] everywhere and never below the floor pi*|A|/eps*(1-..)

Run:  python -m mat.test_offpolicy_weight
"""
import sys

import numpy as np

from mat.runner.shared.joint_nav_runner import JointNavRunner as R


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(cond)


def main():
    ok = True
    eps, n_act = 0.2, 6
    lp = np.log(np.array([0.9, 0.001, 0.005, 0.3]))
    w_no = R.behaviour_weights(lp, np.zeros(4, dtype=bool), eps, n_act)
    ok &= check("non-eligible: w == 1", np.all(w_no == 1.0), str(w_no))
    w = R.behaviour_weights(lp, np.ones(4, dtype=bool), eps, n_act)
    ok &= check("eligible, pi=0.9: w == 1", w[0] == 1.0, "%.3f" % w[0])
    ok &= check("eligible, pi=0.001: w ~ 0.03", 0.02 < w[1] < 0.04, "%.4f" % w[1])
    ok &= check("eligible, pi=0.005: w ~ 0.13", 0.10 < w[2] < 0.16, "%.4f" % w[2])
    rnd = np.log(np.random.RandomState(0).dirichlet(np.ones(n_act), 1000).reshape(-1))
    wr = R.behaviour_weights(rnd, np.ones_like(rnd, dtype=bool), eps, n_act)
    ok &= check("w in (0, 1] for random policies", np.all((wr > 0) & (wr <= 1)))
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
