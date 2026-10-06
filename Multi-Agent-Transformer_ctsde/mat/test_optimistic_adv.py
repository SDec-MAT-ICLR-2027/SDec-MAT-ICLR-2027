"""OptiMAPPO-style optimistic advantages: A <- max(leaky_rate * A, A).

  1. leaky_rate = 1  -> identity
  2. leaky_rate = 0  -> negatives become 0, positives unchanged
  3. leaky_rate = 0.3 -> negatives scaled by 0.3, positives unchanged
  4. the flags reach the trainer (on/off, rate) and a rate outside [0, 1]
     is refused

Run:  python -m mat.test_optimistic_adv
"""
import sys

import numpy as np

from mat.algorithms.mat.mat_trainer import MATTrainer
from mat.test_decoder_self_only import args as base_args, check


class _Pol:        # the trainer only stores the policy at construction
    pass


def main():
    ok = True
    rng = np.random.RandomState(0)
    A = rng.normal(0, 1, (50, 4, 6, 1)).astype(np.float32)
    f = MATTrainer.optimistic_transform
    ok &= check("rate 1: identity", np.array_equal(f(A, 1.0), A))
    z = f(A, 0.0)
    ok &= check("rate 0: negatives -> 0, positives unchanged",
                np.all(z[A < 0] == 0) and np.array_equal(z[A >= 0], A[A >= 0]))
    t = f(A, 0.3)
    ok &= check("rate 0.3: negatives x0.3, positives unchanged",
                np.allclose(t[A < 0], 0.3 * A[A < 0]) and np.array_equal(t[A >= 0], A[A >= 0]))

    a = base_args(0)
    a.optimistic_adv, a.opt_leaky_rate = 1, 0.3
    tr = MATTrainer(a, _Pol(), 6)
    ok &= check("flags reach the trainer", tr.optimistic_adv == 1 and tr.opt_leaky_rate == 0.3)
    a.optimistic_adv = 0
    ok &= check("off by default when flag is 0", MATTrainer(a, _Pol(), 6).optimistic_adv == 0)
    a.optimistic_adv, a.opt_leaky_rate = 1, 1.5
    try:
        MATTrainer(a, _Pol(), 6)
        ok &= check("rate outside [0, 1] refused", False)
    except ValueError:
        ok &= check("rate outside [0, 1] refused", True)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
