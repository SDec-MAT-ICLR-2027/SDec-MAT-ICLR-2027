"""Arm-consistent behaviour weights under the subset-commit mixture.

Behaviour: w.p. eps a group draws a shared action c ~ U(A); each member joins
w.p. q and takes c, the rest act from the policy given the executed prefix.
JointNavRunner.arm_behaviour_weights must match brute-force enumeration:
  1. the enumeration itself matches a Monte-Carlo simulation of the process
  2. AR actor: w_i = min(1, pi_i / mu(a_i | a_<i)) exactly, for random joints
  3. independent actor: w_i = min(1, pi_i / marginal mu_i) exactly
  4. q = 1, independent: equals the original behaviour_weights
  5. agents outside every group: w = 1

Run:  python -m mat.test_behav_w_arm
"""
import itertools
import sys

import numpy as np

from mat.runner.shared.joint_nav_runner import JointNavRunner as R

N, A, EPS, Q = 4, 3, 0.2, 0.5
G = [0, 1, 2, 3]


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(cond)


def make_policy(ar, rng):
    """pi(j, prefix) -> probs over A. AR: depends on the prefix; else not."""
    table = {}

    def pi(j, prefix):
        key = (j, tuple(prefix) if ar else ())
        if key not in table:
            table[key] = rng.dirichlet(np.ones(A) * 0.7)
        return table[key]
    return pi


def joint_prob(pi, a, eps=EPS, q=Q):
    """Exact behaviour probability of the full joint action a (all in G)."""
    no = np.prod([pi(j, a[:j])[a[j]] for j in range(N)])
    com = 0.0
    for c in range(A):
        for mem in itertools.product([0, 1], repeat=N):
            p = np.prod([q if m else 1 - q for m in mem])
            for j, m in enumerate(mem):
                p *= float(a[j] == c) if m else pi(j, a[:j])[a[j]]
            com += p / A
    return (1 - eps) * no + eps * com


def prefix_prob(pi, pre, eps=EPS, q=Q):
    k = len(pre)
    return sum(joint_prob(pi, list(pre) + list(rest), eps, q)
               for rest in itertools.product(range(A), repeat=N - k))


def simulate(pi, rng, n):
    counts = {}
    for _ in range(n):
        a = []
        commit = rng.rand() < EPS
        c = rng.randint(A)
        for j in range(N):
            if commit and rng.rand() < Q:
                a.append(c)
            else:
                a.append(rng.choice(A, p=pi(j, a)))
        counts[tuple(a)] = counts.get(tuple(a), 0) + 1
    return counts


def main():
    ok = True
    rng = np.random.RandomState(0)

    print("== 1. enumeration matches simulation ==")
    pi = make_policy(True, rng)
    n = 200000
    counts = simulate(pi, rng, n)
    worst = max(abs(counts.get(a, 0) / n - joint_prob(pi, list(a)))
                for a in itertools.product(range(A), repeat=N))
    tot = sum(joint_prob(pi, list(a)) for a in itertools.product(range(A), repeat=N))
    ok &= check("joint probs sum to 1", abs(tot - 1) < 1e-9, "%.12f" % tot)
    ok &= check("max |empirical - exact| < 0.004", worst < 0.004, "%.4f" % worst)

    print("\n== 2. AR actor: exact conditional ==")
    err = 0.0
    for _ in range(40):
        a = [int(x) for x in rng.randint(A, size=N)]
        if rng.rand() < 0.5:
            a = [a[0]] * N                      # committed-looking joints too
        logp = np.log([pi(j, a[:j])[a[j]] for j in range(N)])
        w = R.arm_behaviour_weights(logp, a, [G], EPS, Q, A, ar=True)
        for i in range(N):
            mu = prefix_prob(pi, a[:i + 1]) / prefix_prob(pi, a[:i])
            err = max(err, abs(w[i] - min(1.0, np.exp(logp[i]) / mu)))
    ok &= check("AR weights == brute force", err < 1e-9, "%.2e" % err)

    print("\n== 3. independent actor: exact marginal ==")
    pi_ind = make_policy(False, rng)
    err = 0.0
    for _ in range(40):
        a = [int(x) for x in rng.randint(A, size=N)]
        logp = np.log([pi_ind(j, [])[a[j]] for j in range(N)])
        w = R.arm_behaviour_weights(logp, a, [G], EPS, Q, A, ar=False)
        for i in range(N):
            mu = sum(joint_prob(pi_ind, list(b)) for b in
                     itertools.product(range(A), repeat=N) if b[i] == a[i])
            err = max(err, abs(w[i] - min(1.0, np.exp(logp[i]) / mu)))
    ok &= check("independent weights == brute-force marginal", err < 1e-9, "%.2e" % err)

    print("\n== 4. q = 1 reduces to the original formula ==")
    lp = np.log(rng.dirichlet(np.ones(6), 1).reshape(-1)[:4])
    w_new = R.arm_behaviour_weights(lp, [0, 1, 2, 3], [G], EPS, 1.0, 6, ar=False)
    w_old = R.behaviour_weights(lp, np.ones(4, dtype=bool), EPS, 6)
    ok &= check("equal to behaviour_weights", np.allclose(w_new, w_old), f"{w_new} {w_old}")

    print("\n== 5. agents outside groups ==")
    w = R.arm_behaviour_weights(np.log([0.01, 0.2, 0.3, 0.4]), [0, 1, 2, 3],
                                [[1, 2]], EPS, Q, A, ar=True)
    ok &= check("w = 1 for agents 0 and 3", w[0] == 1.0 and w[3] == 1.0, str(w))

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
