"""
Verify the switches ported into InforMARL behave exactly like the MAT fork's.

Run:  python test_ctsde_switches.py
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from onpolicy.utils.ctsde_switches import (
    radius_comm_groups, random_group_commit, arm_behaviour_weights,
    optimistic_transform,
)

FAILED = []


def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  " + extra) if extra else ""))
    if not cond:
        FAILED.append(name)


# ---------------------------------------------------------------- weights
def test_marginal_matches_montecarlo():
    """The closed-form marginal must equal the empirical behaviour frequency
    of the subset-commit mixture."""
    rng = np.random.RandomState(0)
    n_act, eps, q = 6, 0.2, 0.5
    pi = rng.dirichlet(np.ones(n_act) * 2.0)
    N = 400000
    counts = np.zeros(n_act)
    for _ in range(N):
        if rng.rand() < eps:
            c = rng.randint(n_act)
            a = c if rng.rand() < q else rng.choice(n_act, p=pi)
        else:
            a = rng.choice(n_act, p=pi)
        counts[a] += 1
    emp = counts / N
    closed = (1.0 - eps * q) * pi + eps * q / n_act
    err = np.abs(emp - closed).max()
    check("marginal mu matches Monte-Carlo", err < 0.004, "max err %.5f" % err)


def test_weight_formula():
    """w = min(1, pi/mu) with mu the marginal; agents outside a group get 1."""
    n_act, eps, q = 6, 0.2, 0.5
    logp = np.log(np.array([0.5, 0.1, 0.02, 0.7]))
    acts = np.array([0, 1, 2, 3])
    groups = [[0, 1, 2]]           # agent 3 is a singleton -> untouched
    w = arm_behaviour_weights(logp, acts, groups, eps, q, n_act, ar=False)
    pi = np.exp(logp)
    mu = (1.0 - eps * q) * pi + eps * q / n_act
    want = np.minimum(1.0, pi / mu)
    check("grouped agents use the marginal", np.allclose(w[:3], want[:3]))
    check("ungrouped agent has w = 1", w[3] == 1.0)
    check("w is always <= 1", w.max() <= 1.0)
    # A rare action is down-weighted, but only to pi/((1-eps*q)pi + eps*q/|A|).
    # For pi=0.02 that is 0.577, NOT near zero: the uniform commit branch has
    # mass eps*q/|A| = 0.0167, comparable to pi itself.
    check("rare action is down-weighted", w[2] < 0.6, "w=%.4f" % w[2])
    check("down-weighting is monotone in pi", w[2] < w[0] <= 1.0,
          "w=%.3f,%.3f" % (w[2], w[0]))


def test_q1_reduces_to_old_marginal():
    """At q=1 the arm formula must equal the older marginal weighting, so a
    run matching a q=1 CT-SDE run is weighted identically."""
    n_act, eps = 6, 0.2
    pi = np.array([0.5, 0.1, 0.02, 0.3])
    logp = np.log(pi)
    w = arm_behaviour_weights(logp, np.arange(4), [[0, 1, 2, 3]], eps, 1.0,
                              n_act, ar=False)
    old_mu = (1.0 - eps) * pi + eps / n_act
    check("q=1 equals the old marginal", np.allclose(w, np.minimum(1, pi / old_mu)))


# ---------------------------------------------------------------- grouping
def test_grouping():
    """Groups come from the radius obs visibility flags at 4 + 3k + 2."""
    n, D = 4, 32
    o = np.zeros((n, D))

    def see(i, j):
        others = [x for x in range(n) if x != i]
        o[i, 4 + 3 * others.index(j) + 2] = 1.0

    see(0, 1); see(1, 0)          # 0-1 connected
    see(2, 3); see(3, 2)          # 2-3 connected
    g = radius_comm_groups(o, n)
    check("two components found", g == [[0, 1], [2, 3]], str(g))

    g2 = radius_comm_groups(np.zeros((n, D)), n)
    check("isolated agents form no group", g2 == [], str(g2))

    # one-sided visibility still connects (adjacency is symmetrized)
    o3 = np.zeros((n, D)); see_i = [x for x in range(n) if x != 0]
    o3[0, 4 + 3 * see_i.index(2) + 2] = 1.0
    check("one-sided visibility connects", radius_comm_groups(o3, n) == [[0, 2]])


def test_commit_shapes_and_sharing():
    """A committed group takes ONE shared action; non-members keep theirs."""
    rng = np.random.RandomState(1)
    n, D, T, n_act = 4, 32, 64, 6
    obs = np.zeros((T, n, D))
    for th in range(T):
        for i, j in [(0, 1), (1, 0), (2, 3), (3, 2)]:
            others = [x for x in range(n) if x != i]
            obs[th, i, 4 + 3 * others.index(j) + 2] = 1.0
    actions = rng.randint(0, n_act, size=(T, n, 1))

    acts, n_c, groups, forced = random_group_commit(
        obs, actions, n, n_act, eps=1.0, q=1.0, rng=rng)
    check("eps=1,q=1 commits every group", n_c == 2 * T, "n_c=%d" % n_c)
    same = all(len(set(acts[th, g, 0])) == 1 for th in range(T) for g in groups[th])
    check("q=1 gives one shared action per group", same)
    check("forced mask marks all members", forced.all())

    acts0, n0, _, forced0 = random_group_commit(
        obs, actions, n, n_act, eps=0.0, q=1.0, rng=rng)
    check("eps=0 changes nothing", n0 == 0 and np.array_equal(acts0, actions))
    check("eps=0 forces nobody", not forced0.any())

    # q=0.5: only forced members may differ from their policy action
    acts5, _, _, forced5 = random_group_commit(
        obs, actions, n, n_act, eps=1.0, q=0.5, rng=rng)
    check("non-forced keep their own action",
          np.array_equal(acts5[~forced5], actions[~forced5]))


# ---------------------------------------------------------------- optimism
def test_optimism():
    a = np.array([-2.0, -0.5, 0.0, 1.0])
    check("leaky 0 clips negatives to 0",
          np.allclose(optimistic_transform(a, 0.0), [0, 0, 0, 1]))
    check("leaky 0.3 scales negatives",
          np.allclose(optimistic_transform(a, 0.3), [-0.6, -0.15, 0, 1]))
    check("leaky 1 is a no-op", np.allclose(optimistic_transform(a, 1.0), a))


for fn in [test_marginal_matches_montecarlo, test_weight_formula,
           test_q1_reduces_to_old_marginal, test_grouping,
           test_commit_shapes_and_sharing, test_optimism]:
    print(fn.__name__)
    fn()

print("\n%s" % ("ALL PASSED" if not FAILED else "FAILED: %s" % FAILED))
sys.exit(1 if FAILED else 0)
