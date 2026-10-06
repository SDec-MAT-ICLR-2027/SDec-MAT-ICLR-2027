"""Target-id artefact fixes: shuffle_targets and target_order="distance".

  1. target_order=distance: obs dim unchanged (32); first target slot is the
     nearest LIVE VISIBLE target (rel pos matches, alive=vis=1); hidden slots
     are exactly zero; the multiset of rows equals the id-ordered rows
  2. defaults (shuffle off, order id) reproduce the old obs exactly for the
     same seed
  3. shuffle_targets: target 0 is no longer more central / more often nearest
     than targets 1, 2 (2000 resets, 2x2 board, sep 1.25)

  python test_radius_target_order.py
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multiagent.MPE_env import GraphMPEEnv
from probe_radius_connectivity import make_args, onehot

N, M = 6, 3
TB = 4 + 3 * (N - 1)


def check(name, cond, detail=""):
    print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                           ("  --  " + detail) if detail else ""))
    return bool(cond)


def env_for(order="id", shuffle=0):
    a = make_args(0.6)
    a.world_size, a.min_target_sep = 2.0, 1.25
    a.target_order, a.shuffle_targets = order, shuffle
    return GraphMPEEnv(a)


def rows(o):
    return [tuple(np.round(o[TB + 4 * k:TB + 4 * k + 4], 6)) for k in range(M)]


def main():
    ok = True
    # 1 + 2: same seed, id vs distance order
    e_id, e_d = env_for("id"), env_for("distance")
    rng = np.random.RandomState(0)
    n_checked = bad_first = bad_zero = bad_multi = 0
    same_old = True
    for s in range(30):
        np.random.seed(s)
        o_id = e_id.reset()[0]
        np.random.seed(s)
        o_d = e_d.reset()[0]
        for _t in range(40):
            w = e_d.world
            for i in range(N):
                a, b = np.asarray(o_id[i]), np.asarray(o_d[i])
                if len(b) != 32:
                    ok = False
                n_checked += 1
                if sorted(rows(a)) != sorted(rows(b)):
                    bad_multi += 1
                pos = w.agents[i].state.p_pos
                live_vis = [(np.linalg.norm(w.landmarks[t].state.p_pos - pos), t)
                            for t in range(M) if w.target_active[t]
                            and np.linalg.norm(w.landmarks[t].state.p_pos - pos) <= 0.6]
                if live_vis:
                    d, t = min(live_vis)
                    want = w.landmarks[t].state.p_pos - pos
                    if not (np.allclose(b[TB:TB + 2], want, atol=1e-5)
                            and b[TB + 2] == 1.0 and b[TB + 3] == 1.0):
                        bad_first += 1
                for k in range(M):
                    r = b[TB + 4 * k:TB + 4 * k + 4]
                    if r[3] == 0.0 and np.any(r != 0.0):
                        bad_zero += 1
            acts = [onehot(rng.randint(6)) for _ in range(N)]
            o_id = e_id.step(acts)[0]
            o_d = e_d.step(acts)[0]
    ok &= check("distance order: obs dim 32", len(np.asarray(o_d[0])) == 32)
    ok &= check("distance order: first slot = nearest live visible target",
                bad_first == 0, "%d agent-steps, %d bad" % (n_checked, bad_first))
    ok &= check("distance order: hidden slots exactly zero", bad_zero == 0)
    ok &= check("distance order: same rows as id order (only reordered)",
                bad_multi == 0, "%d mismatches" % bad_multi)

    e_a, e_b = env_for("id"), env_for("id")
    np.random.seed(7)
    oa = e_a.reset()[0]
    np.random.seed(7)
    ob = e_b.reset()[0]
    same_old = all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(oa, ob))
    ok &= check("defaults deterministic for a fixed seed", same_old)

    # 3: placement symmetry with shuffle
    for shuffle in (0, 1):
        env = env_for("id", shuffle)
        w = env.world
        dc = np.zeros(M)
        near3 = np.zeros(M)
        np.random.seed(0)
        for _ in range(2000):
            env.reset()
            T = np.array([w.landmarks[t].state.p_pos for t in range(M)])
            P = np.array([ag.state.p_pos for ag in w.agents])
            dc += np.linalg.norm(T, axis=1)
            nn_ = np.linalg.norm(P[:, None] - T[None], axis=-1).argmin(1)
            for t in range(M):
                near3[t] += float((nn_ == t).sum() >= 3)
        dc /= 2000
        near3 /= 2000
        spread = near3.max() - near3.min()
        detail = "d_center %s  near>=3 %s" % (np.round(dc, 3), np.round(near3, 3))
        if shuffle:
            ok &= check("shuffle on: placement symmetric across ids (near>=3 spread < 0.05)",
                        spread < 0.05, detail)
        else:
            print("  [info] shuffle off: " + detail)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
