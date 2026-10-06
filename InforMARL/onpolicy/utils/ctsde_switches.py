"""
The three CT-SDE switches, ported from the MAT fork so InforMARL can be run
with an identical training recipe.

Ported verbatim from `Multi-Agent-Transformer_ctsde/mat/runner/shared/
joint_nav_runner.py`:
  - `radius_comm_groups`      (grouping)
  - `_random_group_commit_comm` -> `random_group_commit`   (exploration)
  - `arm_behaviour_weights`   (off-policy weighting)
  - `optimistic_transform`    (optimism, from mat_trainer.py)

Keep these in sync with the MAT fork. If a formula changes there it must
change here, or the arms stop being comparable.

InforMARL's actor is INDEPENDENT: the GNN aggregates a neighbourhood, but each
agent samples its own action with no conditioning on its neighbours' sampled
actions. So the autoregressive branch (`ar=True`) never applies here, and the
weight is the marginal
    mu_i = (1 - eps*q) * pi_i + eps*q/|A|,
which is exactly what the MAT fork uses for its `mappo` and `mat_comm_dec`
arms. At q = 1 this equals the older `marginal` formula, so a run that matches
a q=1 CT-SDE run is weighted identically.
"""
import numpy as np


def groups_from_adj(adj):
    """Connected components of size >= 2 of a boolean N x N adjacency."""
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    groups = []
    for s in range(n):
        if seen[s]:
            continue
        comp, stack = [], [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.nonzero(adj[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        if len(comp) >= 2:
            groups.append(sorted(comp))
    return groups


def radius_comm_groups(o, n_agents, off=4):
    """Connected components of the agent-radius graph, read off the radius
    observation's visibility flags.

    Slot k of agent i is the k-th OTHER agent in id order, and its visible
    flag sits at off + 3k + 2, where `off` is the own-state prefix length:
    4 for [vel(2), pos(2)], 2 when obs_abs_pos=0 drops the position. Pure
    distance, no task knowledge, so every arm groups identically. Singletons
    are dropped: a commitment needs >= 2.

    ONLY valid when the observation's peer visibility equals the true comm
    graph. Under a census ablation (obs_radius < sense_radius) it does not,
    and the caller must pass the env adjacency instead.
    """
    n = n_agents
    adj = np.eye(n, dtype=bool)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for k, j in enumerate(others):
            if o[i, off + 3 * k + 2] > 0.5:
                adj[i, j] = adj[j, i] = True
    seen = np.zeros(n, dtype=bool)
    groups = []
    for s in range(n):
        if seen[s]:
            continue
        comp, stack = [], [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.nonzero(adj[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        if len(comp) >= 2:
            groups.append(sorted(comp))
    return groups


def random_group_commit(obs, actions, n_agents, n_act, eps, q, rng=np.random,
                        adj=None, off=4):
    """Subset commit. With probability `eps` a radius group draws ONE shared
    uniformly random action c; each member independently takes it with
    probability `q`, and the rest keep their policy action.

    obs     [threads, agents, obs_dim]
    actions [threads, agents, 1]
    adj     [threads, agents, agents] bool, the TRUE comm graph from the env.
            Pass it whenever it is available: this is training machinery and
            must not depend on what the actor can see, or an observation
            ablation silently changes how much correlated exploration each
            condition receives. Falls back to the observation flags only when
            adj is None.
    Returns (new_actions, n_committed, groups_per_thread, forced_mask).
    """
    acts = actions.copy()
    committed = 0
    n_threads = obs.shape[0]
    forced = np.zeros((n_threads, n_agents), dtype=bool)
    groups_all = []
    for th in range(n_threads):
        if adj is not None:
            groups = groups_from_adj(adj[th])
        else:
            groups = radius_comm_groups(obs[th], n_agents, off)
        groups_all.append(groups)
        for comp in groups:
            if rng.rand() < eps:
                c = rng.randint(n_act)
                members = [i for i in comp if q >= 1.0 or rng.rand() < q]
                if not members:
                    continue
                acts[th, members, 0] = c
                forced[th, members] = True
                committed += 1
    return acts, committed, groups_all, forced


def arm_behaviour_weights(logp, acts, groups, eps, q, n_act, ar=False):
    """Decoupled-PPO weight w = min(1, pi/mu) under the subset-commit mixture.

    See the MAT fork for the autoregressive branch. For independent actors
    (InforMARL, mappo, mat_comm_dec) the behaviour marginal is
        mu_i = (1 - eps*q) * pi_i + eps*q/|A|
    for members of a group, and pi_i outside any group (so w = 1 there).
    """
    pi = np.exp(np.asarray(logp, dtype=np.float64).reshape(-1))
    acts = np.asarray(acts).reshape(-1)
    w = np.ones_like(pi)
    for g in groups:
        if not ar:
            mu = (1.0 - eps * q) * pi[g] + eps * q / float(n_act)
            w[g] = np.minimum(1.0, pi[g] / np.maximum(mu, 1e-300))
            continue
        a_nc = 1.0
        b = np.ones(n_act)
        p_prev = 1.0
        for i in g:
            a_nc *= pi[i]
            onehot = np.zeros(n_act)
            onehot[int(acts[i])] = 1.0
            b = b * (q * onehot + (1.0 - q) * pi[i])
            p_new = (1.0 - eps) * a_nc + eps * b.mean()
            mu = p_new / max(p_prev, 1e-300)
            w[i] = min(1.0, pi[i] / max(mu, 1e-300))
            p_prev = p_new
    return w


def optimistic_transform(adv, leaky_rate):
    """OptiMAPPO (Zhao et al. 2024): A <- max(leaky_rate * A, A).
    Positive advantages unchanged; negative ones scaled by leaky_rate."""
    return np.maximum(leaky_rate * adv, adv)
