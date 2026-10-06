"""Communication-group utilities for the comm-constrained MAT variant (`mat_comm`).

Ported from src/modules/agents/ctsde_transformer.py. The PyMARL copy still
closes the graph; this file does not (iter 170). Do not "sync" the copy
back without an explicit decision — the two will diverge on a chain.

The chain is:

    comm bits  ->  raw adjacency (diag on)  ->  two masks
                                                group_mask   = 1-hop neighborhood
                                                group & tril

`group_mask[i, j]` is True iff j is i or i can talk to j. No transitive
closure: a 2-hop chain A—B—C does not put A and C in each other's mask.
Neighborhoods overlap; they are not a partition / equivalence class.

The decoder is per-querier: agent i's pass uses a mask whose every row is
nbhd(i) (diagonal forced on). Tokens on that pass have predecessors inside
nbhd(i). A shared token tensor cannot do this for overlapping neighborhoods.
"""

import itertools
import math

import numpy as np
import torch


def parse_fixed_group_sizes(spec, n_agents):
    """Parse '3,3,4' into sizes that partition n_agents. None/'' → None."""
    if spec is None:
        return None
    text = str(spec).strip()
    if text == "":
        return None
    parts = [p.strip() for p in text.split(",")]
    sizes = []
    for p in parts:
        if not p.isdigit() or int(p) < 1:
            raise ValueError(
                "fixed_groups must be positive ints, got %r in %r" % (p, spec)
            )
        sizes.append(int(p))
    n_agents = int(n_agents)
    if sum(sizes) != n_agents:
        raise ValueError(
            "fixed_groups %r sums to %d, not n_agents=%d"
            % (spec, sum(sizes), n_agents)
        )
    if len(sizes) < 2:
        raise ValueError(
            "fixed_groups needs at least 2 groups (got %r); one group is MAT"
            % (spec,)
        )
    return sizes


def id_group_members(sizes):
    """Consecutive id blocks: [3,3,4] → [[0,1,2],[3,4,5],[6,7,8,9]]."""
    out = []
    start = 0
    for sz in sizes:
        sz = int(sz)
        out.append(list(range(start, start + sz)))
        start += sz
    return out


def nearest_group_members(positions, sizes, alive_agents=None):
    """Tightest remaining cliques of the given sizes, once per episode.

    Sizes are filled largest-first so the leftover is not a dumped far group.
    Each pick is the remaining subset of that size with min pairwise Manhattan.
    """
    pos = np.asarray(positions)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions must be [N, 2], got %s" % (pos.shape,))
    n = int(pos.shape[0])
    sizes = [int(s) for s in sizes]
    if sum(sizes) != n:
        raise ValueError(
            "nearest group sizes %s sum %d != n=%d" % (sizes, sum(sizes), n)
        )
    if alive_agents is None:
        remaining = set(range(n))
    else:
        alive = np.asarray(alive_agents, dtype=np.float32).reshape(n)
        remaining = set(int(i) for i in range(n) if float(alive[i]) >= 0.5)
        dead = [i for i in range(n) if i not in remaining]
        if dead:
            raise ValueError(
                "nearest grouping is reset-only; all agents must be alive, "
                "got dead %s" % dead
            )

    def manh(i, j):
        return abs(int(pos[i, 0]) - int(pos[j, 0])) + abs(
            int(pos[i, 1]) - int(pos[j, 1])
        )

    def pair_sum(combo):
        total = 0
        m = len(combo)
        for a in range(m):
            for b in range(a + 1, m):
                total += manh(combo[a], combo[b])
        return total

    def best_subset(pool, sz):
        pool = sorted(pool)
        if sz == len(pool):
            return list(pool)
        best = None
        for combo in itertools.combinations(pool, sz):
            cost = pair_sum(combo)
            key = (cost, combo)
            if best is None or key < best:
                best = key
        if best is None:
            raise ValueError("no subset of size %d in %s" % (sz, pool))
        return list(best[1])

    # Fill large groups first so leftovers are not four far agents.
    order = sorted(range(len(sizes)), key=lambda i: (-sizes[i], i))
    placed = [None] * len(sizes)
    for idx in order:
        sz = sizes[idx]
        if len(remaining) < sz:
            raise ValueError(
                "need %d remaining agents, have %d" % (sz, len(remaining))
            )
        group = best_subset(remaining, sz)
        remaining.difference_update(group)
        placed[idx] = sorted(group)
    if remaining:
        raise ValueError("leftover agents after nearest grouping: %s" % remaining)
    return placed


def mean_pairwise_manhattan(positions, members):
    """Mean of within-group pairwise Manhattan distances (pairs only)."""
    pos = np.asarray(positions)
    dists = []
    for ids in members:
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                dists.append(
                    abs(int(pos[i, 0]) - int(pos[j, 0]))
                    + abs(int(pos[i, 1]) - int(pos[j, 1]))
                )
    if not dists:
        return float("nan")
    return float(np.mean(dists))


def fixed_group_adjacency(n_agents, sizes, alive_agents=None, members=None):
    """Block-diagonal adj for a partition. Dead agents are diag-only."""
    n = int(n_agents)
    if sum(sizes) != n:
        raise ValueError(
            "fixed group sizes %s sum %d != n_agents %d"
            % (sizes, sum(sizes), n)
        )
    if members is None:
        members = id_group_members(sizes)
    adj = np.eye(n, dtype=np.float32)
    if alive_agents is None:
        alive = np.ones(n, dtype=np.float32)
    else:
        alive = np.asarray(alive_agents, dtype=np.float32).reshape(n)
    seen = []
    for ids in members:
        seen.extend(ids)
        living = [int(i) for i in ids if float(alive[int(i)]) >= 0.5]
        for i in living:
            for j in living:
                adj[i, j] = 1.0
    if sorted(seen) != list(range(n)):
        raise ValueError("members %s are not a partition of 0..%d" % (members, n - 1))
    return adj


def comm_from_share_obs(state, n_agent, kind):
    """Read the communication graph off the tiled share_obs / state tensor.

    `state` is [B, N, state_dim]; every agent row is the same vector. The
    graph lives in the tail of that vector, with two layouts:

      bits      last n_agent entries are per-agent connectivity bits
                (Aerial Ordering: star around the sensor).
      pairwise  last n_agent**2 entries are a flattened [N, N] adjacency
                (DICG Predator-Prey Manhattan graph).

    Returns a tensor `build_group_mask` accepts: [B, N] for bits, [B, N, N]
    for pairwise. Refuses to guess the layout — passing the Aerial tail to
    a pairwise reader (or the last N obs numbers of concat state as bits)
    would silently build the wrong groups.
    """
    if kind not in ("bits", "pairwise"):
        raise ValueError(
            "comm_mask_kind must be 'bits' or 'pairwise', got %r" % (kind,)
        )
    if state.dim() != 3 or state.size(1) != n_agent:
        raise ValueError(
            "state must be [B, n_agent, state_dim], got %s (n_agent=%d)"
            % (tuple(state.shape), n_agent)
        )
    tail = state[:, 0, :]
    if kind == "bits":
        if tail.size(-1) < n_agent:
            raise ValueError(
                "bits layout needs last %d state entries, got state_dim=%d"
                % (n_agent, tail.size(-1))
            )
        return tail[:, -n_agent:]
    n2 = n_agent * n_agent
    if tail.size(-1) < n2:
        raise ValueError(
            "pairwise layout needs last %d state entries, got state_dim=%d"
            % (n2, tail.size(-1))
        )
    return tail[:, -n2:].reshape(-1, n_agent, n_agent)


def build_adjacency(comm_mask):
    """Normalise whatever the env reports into a [B, N, N] bool adjacency.

    Two conventions are accepted:
      - per-agent connectivity bits, [B, N] or [B, N, 1]. Every connected agent
        is taken to be mutually reachable. This is what Aerial Ordering emits
        (`get_comm_mask()` -> one bit per agent, star topology around the
        sensor), and it can only ever yield ONE multi-agent group plus isolated
        singletons.
      - a pairwise adjacency, [B, N, N]. This is the convention that allows
        several simultaneous disjoint groups; nothing below needs to change if
        the env is later extended to emit one.

    The diagonal is always set, so an isolated agent is its own group and no
    attention row can ever be fully masked (which would produce NaNs).
    """
    if comm_mask.dim() == 2:
        bits = comm_mask > 0.5
        adj = bits.unsqueeze(2) & bits.unsqueeze(1)
    elif comm_mask.size(-1) == 1:
        bits = comm_mask.squeeze(-1) > 0.5
        adj = bits.unsqueeze(2) & bits.unsqueeze(1)
    else:
        # Keep the env row as-is. Distance graphs are already symmetric;
        # a per-agent degree cap may be directed and must not be OR-ed back.
        adj = comm_mask > 0.5

    n_agents = adj.size(1)
    eye = torch.eye(n_agents, device=adj.device, dtype=torch.bool).unsqueeze(0)
    return adj | eye


def transitive_closure(adj):
    """Reachability closure of a symmetric [B, N, N] bool adjacency.

    Squaring the boolean matrix doubles reachable path length, so ceil(log2(N))
    squarings saturate reachability for any graph on N nodes.
    """
    n_agents = adj.size(-1)
    reach = adj
    n_steps = max(1, math.ceil(math.log2(max(n_agents, 2))))
    for _ in range(n_steps):
        reach_next = torch.matmul(reach.float(), reach.float()) > 0
        if bool(torch.equal(reach_next, reach)):
            break
        reach = reach_next
    return reach


def build_group_mask(comm_mask):
    """[B, N, N] bool: mask[b, i, j] is True iff j is in i's 1-hop neighborhood.

    Raw adjacency plus the diagonal. No reachability closure — a chain
    A—B—C does not let A attend to C. Overlapping rows are expected.
    `transitive_closure` is kept only for diagnostics / old tests.
    """
    return build_adjacency(comm_mask)


def build_group_causal_mask(group_mask):
    """Neighborhood causal mask: neighbor (or self) AND not later in decode order.

    Non-strict lower triangle, so position i may attend to its own token. That
    token is its predecessor's action, never its own (see build_action_tokens),
    so nothing leaks.
    """
    n_agents = group_mask.size(-1)
    lower = torch.tril(
        torch.ones((n_agents, n_agents), device=group_mask.device, dtype=torch.bool),
        diagonal=0,
    )
    return group_mask.bool() & lower.unsqueeze(0)


def querier_group_masks(group_mask):
    """Per-querier neighborhood masks for per-agent decode.

    qmask[b*N+q, p, k] is True iff k is in querier q's neighborhood or p==k
    (diagonal keeps every row safe for softmax). causal is qmask ∧ tril.

    Returns:
        qmask: [B*N, N, N] bool
        causal: [B*N, N, N] bool
    """
    g = group_mask.bool()
    n_agents = g.size(-1)
    batch = g.size(0)
    device = g.device
    eye = torch.eye(n_agents, device=device, dtype=torch.bool)
    qmask = g.unsqueeze(2).expand(batch, n_agents, n_agents, n_agents)
    qmask = qmask | eye.view(1, 1, n_agents, n_agents)
    qmask = qmask.reshape(batch * n_agents, n_agents, n_agents)
    tril = torch.tril(
        torch.ones((n_agents, n_agents), device=device, dtype=torch.bool)
    )
    return qmask, qmask & tril.unsqueeze(0)


def querier_mask_for(group_mask, querier):
    """[B, N, N] mask / causal for a single querier index (rollout step)."""
    g = group_mask.bool()
    n_agents = g.size(-1)
    device = g.device
    row = g[:, querier, :].unsqueeze(1)
    eye = torch.eye(n_agents, device=device, dtype=torch.bool)
    qmask = row.expand(-1, n_agents, n_agents) | eye.unsqueeze(0)
    tril = torch.tril(
        torch.ones((n_agents, n_agents), device=device, dtype=torch.bool)
    )
    return qmask, qmask & tril.unsqueeze(0)


def previous_in_component(group_mask):
    """Nearest earlier decode position in i's neighborhood (1-hop).

    Returns:
        prev_pos: [B, N] long, clamped to 0 where there is no predecessor.
        has_prev: [B, N] bool, False at the first position of every component.
    """
    n_agents = group_mask.size(-1)
    device = group_mask.device
    strict_lower = torch.tril(
        torch.ones((n_agents, n_agents), device=device, dtype=torch.bool), diagonal=-1
    )
    candidates = group_mask.bool() & strict_lower.unsqueeze(0)
    idx = torch.arange(n_agents, device=device).view(1, 1, n_agents).expand_as(candidates)
    prev_pos = idx.masked_fill(~candidates, -1).max(dim=-1).values
    return prev_pos.clamp(min=0), prev_pos >= 0


def build_action_tokens(group_mask, actions, start_action_id):
    """Decoder input tokens that never cross a component boundary.

    This is the half that attention masking cannot do. MAT right-shifts the
    action sequence globally, so decode position p receives the action of
    position p-1 as its OWN residual input — regardless of whether the two share
    a component. No attention mask can undo that, because the leaked action is
    not something p attends to, it is what p is built from.

    Feeding each position the action of its nearest earlier neighbor keeps
    the residual input inside the mask this pass was built with. Per-agent
    decode passes querier i's neighborhood as that mask, so predecessors
    stay inside nbhd(i).

    Args:
        group_mask: [B, N, N] bool.
        actions: [B, N] long token ids.
        start_action_id: id used where a position has no predecessor.
    """
    prev_pos, has_prev = previous_in_component(group_mask)
    gathered = torch.gather(actions, 1, prev_pos)
    return torch.where(has_prev, gathered, torch.full_like(gathered, start_action_id))


def one_hop_shift_leak(group_mask):
    """[B, N] bool: querier i reads a token whose predecessor is outside nbhd(i).

    pred(j) is taken from the per-querier mask (iter 172), not the shared
    adjacency. After per-agent decode this must be all False.
    """
    qmask, _ = querier_group_masks(group_mask)
    prev_pos, has_prev = previous_in_component(qmask)
    n_agents = group_mask.size(-1)
    batch = group_mask.size(0)
    prev_pos = prev_pos.view(batch, n_agents, n_agents)
    has_prev = has_prev.view(batch, n_agents, n_agents)
    pred_in = torch.gather(group_mask.bool(), 2, prev_pos)
    eye = torch.eye(
        n_agents, device=group_mask.device, dtype=torch.bool
    ).unsqueeze(0)
    neighbor = group_mask.bool() & ~eye
    return (neighbor & has_prev & ~pred_in).any(dim=-1)


def one_hop_shift_leak_np(adj):
    """Numpy wrapper: adj [T, N, N] or [N, N] bool -> leak same leading dims."""
    t = torch.as_tensor(np.asarray(adj, dtype=bool))
    squeeze = t.dim() == 2
    if squeeze:
        t = t.unsqueeze(0)
    out = one_hop_shift_leak(t).numpy()
    return out[0] if squeeze else out


def component_ids(group_mask):
    """[B, N] long: smallest index in i's neighborhood (not an equivalence class)."""
    n_agents = group_mask.size(-1)
    idx = torch.arange(n_agents, device=group_mask.device).view(1, 1, n_agents)
    idx = idx.expand_as(group_mask)
    return idx.masked_fill(~group_mask.bool(), n_agents).min(dim=-1).values


def component_sizes(group_mask):
    """[B, N] long: size of each agent's 1-hop neighborhood (1 when isolated)."""
    return group_mask.bool().sum(dim=-1)
