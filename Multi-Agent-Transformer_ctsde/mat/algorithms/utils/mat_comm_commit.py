"""Discrete commitment pack + parent-token extras for mat_comm.

z is a categorical over n_targets slots plus NULL (last class). Geometry
rides in the share_obs tail *before* the comm mask so comm_from_share_obs
still reads the last N×N. Tokens stay inside the existing querier mask.
"""

import numpy as np
import torch
from torch.nn import functional as F


def pack_size(n_agents, n_targets):
    return 2 * n_agents + 3 * n_targets + n_agents * n_targets


def pack_commitment(agent_xy, target_xy, target_alive, visible, grid_size):
    """Flatten one timestep. Positions are divided by grid_size."""
    n_agents = int(agent_xy.shape[0])
    n_targets = int(target_xy.shape[0])
    g = float(max(int(grid_size), 1))
    ax = np.asarray(agent_xy, dtype=np.float32).reshape(n_agents, 2) / g
    tx = np.asarray(target_xy, dtype=np.float32).reshape(n_targets, 2) / g
    alive = np.asarray(target_alive, dtype=np.float32).reshape(n_targets)
    vis = np.asarray(visible, dtype=np.float32).reshape(n_agents, n_targets)
    return np.concatenate([ax.reshape(-1), tx.reshape(-1), alive, vis.reshape(-1)])


def parse_commitment_pack(state, n_agent, n_targets, comm_kind):
    """Read the pack off tiled share_obs [B, N, D]. Returns tensors on state.device."""
    if comm_kind == "bits":
        comm_tail = n_agent
    elif comm_kind == "pairwise":
        comm_tail = n_agent * n_agent
    else:
        raise ValueError("comm_kind must be bits or pairwise, got %r" % (comm_kind,))
    n_z_slots = n_targets
    psize = pack_size(n_agent, n_targets)
    body = state[:, 0, :]
    need = comm_tail + psize
    if body.size(-1) < need:
        raise ValueError(
            "share_obs dim %d too small for commitment pack %d + comm tail %d"
            % (body.size(-1), psize, comm_tail)
        )
    start = body.size(-1) - need
    chunk = body[:, start : start + psize]
    i = 0
    agent_xy = chunk[:, i : i + 2 * n_agent].reshape(-1, n_agent, 2)
    i += 2 * n_agent
    target_xy = chunk[:, i : i + 2 * n_z_slots].reshape(-1, n_z_slots, 2)
    i += 2 * n_z_slots
    target_alive = chunk[:, i : i + n_z_slots]
    i += n_z_slots
    visible = chunk[:, i : i + n_agent * n_z_slots].reshape(-1, n_agent, n_z_slots)
    return agent_xy, target_xy, target_alive, visible


def commitment_avail(visible, target_alive):
    """[B, N, n_targets+1] float: visible∧alive, NULL always 1."""
    legal = (visible > 0.5) & (target_alive.unsqueeze(1) > 0.5)
    null = torch.ones(legal.size(0), legal.size(1), 1, device=legal.device, dtype=legal.dtype)
    return torch.cat([legal, null], dim=-1).float()


def build_commit_tokens(
    querier_mask,
    actions,
    commits,
    z_self,
    agent_xy,
    target_xy,
    target_alive,
    include_same,
    action_dim,
    n_targets,
    tpdv,
):
    """Parent tokens for one (possibly querier-expanded) batch.

    querier_mask [B, N, N], actions/commits [B, N] long.
    z_self [B] long or None (same_target forced 0).
    agent_xy [B, N, 2], target_xy [B, T, 2], target_alive [B, T].
    Token = [action one-hot dim+1 | z one-hot n_z | rel_pos 2 | same 1].
    Parent z/a are int one-hots (no grad).
    """
    from mat.algorithms.utils.mat_comm_groups import (
        build_action_tokens,
        previous_in_component,
    )

    batch, n_agent = actions.shape
    n_z = n_targets + 1
    null_id = n_targets
    device = actions.device
    token_ids = build_action_tokens(querier_mask, actions + 1, 0)
    a_oh = F.one_hot(token_ids, num_classes=action_dim + 1).to(**tpdv)

    # Same right-shift as actions: position p carries pred(p)'s z, not its own.
    # Start token (no predecessor) is NULL. Parent z is an int one-hot (no grad).
    z_raw = commits.clamp(0, n_targets).long()
    prev_pos, has_prev = previous_in_component(querier_mask)
    z_shifted = torch.gather(z_raw, 1, prev_pos)
    z_ids = torch.where(has_prev, z_shifted, torch.full_like(z_shifted, null_id))
    z_oh = F.one_hot(z_ids, num_classes=n_z).to(**tpdv)

    # rel_pos of the parent's declared target in the querier's frame.
    # Pass agent_xy already replaced so every row uses q's xy (expand helper).
    gathered_xy = torch.gather(
        target_xy, 1, z_ids.clamp(max=n_targets - 1).unsqueeze(-1).expand(-1, -1, 2)
    )
    is_null = z_ids >= n_targets
    dead = torch.gather(
        target_alive, 1, z_ids.clamp(max=n_targets - 1)
    ) <= 0.5
    dead_or_null = is_null | dead
    rel = gathered_xy - agent_xy
    rel = rel.masked_fill(dead_or_null.unsqueeze(-1), 0.0)

    if include_same and z_self is not None:
        same = (
            (z_ids == z_self.unsqueeze(1))
            & (z_ids < n_targets)
            & (~dead_or_null)
        ).float().unsqueeze(-1)
    else:
        same = torch.zeros(batch, n_agent, 1, device=device, dtype=a_oh.dtype)

    return torch.cat([a_oh, z_oh, rel.to(**tpdv), same.to(**tpdv)], dim=-1)


def commit_token_dim(action_dim, n_targets):
    return (action_dim + 1) + (n_targets + 1) + 2 + 1


def expand_querier_xy(agent_xy, n_agent):
    """[B, N, 2] -> [B*N, N, 2] with every row of copy q equal to agent q's xy."""
    batch = agent_xy.size(0)
    q_xy = agent_xy.unsqueeze(2).expand(batch, n_agent, n_agent, 2)
    return q_xy.reshape(batch * n_agent, n_agent, 2)


def own_commit_rel(agent_xy, target_xy, target_alive, commits, n_targets):
    """[B, N, 2] relative position of each agent's own z. 0 if NULL or dead."""
    idx = commits.clamp(0, max(int(n_targets) - 1, 0))
    gathered = target_xy.gather(1, idx.unsqueeze(-1).expand(-1, -1, 2))
    dead = target_alive.gather(1, idx) <= 0.5
    is_null = commits >= n_targets
    rel = gathered - agent_xy
    return rel.masked_fill((is_null | dead).unsqueeze(-1), 0.0)


def action_reduces_dist(action, agent_xy, target_xy, schema):
    """True if the discrete action strictly cuts Manhattan distance.

    schema 'lbf': 0=NONE 1=N 2=S 3=W 4=E 5+=LOAD (no).
    schema 'pp':  0=N 1=S 2=W 3=E 4+=stay (no).
    Positions are in the same units (grid cells, not normalized).
    """
    ay, ax = float(agent_xy[0]), float(agent_xy[1])
    ty, tx = float(target_xy[0]), float(target_xy[1])
    before = abs(ty - ay) + abs(tx - ax)
    a = int(action)
    ny, nx = ay, ax
    if schema == "lbf":
        if a == 1:
            ny = ay - 1
        elif a == 2:
            ny = ay + 1
        elif a == 3:
            nx = ax - 1
        elif a == 4:
            nx = ax + 1
        else:
            return False
    elif schema == "pp":
        if a == 0:
            ny = ay - 1
        elif a == 1:
            ny = ay + 1
        elif a == 2:
            nx = ax - 1
        elif a == 3:
            nx = ax + 1
        else:
            return False
    else:
        raise ValueError("schema must be lbf or pp, got %r" % (schema,))
    after = abs(ty - ny) + abs(tx - nx)
    return after < before - 1e-6


def z_meaning_step(actions, commits, agent_xy, target_xy, target_alive, grid_size, schema):
    """Per-agent approach / switch helpers. xy are pack-normalized.

    Returns approach_hit, approach_n, null_n, n_agents as python ints.
    approach excludes NULL and dead named targets.
    """
    n = int(actions.shape[0])
    n_targets = int(target_xy.shape[0])
    g = float(max(int(grid_size), 1))
    hit = 0
    n_app = 0
    n_null = 0
    for i in range(n):
        z = int(commits[i])
        if z >= n_targets:
            n_null += 1
            continue
        if float(target_alive[z]) <= 0.5:
            continue
        ax = (float(agent_xy[i, 0]) * g, float(agent_xy[i, 1]) * g)
        tx = (float(target_xy[z, 0]) * g, float(target_xy[z, 1]) * g)
        n_app += 1
        if action_reduces_dist(int(actions[i]), ax, tx, schema):
            hit += 1
    return hit, n_app, n_null, n


def z_agreement_step(commits, n_targets):
    """1 if at least two agents declare the same non-NULL food."""
    z = np.asarray(commits).reshape(-1).astype(np.int64)
    named = z[z < int(n_targets)]
    if named.size < 2:
        return 0
    _, counts = np.unique(named, return_counts=True)
    return int(np.any(counts >= 2))
