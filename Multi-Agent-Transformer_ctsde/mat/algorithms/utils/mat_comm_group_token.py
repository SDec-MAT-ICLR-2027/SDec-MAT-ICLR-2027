"""Per-clique GO/NO-GO token for Pursuit mat_comm.

z in {0=NO-GO, 1=GO}. One draw per same-prey clique (size >= 2).
Singletons do not sample. Decoder START is blank (id 2) for degree-1
agents, GO/NO-GO only for clique heads. Every clique member sees the
same z. Token log-prob is counted once, on the lowest-id member (is_rep).

Pack lives in share_obs immediately before the pairwise comm tail so
comm_from_share_obs still reads the last N*N entries.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F

from mat.algorithms.mat.algorithm.ma_transformer import init_
from mat.algorithms.utils.mat_comm_groups import previous_in_component

# Decoder token ids: 0=NO-GO, 1=GO, 2=blank START, 3+a=action a.
ACTION_TOKEN_SHIFT = 3
TOKEN_WIDTH_EXTRA = 3

Z_NOGO = 0
Z_GO = 1
Z_BLANK = 2
N_Z = 2
HONOR_NONE = 8
HEAR_NONE = -1
NEIGHBORS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def pack_width(n_agent):
    return 4 * int(n_agent)


def decoder_token_dim(action_dim):
    return int(action_dim) + TOKEN_WIDTH_EXTRA


def tokens_to_input_z(group_mask, actions, z, action_dim, tpdv):
    """AR input: clique head is z (NO-GO/GO); loner is blank START; else prior action."""
    prev_pos, has_prev = previous_in_component(group_mask)
    act_ids = actions.long() + ACTION_TOKEN_SHIFT
    gathered = torch.gather(act_ids, 1, prev_pos)
    has_partner = group_mask.bool().sum(dim=-1) > 1
    start = torch.where(
        has_partner, z.long(), torch.full_like(z.long(), Z_BLANK)
    )
    token_ids = torch.where(has_prev, gathered, start)
    return F.one_hot(token_ids, num_classes=decoder_token_dim(action_dim)).to(**tpdv)


def honor_action_toward(ax, ay, px, py):
    dx = int(px) - int(ax)
    dy = int(py) - int(ay)
    for a, (ndx, ndy) in enumerate(NEIGHBORS):
        if dx == ndx and dy == ndy:
            return a
    return HONOR_NONE


def build_group_token_pack(
    agent_positions,
    prey_positions,
    alive_agents,
    alive_preys,
    assigned,
    recruit_assigned,
):
    """ready_count, honor_action, is_rep, hear_rep — each length N.

    ready / honor / is_rep use catch-radius assignment. hear_rep copies
    the clique representative to every agent assigned at recruit radius
    to a prey that already has a size>=2 catch clique.
    """
    agent_positions = np.asarray(agent_positions)
    prey_positions = np.asarray(prey_positions)
    alive_agents = np.asarray(alive_agents, dtype=np.float32).reshape(-1)
    assigned = np.asarray(assigned, dtype=np.int64).reshape(-1)
    recruit_assigned = np.asarray(recruit_assigned, dtype=np.int64).reshape(-1)
    n_agents = int(alive_agents.shape[0])
    if assigned.shape != (n_agents,):
        raise ValueError("assigned must be [%d], got %s" % (n_agents, assigned.shape))
    if recruit_assigned.shape != (n_agents,):
        raise ValueError(
            "recruit_assigned must be [%d], got %s"
            % (n_agents, recruit_assigned.shape)
        )

    members = {}
    for i in range(n_agents):
        p = int(assigned[i])
        if p < 0 or alive_agents[i] < 0.5:
            continue
        members.setdefault(p, []).append(i)

    ready = np.ones(n_agents, dtype=np.float32)
    honor = np.full(n_agents, float(HONOR_NONE), dtype=np.float32)
    is_rep = np.zeros(n_agents, dtype=np.float32)
    hear = np.full(n_agents, float(HEAR_NONE), dtype=np.float32)
    rep_of_prey = {}
    for p, ids in members.items():
        n = len(ids)
        for i in ids:
            ready[i] = float(n)
            ax = int(agent_positions[i, 0])
            ay = int(agent_positions[i, 1])
            px = int(prey_positions[p, 0])
            py = int(prey_positions[p, 1])
            honor[i] = float(honor_action_toward(ax, ay, px, py))
        if n >= 2:
            rep = min(ids)
            is_rep[rep] = 1.0
            rep_of_prey[p] = rep
            for i in ids:
                hear[i] = float(rep)
    for i in range(n_agents):
        if alive_agents[i] < 0.5:
            continue
        p = int(recruit_assigned[i])
        if p in rep_of_prey:
            hear[i] = float(rep_of_prey[p])
    return np.concatenate([ready, honor, is_rep, hear], axis=0).astype(np.float32)


def parse_group_token_pack(state, n_agent, comm_kind):
    """state [B, N, D] tiled. Pack sits just before the pairwise comm tail."""
    if comm_kind != "pairwise":
        raise ValueError("group token pack requires comm_mask_kind=pairwise")
    if state.dim() != 3 or state.size(1) != n_agent:
        raise ValueError(
            "state must be [B, n_agent, D], got %s" % (tuple(state.shape),)
        )
    n2 = n_agent * n_agent
    width = pack_width(n_agent)
    tail = state[:, 0, :]
    need = n2 + width
    if tail.size(-1) < need:
        raise ValueError(
            "share_obs dim %d too small for group-token pack %d + comm %d"
            % (tail.size(-1), width, n2)
        )
    pack = tail[:, -(n2 + width) : -n2]
    ready = pack[:, :n_agent]
    honor = pack[:, n_agent : 2 * n_agent]
    is_rep = pack[:, 2 * n_agent : 3 * n_agent]
    hear = pack[:, 3 * n_agent :]
    return ready, honor, is_rep, hear


def masked_mean_pool(obs_rep, group_mask):
    """Mean over 1-hop neighborhood. obs_rep [B,N,D], mask [B,N,N] bool."""
    w = group_mask.to(dtype=obs_rep.dtype)
    denom = w.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return torch.matmul(w, obs_rep) / denom


def recruit_start_override(token_head, pooled, ready, hear_rep):
    """W(pair pool) for hear-but-not-catcher; mask [B,N], emb [B,N,D]."""
    if ready.dim() != 2:
        ready = ready.squeeze(-1)
    if hear_rep.dim() != 2:
        hear_rep = hear_rep.squeeze(-1)
    if ready.shape != hear_rep.shape:
        raise ValueError(
            "ready and hear_rep must match, got %s vs %s"
            % (tuple(ready.shape), tuple(hear_rep.shape))
        )
    if pooled.dim() != 3 or pooled.shape[:2] != ready.shape:
        raise ValueError(
            "pooled must be [B,N,D] matching ready, got %s"
            % (tuple(pooled.shape),)
        )
    is_recruit = (hear_rep >= 0) & (ready < 2.0 - 1e-6)
    dim = pooled.size(-1)
    idx = hear_rep.long().clamp(min=0).unsqueeze(-1).expand(-1, -1, dim)
    gathered = pooled.gather(1, idx)
    return token_head.recruit_start(gathered), is_recruit


def broadcast_z(z_rep, is_rep, group_mask):
    """Copy each representative's z to its clique; others stay NO-GO."""
    batch, n_agent = is_rep.shape
    device = z_rep.device
    ids = torch.arange(n_agent, device=device).view(1, 1, n_agent)
    cand = group_mask & (is_rep > 0.5).unsqueeze(1)
    big = n_agent
    tagged = torch.where(cand, ids, torch.full_like(ids, big))
    rep_j = tagged.min(dim=-1).values
    has = rep_j < n_agent
    rep_j = rep_j.clamp(max=n_agent - 1)
    gathered = z_rep.gather(1, rep_j)
    return torch.where(has, gathered, torch.zeros_like(gathered))


class GroupTokenHead(nn.Module):
    """Shared MLP_score(pooled, ready) and MLP_nogo(pooled). Two layers."""

    def __init__(self, n_embd):
        super(GroupTokenHead, self).__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(n_embd + 1),
            init_(nn.Linear(n_embd + 1, n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(n_embd, 1)),
        )
        self.nogo = nn.Sequential(
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(n_embd, 1)),
        )
        self.recruit_start = nn.Sequential(
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
        )

    def forward(self, pooled, ready_count):
        if ready_count.dim() == 2:
            ready = ready_count.unsqueeze(-1)
        else:
            ready = ready_count
        feat = torch.cat([pooled, ready], dim=-1)
        go = self.score(feat).squeeze(-1)
        nogo = self.nogo(pooled).squeeze(-1)
        return torch.stack([nogo, go], dim=-1)


def token_logits(head, pooled, ready_count, is_rep):
    """GO vs NO-GO; no ready/catch_min mask. Dead rows (not is_rep) do not sample."""
    logits = head(pooled, ready_count)
    dead = is_rep < 0.5
    logits = logits.masked_fill(dead.unsqueeze(-1), -1e10)
    logits = logits.clone()
    logits[..., Z_NOGO] = torch.where(
        dead, torch.zeros_like(logits[..., Z_NOGO]), logits[..., Z_NOGO]
    )
    return logits


def sample_group_z(
    head, pooled, ready_count, is_rep, group_mask, deterministic
):
    logits = token_logits(head, pooled, ready_count, is_rep)
    dist = Categorical(logits=logits)
    z_rep = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
    logp = dist.log_prob(z_rep)
    ent = dist.entropy()
    z = broadcast_z(z_rep, is_rep, group_mask)
    logp = torch.where(is_rep > 0.5, logp, torch.zeros_like(logp))
    ent = torch.where(is_rep > 0.5, ent, torch.zeros_like(ent))
    return z, logp, ent


def clique_z_outcome(
    is_rep,
    ready,
    group_z,
    group_mask,
    events_batch,
    catch_reward,
    catch_fail_reward,
):
    """Same-step teacher for log π(z). GO + this clique jointly caught → +catch_reward;
    GO + this clique undershot → catch_fail_reward; NO-GO and non-reps → 0.

    is_rep/ready/group_z: [B, N]. group_mask: [B, N, N] bool. events_batch: length B,
    each a list of dicts with kind ('joint'|'undershoot') and agents (ids).
    """
    is_rep = np.asarray(is_rep)
    ready = np.asarray(ready)
    group_z = np.asarray(group_z)
    group_mask = np.asarray(group_mask, dtype=bool)
    if is_rep.ndim != 2:
        raise ValueError("is_rep must be [B, N], got %s" % (is_rep.shape,))
    batch, n_agent = is_rep.shape
    if ready.shape != (batch, n_agent) or group_z.shape != (batch, n_agent):
        raise ValueError("ready/group_z must match is_rep")
    if group_mask.shape != (batch, n_agent, n_agent):
        raise ValueError(
            "group_mask must be [B, N, N], got %s" % (group_mask.shape,)
        )
    if len(events_batch) != batch:
        raise ValueError(
            "events_batch length %d != batch %d" % (len(events_batch), batch)
        )
    out = np.zeros((batch, n_agent), dtype=np.float32)
    cr = float(catch_reward)
    cfr = float(catch_fail_reward)
    for b in range(batch):
        events = events_batch[b]
        if events is None:
            raise ValueError("catch_events missing for batch %d" % b)
        for i in range(n_agent):
            if is_rep[b, i] < 0.5 or ready[b, i] < 2.0 - 1e-6:
                continue
            if int(group_z[b, i]) != Z_GO:
                continue
            members = set(np.flatnonzero(group_mask[b, i]).tolist())
            kind = None
            for ev in events:
                agents = set(int(a) for a in ev["agents"])
                if not agents.intersection(members):
                    continue
                if ev["kind"] == "joint":
                    kind = "joint"
                    break
                if ev["kind"] == "undershoot":
                    kind = "undershoot"
            if kind == "joint":
                out[b, i] = cr
            elif kind == "undershoot":
                out[b, i] = cfr
    return out


def balance_z_outcome_by_ready(z_out, ready, catch_min):
    """Inverse-frequency weights on nonzero z outcomes.

    Two buckets: 2 <= ready < catch_min (pairs) vs ready >= catch_min.
    Every row is kept. If a bucket is empty, weights stay 1.
    """
    if z_out.shape != ready.shape:
        raise ValueError(
            "z_out and ready must match, got %s vs %s"
            % (tuple(z_out.shape), tuple(ready.shape))
        )
    cm = float(catch_min)
    if cm < 2:
        raise ValueError("catch_min must be >= 2, got %r" % catch_min)
    nz = z_out.abs() > 1e-8
    low = nz & (ready >= 2.0 - 1e-6) & (ready < cm - 1e-6)
    high = nz & (ready >= cm - 1e-6)
    n_low = low.sum()
    n_high = high.sum()
    w = torch.ones_like(z_out)
    if int(n_low.item()) > 0 and int(n_high.item()) > 0:
        n = (n_low + n_high).to(dtype=z_out.dtype)
        two = z_out.new_tensor(2.0)
        w = torch.where(low, n / (two * n_low.to(dtype=z_out.dtype)), w)
        w = torch.where(high, n / (two * n_high.to(dtype=z_out.dtype)), w)
    return z_out * w


def eval_group_z(head, pooled, ready_count, is_rep, group_z):
    logits = token_logits(head, pooled, ready_count, is_rep)
    dist = Categorical(logits=logits)
    z = group_z.long()
    logp = dist.log_prob(z)
    ent = dist.entropy()
    logp = torch.where(is_rep > 0.5, logp, torch.zeros_like(logp))
    ent = torch.where(is_rep > 0.5, ent, torch.zeros_like(ent))
    return logp, ent
