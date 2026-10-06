"""Group-latent action sampling / evaluation for `mat_comm` (CT-SDE).

Identical to `mat_comm_act.discrete_autoregressive_act` /
`discrete_parallel_act` except for one thing: the START token of each group's
autoregressive chain is `Embed(z_g)` instead of a single fixed learned START.
Every member of a group therefore conditions its decode on the same latent
draw, which is what correlates their actions within the step.

Kept in a separate module so `mat_comm_act.py` — shared by the other arms —
is untouched.
"""
from __future__ import annotations

import torch
from torch.distributions import Categorical

from mat.algorithms.utils.mat_comm_act import (
    _tokens_to_input,
    querier_group_masks,
    querier_mask_for,
)
from mat.algorithms.utils.mat_comm_groups import previous_in_component


def chain_head_mask(group_mask):
    """[B, N] bool: True at the first decode position of every group chain."""
    _, has_prev = previous_in_component(group_mask)
    return ~has_prev


def discrete_autoregressive_act_group_latent(
    decoder, latent_head, obs_rep, obs, group_mask, z, action_dim, tpdv,
    available_actions=None, deterministic=False,
):
    """Sample one action per agent, each group's chain seeded by Embed(z_g)."""
    device = tpdv["device"]
    batch_size, n_agent = z.shape
    actions = torch.zeros((batch_size, n_agent), dtype=torch.long, device=device)
    out_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long, device=device)
    out_log = torch.zeros((batch_size, n_agent, 1), dtype=torch.float32, device=device)

    start_emb = latent_head.start_embedding(z)               # [B, N, D]
    # z also conditions every member's decoder query, not only the chain head
    rep_cond = latent_head.condition(obs_rep, z)
    # ...and biases the action logits directly, which is the part that
    # actually raises P(a group picks the same action). See
    # probe_latent_injection.py.
    for i in range(n_agent):
        qmask, qcausal = querier_mask_for(group_mask, i)
        dec_in = _tokens_to_input(qmask, actions, action_dim, tpdv)
        logit = decoder.forward_with_start(
            dec_in, rep_cond, obs, qcausal, start_emb, chain_head_mask(qmask)
        )[:, i, :]
        logit = latent_head.apply_bias(logit, z[:, i])
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10

        distri = Categorical(logits=logit)
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        actions[:, i] = action
        out_action[:, i, 0] = action
        out_log[:, i, 0] = distri.log_prob(action)

    return out_action, out_log


def decode_logits_per_agent_latent(
    decoder, latent_head, obs_rep, obs, actions, group_mask, z, action_dim, tpdv
):
    """Teacher-forced logits under per-querier masks, with latent STARTs."""
    batch, n_agent = actions.shape
    qmask, causal = querier_group_masks(group_mask)
    act_exp = actions.unsqueeze(1).expand(batch, n_agent, n_agent).reshape(
        batch * n_agent, n_agent
    )
    dec_in = _tokens_to_input(qmask, act_exp, action_dim, tpdv)

    emb = obs_rep.size(-1)
    obs_dim = obs.size(-1)
    # z conditions every member's decoder query, matching the sampling path
    rep_cond = latent_head.condition(obs_rep, z)
    rep_exp = rep_cond.unsqueeze(1).expand(batch, n_agent, n_agent, emb).reshape(
        batch * n_agent, n_agent, emb
    )
    obs_exp = obs.unsqueeze(1).expand(batch, n_agent, n_agent, obs_dim).reshape(
        batch * n_agent, n_agent, obs_dim
    )
    start_emb = latent_head.start_embedding(z)               # [B, N, D]
    start_exp = start_emb.unsqueeze(1).expand(
        batch, n_agent, n_agent, emb
    ).reshape(batch * n_agent, n_agent, emb)

    logit = decoder.forward_with_start(
        dec_in, rep_exp, obs_exp, causal, start_exp, chain_head_mask(qmask)
    )
    logit = logit.view(batch, n_agent, n_agent, -1)
    idx = torch.arange(n_agent, device=logit.device)
    out = logit[:, idx, idx, :]
    return latent_head.apply_bias(out, z)


def discrete_parallel_act_group_latent(
    decoder, latent_head, obs_rep, obs, action, group_mask, z, action_dim, tpdv,
    available_actions=None,
):
    """Teacher-forced evaluation with latent STARTs. Returns (log_prob, entropy)."""
    action = action.squeeze(-1)
    logit = decode_logits_per_agent_latent(
        decoder, latent_head, obs_rep, obs, action, group_mask, z,
        action_dim, tpdv,
    )
    if available_actions is not None:
        logit[available_actions == 0] = -1e10
    distri = Categorical(logits=logit)
    return (distri.log_prob(action).unsqueeze(-1),
            distri.entropy().unsqueeze(-1))
