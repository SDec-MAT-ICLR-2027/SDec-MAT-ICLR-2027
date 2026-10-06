"""Group-aware action sampling / evaluation for `mat_comm`.

The comm-constrained counterparts of `transformer_act.discrete_autoregreesive_act`
and `discrete_parallel_act`.

Decoder is per-querier (iter 172). Agent i's pass uses a mask whose every
row is nbhd(i) (diagonal on) ∧ tril, so every token predecessor on that
pass is inside nbhd(i). Teacher forcing batches the N queriers (B → B·N).
Rollout stays one sequential step per agent with that agent's mask.

Token space is MAT's original: width `action_dim + 1`, id 0 is the start token
and action `a` is id `a + 1`.

Rollout and training MUST build tokens the same way (querier neighborhood),
or the PPO importance ratio is wrong at the first epoch.

Only the discrete case is implemented — Aerial Ordering is discrete, and a
continuous variant would need its own token convention.
"""

import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from mat.algorithms.utils.mat_comm_groups import (
    build_action_tokens,
    querier_group_masks,
    querier_mask_for,
)

START_TOKEN_ID = 0


def _tokens_to_input(group_mask, actions, action_dim, tpdv):
    """[B, N] actions -> [B, N, action_dim + 1] one-hot decoder input."""
    token_ids = build_action_tokens(group_mask, actions + 1, START_TOKEN_ID)
    return F.one_hot(token_ids, num_classes=action_dim + 1).to(**tpdv)


def decode_logits_per_agent(decoder, obs_rep, obs, actions, group_mask,
                            action_dim, tpdv):
    """Teacher-forced logits under per-querier masks. actions [B, N] -> [B, N, A]."""
    batch, n_agent = actions.shape
    qmask, causal = querier_group_masks(group_mask)
    act_exp = actions.unsqueeze(1).expand(batch, n_agent, n_agent).reshape(
        batch * n_agent, n_agent
    )
    dec_in = _tokens_to_input(qmask, act_exp, action_dim, tpdv)
    emb = obs_rep.size(-1)
    obs_dim = obs.size(-1)
    rep_exp = obs_rep.unsqueeze(1).expand(batch, n_agent, n_agent, emb).reshape(
        batch * n_agent, n_agent, emb
    )
    obs_exp = obs.unsqueeze(1).expand(batch, n_agent, n_agent, obs_dim).reshape(
        batch * n_agent, n_agent, obs_dim
    )
    logit = decoder(dec_in, rep_exp, obs_exp, causal)
    logit = logit.view(batch, n_agent, n_agent, -1)
    idx = torch.arange(n_agent, device=logit.device)
    return logit[:, idx, idx, :]


def discrete_autoregressive_act(decoder, obs_rep, obs, group_mask, causal_mask,
                                batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False,
                                forced_actions=None, forced_mask=None):
    """Sample one action per agent. Each i decodes under nbhd(i) ∧ tril.

    `forced_actions` / `forced_mask` [B, N] (optional): agents with mask True
    take the given action instead of sampling, and every later agent decodes
    conditioned on it (exploration with a partially forced joint action).

    `causal_mask` is unused (kept so callers stay unchanged); the querier
    mask is rebuilt from `group_mask`.
    """
    del causal_mask
    device = tpdv["device"]
    actions = torch.zeros((batch_size, n_agent), dtype=torch.long, device=device)
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long, device=device)
    output_action_log = torch.zeros((batch_size, n_agent, 1), dtype=torch.float32, device=device)

    for i in range(n_agent):
        qmask, qcausal = querier_mask_for(group_mask, i)
        decoder_input = _tokens_to_input(qmask, actions, action_dim, tpdv)
        logit = decoder(decoder_input, obs_rep, obs, qcausal)[:, i, :]
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10

        distri = Categorical(logits=logit)
        if deterministic:
            action = distri.probs.argmax(dim=-1)
        else:
            action = distri.sample()
        if forced_mask is not None:
            action = torch.where(forced_mask[:, i], forced_actions[:, i], action)

        actions[:, i] = action
        output_action[:, i, 0] = action
        output_action_log[:, i, 0] = distri.log_prob(action)

    return output_action, output_action_log


def discrete_parallel_act(decoder, obs_rep, obs, action, group_mask, causal_mask,
                          batch_size, n_agent, action_dim, tpdv,
                          available_actions=None):
    """Teacher-forced evaluation: N queriers batched as B·N."""
    del causal_mask, batch_size
    action = action.squeeze(-1)
    logit = decode_logits_per_agent(
        decoder, obs_rep, obs, action, group_mask, action_dim, tpdv
    )
    if available_actions is not None:
        logit[available_actions == 0] = -1e10

    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    return action_log, entropy


def _expand_bn(x, n_agent):
    batch = x.size(0)
    extra = x.shape[2:]
    return x.unsqueeze(1).expand(batch, n_agent, *x.shape[1:]).reshape(
        batch * n_agent, *x.shape[1:]
    )


def discrete_autoregressive_act_commit(
    decoder, obs_rep, obs, group_mask,
    agent_xy, target_xy, target_alive, visible,
    z_prev, n_targets, action_dim, tpdv,
    available_actions=None, deterministic=False,
    obs_rep_a=None, action_z_rel=False, z_hold=None,
):
    """Rollout: for each i, sample z_i then a_i under nbhd(i)."""
    from mat.algorithms.utils.mat_comm_commit import (
        build_commit_tokens,
        commitment_avail,
        own_commit_rel,
    )
    if obs_rep_a is None:
        obs_rep_a = obs_rep

    batch, n_agent = obs_rep.shape[:2]
    device = tpdv["device"]
    n_z = n_targets + 1
    actions = torch.zeros((batch, n_agent), dtype=torch.long, device=device)
    commits = torch.full((batch, n_agent), n_targets, dtype=torch.long, device=device)
    out_a = torch.zeros((batch, n_agent, 1), dtype=torch.long, device=device)
    out_z = torch.zeros((batch, n_agent, 1), dtype=torch.long, device=device)
    log_a = torch.zeros((batch, n_agent, 1), dtype=torch.float32, device=device)
    log_z = torch.zeros((batch, n_agent, 1), dtype=torch.float32, device=device)
    avail_z = commitment_avail(visible, target_alive)

    for i in range(n_agent):
        qmask, qcausal = querier_mask_for(group_mask, i)
        q_xy = agent_xy[:, i : i + 1, :].expand(-1, n_agent, 2)
        tok_z = build_commit_tokens(
            qmask, actions, commits, None, q_xy, target_xy, target_alive,
            False, action_dim, n_targets, tpdv,
        )
        hid = decoder.hidden(tok_z, obs_rep, qcausal)[:, i, :]
        z_prev_oh = F.one_hot(z_prev[:, i].clamp(0, n_targets), num_classes=n_z).to(**tpdv)
        z_logit_raw = decoder.commit_logits(hid, z_prev_oh)
        z_logit = z_logit_raw.masked_fill(avail_z[:, i, :] < 0.5, -1e10)
        z_dist = Categorical(logits=z_logit)
        z_i = z_dist.probs.argmax(-1) if deterministic else z_dist.sample()
        if z_hold is not None:
            held = z_prev[:, i].clamp(0, n_targets)
            z_i = torch.where(z_hold, held, z_i)
            # Held z can be stale after a food dies. Unmask that class so
            # logp stays finite and PPO does not explode on hold steps.
            z_logit_lp = z_logit.clone()
            rows = torch.arange(batch, device=device)
            z_logit_lp[rows, held] = z_logit_raw[rows, held]
            log_z[:, i, 0] = torch.where(
                z_hold, Categorical(logits=z_logit_lp).log_prob(z_i),
                z_dist.log_prob(z_i),
            )
        else:
            log_z[:, i, 0] = z_dist.log_prob(z_i)
        commits[:, i] = z_i
        out_z[:, i, 0] = z_i

        tok_a = build_commit_tokens(
            qmask, actions, commits, z_i, q_xy, target_xy, target_alive,
            True, action_dim, n_targets, tpdv,
        )
        hid_a = decoder.hidden(tok_a, obs_rep_a, qcausal)[:, i, :]
        z_oh = F.one_hot(z_i, num_classes=n_z).to(**tpdv)
        rel = None
        if action_z_rel:
            rel = own_commit_rel(
                agent_xy, target_xy, target_alive, commits, n_targets
            )[:, i, :].to(**tpdv)
        a_logit = decoder.action_logits_given_z(hid_a, z_oh, rel)
        if available_actions is not None:
            a_logit = a_logit.masked_fill(available_actions[:, i, :] == 0, -1e10)
        a_dist = Categorical(logits=a_logit)
        a_i = a_dist.probs.argmax(-1) if deterministic else a_dist.sample()
        actions[:, i] = a_i
        out_a[:, i, 0] = a_i
        log_a[:, i, 0] = a_dist.log_prob(a_i)

    return out_a, out_z, log_a + log_z, log_a, log_z


def discrete_parallel_act_commit(
    decoder, obs_rep, obs, action, commits, z_prev, group_mask,
    agent_xy, target_xy, target_alive, visible,
    n_targets, action_dim, tpdv, available_actions=None,
    z_entropy_scale=1.0, obs_rep_a=None, action_z_rel=False,
):
    """Teacher-forced z then a. Combined log-prob; H_a + scale * H_z."""
    from mat.algorithms.utils.mat_comm_commit import (
        build_commit_tokens,
        commitment_avail,
        expand_querier_xy,
        own_commit_rel,
    )

    action = action.squeeze(-1)
    commits = commits.squeeze(-1)
    z_prev = z_prev.squeeze(-1)
    batch, n_agent = action.shape
    n_z = n_targets + 1
    qmask, causal = querier_group_masks(group_mask)
    act_exp = action.unsqueeze(1).expand(batch, n_agent, n_agent).reshape(
        batch * n_agent, n_agent
    )
    z_exp = commits.unsqueeze(1).expand(batch, n_agent, n_agent).reshape(
        batch * n_agent, n_agent
    )
    xy_exp = expand_querier_xy(agent_xy, n_agent)
    txy_exp = _expand_bn(target_xy, n_agent)
    talive_exp = _expand_bn(target_alive, n_agent)
    emb = obs_rep.size(-1)
    rep_exp = obs_rep.unsqueeze(1).expand(batch, n_agent, n_agent, emb).reshape(
        batch * n_agent, n_agent, emb
    )
    idx = torch.arange(n_agent, device=action.device)

    tok_z = build_commit_tokens(
        qmask, act_exp, z_exp, None, xy_exp, txy_exp, talive_exp,
        False, action_dim, n_targets, tpdv,
    )
    hid_z = decoder.hidden(tok_z, rep_exp, causal).view(batch, n_agent, n_agent, -1)
    hid_z = hid_z[:, idx, idx, :]
    z_prev_oh = F.one_hot(z_prev.clamp(0, n_targets), num_classes=n_z).to(**tpdv)
    z_logit = decoder.commit_logits(hid_z, z_prev_oh)
    avail_z = commitment_avail(visible, target_alive)
    z_logit = z_logit.masked_fill(avail_z < 0.5, -1e10)
    z_dist = Categorical(logits=z_logit)
    log_z = z_dist.log_prob(commits).unsqueeze(-1)
    ent_z = z_dist.entropy().unsqueeze(-1)

    z_self_exp = commits.reshape(batch * n_agent)
    tok_a = build_commit_tokens(
        qmask, act_exp, z_exp, z_self_exp, xy_exp, txy_exp, talive_exp,
        True, action_dim, n_targets, tpdv,
    )
    if obs_rep_a is None:
        rep_exp_a = rep_exp
    else:
        rep_exp_a = obs_rep_a.unsqueeze(1).expand(
            batch, n_agent, n_agent, emb
        ).reshape(batch * n_agent, n_agent, emb)
    hid_a = decoder.hidden(tok_a, rep_exp_a, causal).view(batch, n_agent, n_agent, -1)
    hid_a = hid_a[:, idx, idx, :]
    z_oh = F.one_hot(commits.clamp(0, n_targets), num_classes=n_z).to(**tpdv)
    rel = None
    if action_z_rel:
        rel = own_commit_rel(
            agent_xy, target_xy, target_alive, commits, n_targets
        ).to(**tpdv)
    a_logit = decoder.action_logits_given_z(hid_a, z_oh, rel)
    if available_actions is not None:
        a_logit = a_logit.masked_fill(available_actions == 0, -1e10)
    a_dist = Categorical(logits=a_logit)
    log_a = a_dist.log_prob(action).unsqueeze(-1)
    ent_a = a_dist.entropy().unsqueeze(-1)
    return log_a + log_z, ent_a + float(z_entropy_scale) * ent_z, log_a, log_z


def _action_logits_given_z(decoder, hid, z, n_z, tpdv):
    z_oh = F.one_hot(z.long().clamp(0, n_z - 1), num_classes=n_z).to(**tpdv)
    return decoder.action_logits_given_z(hid, z_oh)


def _add_obs_skip(decoder, logit, obs):
    if decoder.obs_skip is None:
        return logit
    if obs is None:
        raise ValueError("obs skip is on; group-token act needs obs")
    return logit + decoder.obs_skip(obs)


def discrete_autoregressive_act_group_token(
    decoder, token_head, obs_rep, group_mask,
    ready, is_rep, action_dim, tpdv,
    available_actions=None, deterministic=False, obs=None,
):
    from mat.algorithms.utils.mat_comm_group_token import (
        masked_mean_pool,
        sample_group_z,
        tokens_to_input_z,
    )

    batch, n_agent = obs_rep.shape[:2]
    device = tpdv["device"]
    pooled = masked_mean_pool(obs_rep, group_mask)
    z, log_z, _ = sample_group_z(
        token_head, pooled, ready, is_rep, group_mask, deterministic,
    )
    actions = torch.zeros((batch, n_agent), dtype=torch.long, device=device)
    out_a = torch.zeros((batch, n_agent, 1), dtype=torch.long, device=device)
    log_a = torch.zeros((batch, n_agent, 1), dtype=torch.float32, device=device)
    for i in range(n_agent):
        qmask, qcausal = querier_mask_for(group_mask, i)
        dec_in = tokens_to_input_z(qmask, actions, z, action_dim, tpdv)
        hid = decoder.hidden(dec_in, obs_rep, qcausal)[:, i, :]
        obs_i = None if obs is None else obs[:, i, :]
        logit = _add_obs_skip(decoder, decoder.head(hid), obs_i)
        if available_actions is not None:
            logit = logit.masked_fill(available_actions[:, i, :] == 0, -1e10)
        dist = Categorical(logits=logit)
        a_i = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        actions[:, i] = a_i
        out_a[:, i, 0] = a_i
        log_a[:, i, 0] = dist.log_prob(a_i)
    out_z = z.unsqueeze(-1)
    log_z = log_z.unsqueeze(-1)
    return out_a, out_z, log_a, log_a, log_z


def discrete_parallel_act_group_token(
    decoder, token_head, obs_rep, action, group_z, group_mask,
    ready, is_rep, action_dim, tpdv,
    available_actions=None, z_entropy_scale=1.0, obs=None,
):
    from mat.algorithms.utils.mat_comm_group_token import (
        eval_group_z,
        masked_mean_pool,
        tokens_to_input_z,
    )

    action = action.squeeze(-1)
    z = group_z.squeeze(-1).long()
    pooled = masked_mean_pool(obs_rep, group_mask)
    log_z, ent_z = eval_group_z(
        token_head, pooled, ready, is_rep, z
    )
    batch, n_agent = action.shape
    qmask, causal = querier_group_masks(group_mask)
    act_exp = action.unsqueeze(1).expand(batch, n_agent, n_agent).reshape(
        batch * n_agent, n_agent
    )
    z_exp = z.unsqueeze(1).expand(batch, n_agent, n_agent).reshape(
        batch * n_agent, n_agent
    )
    dec_in = tokens_to_input_z(qmask, act_exp, z_exp, action_dim, tpdv)
    emb = obs_rep.size(-1)
    rep_exp = obs_rep.unsqueeze(1).expand(batch, n_agent, n_agent, emb).reshape(
        batch * n_agent, n_agent, emb
    )
    hid = decoder.hidden(dec_in, rep_exp, causal)
    hid = hid.view(batch, n_agent, n_agent, -1)
    idx = torch.arange(n_agent, device=action.device)
    hid = hid[:, idx, idx, :]
    logit = _add_obs_skip(decoder, decoder.head(hid), obs)
    if available_actions is not None:
        logit = logit.masked_fill(available_actions == 0, -1e10)
    dist = Categorical(logits=logit)
    log_a = dist.log_prob(action).unsqueeze(-1)
    ent_a = dist.entropy().unsqueeze(-1)
    return (
        log_a,
        ent_a + float(z_entropy_scale) * ent_z.unsqueeze(-1),
        log_a,
        log_z.unsqueeze(-1),
    )
