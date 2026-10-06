"""Group-scoped committed-exploration latent for mat_comm (CT-SDE).

Why this exists
---------------
Independent exploration discovers a k-agent simultaneous action with
probability ~p^k (~0.005 at k=3 with p=1/6). Measured on joint_nav: four 1M
runs ended with joint_captures = 0.00 at EVERY logged point, because the
capture action was extinguished by the miscoordination penalty long before a
single success was ever sampled. Injecting correlated captures during rollout
(`--explore_group_capture_eps`) removed the blocker outright and CT-SDE then
beat obs-sharing MAPPO at every matched update. This module replaces that
hand-injected correlation with a learned one.

What it is
----------
Each GROUP draws a latent z_g ~ Categorical(K) from its own pooled encoder
representation. Embed(z_g) becomes the START token of that group's
autoregressive decode, so every member conditions on the same draw. Group
members are therefore correlated within the step, and (via group-lifetime
holding) across the steps the group persists.

This is MAVEN's committed exploration (Mahajan et al. 2019) scoped to the
coupling structure rather than the whole team: MAVEN's z is global and
per-episode, this one is group-local and group-lifetime, and it doubles as
the decoder start token so it conditions execution rather than only
exploration.

Why it is not the pursuit GO/NO-GO head
---------------------------------------
z here has NO assigned meaning. The head's only input is
`masked_mean_pool(obs_rep, group_mask)` — a tensor. There is no ready_count,
no go_min, no forced honoring, no recruit radius, and no task-defined credit.
What a mode means is learned from the team advantage plus a mutual-information
term, so the same module ports to SMAC (or anything else with groups)
unchanged. `is_rep` is derived from the comm graph structurally, not packed
by the environment.

Training signal (all task-agnostic):
  * policy gradient on log pi(z_g) with the same team advantage as actions
  * entropy bonus on z_g, so it does not collapse early
  * MI(z_g ; the group's joint action), via a discriminator that tries to
    predict z_g from what the group actually did. Without this z collapses:
    every mode produces identical behaviour and the latent does nothing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F

from mat.algorithms.mat.algorithm.ma_transformer import init_

# Decoder token layout, mirroring mat_comm_group_token: the first N_LATENT
# ids are latent START tokens, then a blank START, then the actions.
DEFAULT_N_LATENT = 3


def token_width_extra(n_latent):
    """Decoder token ids reserved before actions: K latents + 1 blank."""
    return int(n_latent) + 1


def blank_token_id(n_latent):
    return int(n_latent)


def action_token_shift(n_latent):
    return int(n_latent) + 1


def structural_is_rep(group_mask):
    """
    Representative of each group = its lowest-index member.

    Derived purely from the comm graph, so no environment packing is needed
    and nothing task-specific enters. The representative is the row whose
    latent draw is counted once and broadcast to the rest of the group.

    group_mask: [B, N, N] bool (diagonal on)
    returns:    [B, N] float, 1.0 where the agent represents its group
    """
    if group_mask.dim() != 3:
        raise ValueError("group_mask must be [B, N, N], got %s"
                         % (tuple(group_mask.shape),))
    n = group_mask.size(-1)
    device = group_mask.device
    ids = torch.arange(n, device=device).view(1, 1, n)
    tagged = torch.where(group_mask, ids, torch.full_like(ids, n))
    lowest = tagged.min(dim=-1).values                      # [B, N]
    own = torch.arange(n, device=device).view(1, n)
    return (lowest == own).to(dtype=torch.float32)


def masked_mean_pool(obs_rep, group_mask):
    """Mean of encoder outputs over each agent's group. [B,N,D] x [B,N,N]."""
    w = group_mask.to(dtype=obs_rep.dtype)
    denom = w.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return torch.matmul(w, obs_rep) / denom


def broadcast_from_rep(values, is_rep, group_mask):
    """Copy each representative's value to every member of its group."""
    batch, n_agent = is_rep.shape
    device = values.device
    ids = torch.arange(n_agent, device=device).view(1, 1, n_agent)
    cand = group_mask & (is_rep > 0.5).unsqueeze(1)
    tagged = torch.where(cand, ids, torch.full_like(ids, n_agent))
    rep_j = tagged.min(dim=-1).values
    has = rep_j < n_agent
    rep_j = rep_j.clamp(max=n_agent - 1)
    gathered = values.gather(1, rep_j)
    return torch.where(has, gathered, torch.zeros_like(gathered))


def groups_changed(prev_mask, cur_mask):
    """
    [B, N] bool: True where an agent's group membership differs from the
    previous step. Used to hold z for a group's LIFETIME — resample only when
    the group actually changes, which is what makes the exploration committed
    over time rather than flickering every step.
    """
    if prev_mask is None:
        return torch.ones(cur_mask.shape[:2], dtype=torch.bool,
                          device=cur_mask.device)
    if prev_mask.shape != cur_mask.shape:
        return torch.ones(cur_mask.shape[:2], dtype=torch.bool,
                          device=cur_mask.device)
    return (prev_mask != cur_mask).any(dim=-1)


class GroupLatentHead(nn.Module):
    """
    Categorical head over K joint-behaviour modes.

    ONLY input is the pooled group representation. That single fact is the
    generalizability argument: nothing task-specific can enter through this
    signature.
    """

    def __init__(self, n_embd, n_latent=DEFAULT_N_LATENT, cond_std=1.0,
                 action_dim=None, bias_std=1.0, span_init=True,
                 bias_norm=True, bias_scale=1.5, bias_floor=1.0):
        super(GroupLatentHead, self).__init__()
        self.n_latent = int(n_latent)
        self.logits = nn.Sequential(
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(n_embd, self.n_latent)),
        )
        # START embedding: seeds the group's autoregressive chain. It passes
        # through a LayerNorm in the decoder, so only its direction matters.
        self.start = nn.Embedding(self.n_latent, n_embd)
        nn.init.normal_(self.start.weight, std=0.02)
        # Conditioning embedding: added to every group member's decoder query.
        #
        # This is needed because CommDecodeBlock carries its residual on
        # `rep_enc`, not on the token stream:
        #     x = ln2(rep_enc + attn2(key=x, value=x, query=rep_enc, ...))
        # so tokens only enter through attn2, which is scaled by
        # decoder_attn_gain (0.01). Measured: overriding the START token moves
        # the token embedding by 3.84 but the output logits by only 7.5e-6.
        # A latent whose job is to correlate actions during EARLY exploration
        # cannot start at ~1% influence, so z also conditions each member's
        # query directly — as in MAVEN, where z conditions every agent's
        # network rather than only a start token.
        #
        # Init scale: the decoder applies LayerNorm right after
        # `rep_enc + cond`, so the effect of this embedding saturates — a
        # sweep over cond_std gave logit deltas 0.006 / 0.019 / 0.028 / 0.036
        # / 0.035 / 0.034 at std 0.1 / 0.3 / 0.5 / 1.0 / 2.0 / 3.0. 1.0 sits
        # at the saturation point. That delta is small in absolute terms but
        # large relative to the initial logit scale (~0.01, i.e. a near-uniform
        # policy), and it is applied IDENTICALLY to every member of a group,
        # so it biases the whole group the same way. If training shows z being
        # ignored, FiLM-style (scale + shift) conditioning is the upgrade —
        # it is not attenuated by the following LayerNorm.
        self.cond = nn.Embedding(self.n_latent, n_embd)
        nn.init.normal_(self.cond.weight, std=float(cond_std))
        # Per-mode bias added straight to the action logits.
        #
        # This is what actually makes the latent bite. Measured P(3 agents in
        # a group pick the same action), against an independent baseline of
        # 0.02778:
        #     rep_enc+cond (before LayerNorm)   0.02779
        #     hidden+cond  (after every block)  0.02779   <- LN is NOT the cause
        #     logits+mode_bias                  0.13625   <- ~5x baseline
        # Injecting after all LayerNorms changes nothing, so the attenuation
        # was never normalization: the path was simply too weak. A direct
        # logit bias is also the more faithful reading of MAVEN, which
        # conditions each agent's action-value network on z.
        #
        # Task-agnostic by construction: a learned [K, action_dim] table with
        # no privileged index. Which action a mode favours is decided by the
        # return and the MI term, exactly as before. The task-SPECIFIC version
        # would be "if mode == 1: boost CAPTURE"; this is not that.
        # Init must guarantee the modes SPAN the action space. A random init
        # at a strength large enough to correlate a group makes each mode a
        # near-deterministic single-action choice, and K random draws over A
        # actions can simply miss the action that matters: measured with
        # K=3, A=6, bias_std=8 the learned table was
        #   mode argmax = [4, 2, 0], P(capture) per mode = [0.0, 0.0, 0.004]
        # so CAPTURE was never proposed by any mode and capture_acts started
        # at 0.84 vs 50.0 for uniform. P(no mode covers a given action) is
        # (1-1/A)^K = 0.58 here, i.e. most seeds fail this way.
        #
        # Identity init (mode k prefers action k) makes coverage structural
        # instead of a lottery. Still task-agnostic: nothing names any
        # particular action, the claim is only that modes should be distinct
        # action priors covering the space, and which mode proves useful is
        # learned from the return and the MI term as before.
        self.bias_norm = bool(bias_norm)
        self.bias_scale = float(bias_scale)
        self.bias_floor = float(bias_floor)
        self.action_dim = None if action_dim is None else int(action_dim)
        if self.action_dim is not None:
            self.mode_bias = nn.Embedding(self.n_latent, self.action_dim)
            if span_init:
                w = torch.zeros(self.n_latent, self.action_dim)
                for k_i in range(self.n_latent):
                    w[k_i, k_i % self.action_dim] = float(bias_std)
                with torch.no_grad():
                    self.mode_bias.weight.copy_(w)
                if self.n_latent < self.action_dim:
                    print(
                        "[group_latent] WARNING n_latent=%d < action_dim=%d: "
                        "actions %s are not covered by any mode and will not "
                        "be explored by the latent."
                        % (self.n_latent, self.action_dim,
                           list(range(self.n_latent, self.action_dim)))
                    )
            else:
                nn.init.normal_(self.mode_bias.weight, std=float(bias_std))
        else:
            self.mode_bias = None

    def forward(self, pooled):
        return self.logits(pooled)                           # [B, N, K]

    def start_embedding(self, z):
        return self.start(z.long())                          # [B, N, D]

    def cond_embedding(self, z):
        """Per-member conditioning added to obs_rep. z is already broadcast."""
        return self.cond(z.long())                           # [B, N, D]

    def condition(self, obs_rep, z):
        """obs_rep conditioned on each agent's group latent."""
        return obs_rep + self.cond_embedding(z)

    def action_bias(self, z):
        """Raw per-mode logit bias, [B, N, action_dim]. None if not configured."""
        if self.mode_bias is None:
            return None
        return self.mode_bias(z.long())

    def mode_action(self, z):
        """The action each mode favours, [B, N] long. None if not configured.

        Used by group-correlated exploration: with probability eps a whole
        group commits to this action together. Nothing here names a
        particular action -- it is the argmax of the learned mode table, and
        which action a mode prefers is decided by the return and the MI term.
        """
        if self.mode_bias is None:
            return None
        return self.mode_bias(z.long()).argmax(dim=-1)

    def apply_bias(self, logits, z):
        """
        Add the mode bias to action logits, scaled to the logits' own spread.

        An ABSOLUTE bias cannot work across training: measured on joint_nav,
        bias_std=2 was too weak to flip an already-suppressed capture logit
        (P(all fire | READY) = 0.0000, firing indistinguishable from
        independent), while bias_std=8 was strong enough but froze the policy
        into "stand still or fire" and navigation collapsed (comm degree 1.80
        vs 2.84 for the control). The value that works also drifts, because
        the policy's logit magnitudes grow during training while a fixed bias
        does not.

        Scaling by the logits' own standard deviation makes the bias a
        dimensionless ratio instead: decisive enough to flip a choice, never
        large enough to dominate. `bias_floor` keeps it effective at init,
        where logits are ~0.01 and a purely relative bias would vanish.

        Task-agnostic: this is scale matching. The table is still learned with
        no privileged index, and which action a mode favours still comes from
        the return and the MI term.
        """
        if self.mode_bias is None:
            return logits
        b = self.mode_bias(z.long())
        if self.bias_norm:
            b = b / (b.std(dim=-1, keepdim=True) + 1e-6)
            scale = logits.detach().std(dim=-1, keepdim=True) + self.bias_floor
            b = self.bias_scale * scale * b
        return logits + b


def latent_logits(head, pooled, is_rep):
    """Logits with non-representative rows made inert (they do not sample)."""
    logits = head(pooled)
    dead = is_rep < 0.5
    logits = logits.masked_fill(dead.unsqueeze(-1), -1e10)
    logits = logits.clone()
    logits[..., 0] = torch.where(
        dead, torch.zeros_like(logits[..., 0]), logits[..., 0]
    )
    return logits


def sample_group_latent(head, pooled, is_rep, group_mask, deterministic=False,
                        z_hold=None, z_prev=None):
    """
    Draw one latent per group and broadcast it to the members.

    z_hold: [B] or [B, N] bool. Where True, keep `z_prev` instead of
            resampling — this is the group-lifetime commitment.
    returns z [B,N] long, logp [B,N], entropy [B,N]
    """
    logits = latent_logits(head, pooled, is_rep)
    dist = Categorical(logits=logits)
    z_rep = dist.probs.argmax(dim=-1) if deterministic else dist.sample()

    if z_hold is not None and z_prev is not None:
        hold = z_hold if z_hold.dim() == 2 else z_hold.unsqueeze(-1)
        hold = hold.expand_as(z_rep).bool()
        z_rep = torch.where(hold, z_prev.long(), z_rep)

    logp = dist.log_prob(z_rep)
    ent = dist.entropy()
    keep = is_rep > 0.5
    logp = torch.where(keep, logp, torch.zeros_like(logp))
    ent = torch.where(keep, ent, torch.zeros_like(ent))
    z = broadcast_from_rep(z_rep, is_rep, group_mask)
    return z.long(), logp, ent


def eval_group_latent(head, pooled, is_rep, z):
    """Log-prob and entropy of an already-drawn z, for the PPO update."""
    logits = latent_logits(head, pooled, is_rep)
    dist = Categorical(logits=logits)
    logp = dist.log_prob(z.long())
    ent = dist.entropy()
    keep = is_rep > 0.5
    return (torch.where(keep, logp, torch.zeros_like(logp)),
            torch.where(keep, ent, torch.zeros_like(ent)))


class LatentDiscriminator(nn.Module):
    """
    Variational MI estimator: predict z_g from the group's realized joint
    action. Maximizing MI(z ; joint action) stops z from collapsing to a
    single mode, which is what makes different latents mean different joint
    behaviours. Consumes only pooled one-hot actions — no task features.
    """

    def __init__(self, action_dim, n_latent=DEFAULT_N_LATENT, hidden=64):
        super(LatentDiscriminator, self).__init__()
        self.n_latent = int(n_latent)
        self.net = nn.Sequential(
            nn.LayerNorm(action_dim),
            init_(nn.Linear(action_dim, hidden), activate=True),
            nn.GELU(),
            init_(nn.Linear(hidden, self.n_latent)),
        )

    def forward(self, action_pool):
        return self.net(action_pool)


def group_action_pool(actions_onehot, group_mask):
    """The group's joint-action signature: mean one-hot action over members."""
    w = group_mask.to(dtype=actions_onehot.dtype)
    denom = w.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return torch.matmul(w, actions_onehot) / denom


def latent_mi_terms(disc, actions_onehot, group_mask, is_rep, z):
    """
    Two things, because MI needs both and they act on different parameters.

    Returns (disc_loss, mi_bonus):

      disc_loss  scalar CE. Trains the DISCRIMINATOR to predict z from the
                 group's joint action. Touches only the discriminator.

      mi_bonus   [B, N] detached log q(z | joint action), zero off the
                 representative rows. This is an intrinsic REWARD on the
                 latent: a z that produced a distinguishable joint behaviour
                 scores high, one that produced the same behaviour as every
                 other mode scores low. It enters the latent's advantage.

    Why the split: actions are discrete samples, so `F.one_hot(action)` has no
    gradient path back to the latent head. An earlier version added the CE
    straight to the loss, which trained only the discriminator and left the
    head untouched — a 20x sweep over latent_mi_coef produced bit-identical
    results (usage [0.988, 0.002, 0.010] at every value), which is how the
    disconnect was caught. This is the MAVEN formulation: the variational
    bound reaches the policy as a reward, not as a differentiable loss.
    """
    pooled_a = group_action_pool(actions_onehot, group_mask)
    logits = disc(pooled_a)
    k = logits.size(-1)
    ce = F.cross_entropy(
        logits.reshape(-1, k), z.reshape(-1).long(), reduction="none"
    ).view_as(z)
    keep = (is_rep > 0.5).to(dtype=ce.dtype)
    disc_loss = (ce * keep).sum() / keep.sum().clamp(min=1.0)
    mi_bonus = (-ce.detach()) * keep
    return disc_loss, mi_bonus


def latent_mi_loss(disc, actions_onehot, group_mask, is_rep, z):
    """Discriminator CE only. Kept for the unit tests; see latent_mi_terms."""
    return latent_mi_terms(disc, actions_onehot, group_mask, is_rep, z)[0]


def latent_usage(z, is_rep, n_latent):
    """Fraction of groups choosing each mode — collapse diagnostic."""
    keep = is_rep > 0.5
    if keep.sum() == 0:
        return torch.zeros(n_latent, device=z.device)
    picked = z[keep].long()
    return torch.bincount(picked, minlength=n_latent).float() / picked.numel()
