"""Communication-constrained Multi-Agent Transformer (`mat_comm`).

Stock MAT treats the swarm as one fully connected group: the encoder's
self-attention is dense over all agents (`masked=False` in ma_transformer.py)
and the decoder's mask is purely the autoregressive causal mask. The
environment's communication radius therefore constrains nothing inside the
network — MAT reads every agent's observation and conditions on every earlier
agent's action regardless of who can actually talk to whom.

This variant masks each agent to its 1-hop communication neighborhood
(agents it can see, plus itself). Neighborhoods overlap; they are not a
partition. An agent that cannot talk to j does not attend to j's observation
or condition on j's action. Multi-hop reachability is not assumed to complete
inside one timestep.

Three places had to change, and all three are necessary:

  1. Encoder attention: masked with `group_mask` (raw adjacency, diag on).
     An agent's representation — and therefore its value estimate — aggregates
     only over its 1-hop neighborhood.
  2. Decoder is per-querier: agent i decodes under a mask whose every row
     is nbhd(i) (diag on) ∧ tril. Tokens on that pass have predecessors
     inside nbhd(i). Teacher forcing batches the N queriers; rollout is
     still one sequential step per agent.
  3. Decoder INPUT tokens: `build_action_tokens` on the querier mask, not
     a global right-shift and not one shared neighborhood graph.

Scope of the constraint. By default (`encoder_critic_attn="hard"`) the
critic is constrained too: MAT's encoder produces both `v_loc` and the
decoder's `obs_rep` from one masked pass. That is "N parallel MATs".

`encoder_critic_attn="soft_group_bias"` keeps the actor on that hard pass
and runs the encoder blocks a second time with the same weights but no
hard mask, adding a learned scalar `b` to in-group logits
(`s_ij = q_i k_j / sqrt(d) + b * G_ij`, `b` init 0 so the critic starts
as dense MAT attention). Value head reads the soft stream; decoder still
reads the hard stream. This is CTDE on a shared encoder, not a second net.

Where the comm mask comes from. It is stored in the tail of tiled
`share_obs` / state, with an explicit `comm_mask_kind`:

  bits      last n_agent entries — Aerial Ordering connectivity bits
  pairwise  last n_agent**2 entries — flattened [N, N] adjacency
            (DICG Predator-Prey). The wrapper concatenates
            `get_comm_mask().reshape(-1)` onto `get_state()`.

`comm_from_share_obs` refuses to guess the layout. Default is bits so
Aerial and the existing isolation tests keep working; DICG's train
script forces pairwise and the wrapper raises if it does not see that.

Note this file does NOT reproduce ma_transformer.py's `state_dim = 37` /
`np.zeros(..., 37)` hack. The real state is used, which is what makes the comm
bits reachable in the first place.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from mat.algorithms.mat.algorithm.ma_transformer import init_
from mat.algorithms.utils.mat_comm_act import (
    discrete_autoregressive_act,
    discrete_autoregressive_act_commit,
    discrete_autoregressive_act_group_token,
    discrete_parallel_act,
    discrete_parallel_act_commit,
    discrete_parallel_act_group_token,
)
from mat.algorithms.utils.mat_comm_group_token import (
    GroupTokenHead,
    parse_group_token_pack,
)
from mat.algorithms.utils.mat_comm_commit import parse_commitment_pack
from mat.algorithms.utils.mat_comm_groups import (
    build_group_causal_mask,
    build_group_mask,
    comm_from_share_obs,
)
from mat.algorithms.utils.util import check


class MaskedSelfAttention(nn.Module):
    """Multi-head attention with a per-batch-element boolean attention mask.

    Differs from ma_transformer.SelfAttention only in taking `attn_mask` as an
    argument instead of holding a fixed triangular buffer: the communication
    graph changes every timestep, so the mask cannot be a registered constant.

    `attn_mask` is [B, N, N] bool, True where attention is permitted. Every mask
    built by mat_comm_groups keeps the diagonal set, so no row is ever fully
    masked and softmax cannot produce NaNs.
    """

    def __init__(self, n_embd, n_head, attn_gain=0.01):
        super(MaskedSelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.n_head = n_head
        self.key = init_(nn.Linear(n_embd, n_embd), gain=attn_gain)
        self.query = init_(nn.Linear(n_embd, n_embd), gain=attn_gain)
        self.value = init_(nn.Linear(n_embd, n_embd), gain=attn_gain)
        self.proj = init_(nn.Linear(n_embd, n_embd), gain=attn_gain)

    def forward(self, key, value, query, attn_mask, attn_bias=None):
        B, L, D = query.size()

        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        # Bias before the hard mask so it cannot resurrect a blocked pair.
        if attn_bias is not None:
            att = att + attn_bias.unsqueeze(1)
        att = att.masked_fill(~attn_mask.unsqueeze(1), float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(y)


class CommEncodeBlock(nn.Module):
    """Encoder block whose attention is confined to the 1-hop neighborhood."""

    def __init__(self, n_embd, n_head):
        super(CommEncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MaskedSelfAttention(n_embd, n_head)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, attn_mask, attn_bias=None):
        x = self.ln1(x + self.attn(x, x, x, attn_mask, attn_bias))
        x = self.ln2(x + self.mlp(x))
        return x


class CommDecodeBlock(nn.Module):
    """Decoder block: AR attention only over earlier 1-hop neighbors."""

    def __init__(self, n_embd, n_head, attn_gain=0.01):
        super(CommDecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = MaskedSelfAttention(n_embd, n_head, attn_gain=attn_gain)
        self.attn2 = MaskedSelfAttention(n_embd, n_head, attn_gain=attn_gain)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, rep_enc, causal_mask):
        x = self.ln1(x + self.attn1(x, x, x, causal_mask))
        # Query is the encoder representation, keys/values are action tokens:
        # row i attends to the tokens of earlier 1-hop neighbors j <= i.
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc,
                                          attn_mask=causal_mask))
        x = self.ln3(x + self.mlp(x))
        return x


class CommEncoder(nn.Module):

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, encode_state,
                 critic_attn_mode="hard", critic_z_dim=0, critic_pair_mlp=False,
                 n_commit_targets=0, critic_state_dim=0, n_agent=0,
                 norm_raw_obs=True):
        super(CommEncoder, self).__init__()
        # Centralized critic: when obs is partial, the critic token of agent i
        # also reads agent i's full-view state chunk (from share_obs). The
        # actor stream (obs_rep) never sees it, so execution stays local.
        self.critic_state_dim = int(critic_state_dim)
        self.n_agent = int(n_agent)
        if self.critic_state_dim > 0:
            self.critic_state_encoder = nn.Sequential(
                nn.LayerNorm(self.critic_state_dim),
                init_(nn.Linear(self.critic_state_dim, n_embd), activate=True),
                nn.GELU())

        if critic_attn_mode not in ("hard", "soft_group_bias"):
            raise ValueError(
                "critic_attn_mode must be 'hard' or 'soft_group_bias', got %r"
                % (critic_attn_mode,))

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.encode_state = encode_state
        self.critic_attn_mode = critic_attn_mode
        self.critic_pair_mlp = bool(critic_pair_mlp)
        self.n_commit_targets = int(n_commit_targets)
        if self.critic_pair_mlp:
            if critic_attn_mode != "soft_group_bias":
                raise ValueError("critic_pair_mlp requires soft_group_bias")
            if self.n_commit_targets < 1:
                raise ValueError("critic_pair_mlp requires n_commit_targets >= 1")

        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True),
                                           nn.GELU())
        # LayerNorm over the RAW observation is rank-degenerate when obs_dim is
        # small: for a vector holding only two distinct values it divides out the
        # gap magnitude and leaves only its sign, whatever the dimension. Pickup
        # Consensus is exactly that case (obs = [K/N, N/n_max]), so [0.4,1.0],
        # [0.6,1.0] and [0.8,1.0] all normalise to the same +-1 pattern and K
        # never reaches the decoder. `norm_raw_obs=False` drops it; self.ln
        # (LayerNorm(n_embd), applied right after in forward) still normalises,
        # just after the projection has spread the input over n_embd dims.
        # Default True keeps MAT's original behaviour bit-for-bit.
        obs_layers = [nn.LayerNorm(obs_dim)] if norm_raw_obs else []
        obs_layers += [init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU()]
        self.obs_encoder = nn.Sequential(*obs_layers)

        self.ln = nn.LayerNorm(n_embd)
        # ModuleList, not Sequential: the blocks need the mask as a second arg.
        self.blocks = nn.ModuleList([CommEncodeBlock(n_embd, n_head) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(),
                                  nn.LayerNorm(n_embd), init_(nn.Linear(n_embd, 1)))
        if critic_attn_mode == "soft_group_bias" and not self.critic_pair_mlp:
            # Init 0: critic starts as unmasked MAT attention.
            self.group_bias = nn.Parameter(torch.zeros(1))
        self.critic_z_dim = int(critic_z_dim)
        if self.critic_z_dim > 0:
            self.critic_z_proj = nn.Sequential(
                init_(nn.Linear(n_embd + self.critic_z_dim, n_embd), activate=True),
                nn.GELU(),
            )
        if self.critic_pair_mlp:
            # Last layer 0: starts as dense MAT (same as group_bias=0).
            self.pair_mlp = nn.Sequential(
                nn.Linear(2, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )
            nn.init.zeros_(self.pair_mlp[-1].weight)
            nn.init.zeros_(self.pair_mlp[-1].bias)

    def _own_state(self, state):
        """[B, N, state_dim] tiled share_obs -> [B, N, d]: agent i's own
        full-view chunk (share_obs = N chunks of d, then the comm tail)."""
        n, d = self.n_agent, self.critic_state_dim
        s = state[..., :n * d].reshape(state.shape[0], n, n, d)
        idx = torch.arange(n, device=state.device)
        return s[:, idx, idx, :]

    def _run_blocks(self, x, attn_mask, attn_bias=None):
        for block in self.blocks:
            x = block(x, attn_mask, attn_bias)
        return x

    def _critic_token(self, x, commits):
        """Actor token x plus detached one-hot z. Critic-only; no grad into z."""
        if self.critic_z_dim <= 0:
            return x
        if commits is None:
            raise ValueError("critic_z_dim>0 requires commits")
        z = commits.detach().long().clamp(0, self.critic_z_dim - 1)
        if z.dim() == 3:
            z = z.squeeze(-1)
        z_oh = F.one_hot(z, num_classes=self.critic_z_dim).to(dtype=x.dtype)
        return self.critic_z_proj(torch.cat([x, z_oh], dim=-1))

    def _pair_attn_bias(self, group_mask, commits):
        """MLP([G_ij, same_target(i,j)]) as a [B,N,N] attention bias. Detach z."""
        if commits is None:
            raise ValueError("critic_pair_mlp requires commits")
        z = commits.detach().long()
        if z.dim() == 3:
            z = z.squeeze(-1)
        zi = z.unsqueeze(2)
        zj = z.unsqueeze(1)
        named = (zi < self.n_commit_targets) & (zj < self.n_commit_targets)
        same = (zi == zj) & named
        feat = torch.stack([group_mask.float(), same.to(dtype=group_mask.dtype)], dim=-1)
        return self.pair_mlp(feat).squeeze(-1)

    def forward(self, state, obs, group_mask, need_rep=True, need_v=True, commits=None):
        x = self.state_encoder(state) if self.encode_state else self.obs_encoder(obs)
        x = self.ln(x)

        if (self.critic_attn_mode == "hard" and self.critic_z_dim <= 0
                and not self.critic_pair_mlp and self.critic_state_dim <= 0):
            x = self._run_blocks(x, group_mask)
            return self.head(x), x

        # Actor stream stays hard and z-free. Critic may use detached z.
        obs_rep = self._run_blocks(x, group_mask) if need_rep else None
        if not need_v:
            return None, obs_rep
        x_c = self._critic_token(x, commits)
        if self.critic_state_dim > 0:
            x_c = x_c + self.critic_state_encoder(self._own_state(state))
        if self.critic_attn_mode == "soft_group_bias":
            ones = torch.ones_like(group_mask)
            if self.critic_pair_mlp:
                bias = self._pair_attn_bias(group_mask, commits)
            else:
                bias = self.group_bias * group_mask.float()
            x_critic = self._run_blocks(x_c, ones, bias)
        else:
            x_critic = self._run_blocks(x_c, group_mask)
        return self.head(x_critic), obs_rep


class CommDecoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head,
                 decoder_attn_gain=0.01, token_dim=None, n_commit=0,
                 action_z_rel=False, use_obs_skip=True, action_z_classes=0,
                 norm_raw_obs=True):
        super(CommDecoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.n_commit = int(n_commit)
        self.use_obs_skip = bool(use_obs_skip)
        if token_dim is None:
            token_dim = action_dim + 1
        self.token_dim = int(token_dim)

        self.action_encoder = nn.Sequential(
            init_(nn.Linear(self.token_dim, n_embd, bias=False), activate=True), nn.GELU())
        # MAPPO-style crop → action scores. Off when a group token replaces it.
        if self.use_obs_skip:
            # Same rank-degeneracy as the encoder's raw-obs LayerNorm (see
            # CommEncoder): with --no_raw_obs_layernorm this front LayerNorm is
            # dropped so a 2-D observation survives into the skip MLP too.
            self.obs_skip = nn.Sequential(
                *([nn.LayerNorm(obs_dim)] if norm_raw_obs else []),
                init_(nn.Linear(obs_dim, n_embd), activate=True),
                nn.GELU(),
                nn.LayerNorm(n_embd),
                init_(nn.Linear(n_embd, n_embd), activate=True),
                nn.GELU(),
                nn.LayerNorm(n_embd),
                init_(nn.Linear(n_embd, action_dim)),
            )
        else:
            self.obs_skip = None
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.ModuleList([
            CommDecodeBlock(n_embd, n_head, attn_gain=decoder_attn_gain)
            for _ in range(n_block)
        ])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(),
                                  nn.LayerNorm(n_embd), init_(nn.Linear(n_embd, action_dim)))
        if self.n_commit > 0:
            self.commit_head = nn.Sequential(
                init_(nn.Linear(n_embd + self.n_commit, n_embd), activate=True),
                nn.GELU(),
                nn.LayerNorm(n_embd),
                init_(nn.Linear(n_embd, self.n_commit)),
            )
            z_extra = self.n_commit + (2 if action_z_rel else 0)
            self.action_z_rel = bool(action_z_rel)
            self.action_given_z = nn.Sequential(
                init_(nn.Linear(n_embd + z_extra, n_embd), activate=True),
                nn.GELU(),
                nn.LayerNorm(n_embd),
                init_(nn.Linear(n_embd, action_dim)),
            )
        else:
            self.action_z_rel = False
            az = int(action_z_classes)
            if az > 0:
                self.action_given_z = nn.Sequential(
                    init_(nn.Linear(n_embd + az, n_embd), activate=True),
                    nn.GELU(),
                    nn.LayerNorm(n_embd),
                    init_(nn.Linear(n_embd, action_dim)),
                )

    def hidden(self, tokens, obs_rep, causal_mask, start_override=None,
               start_mask=None):
        x = self.ln(self.action_encoder(tokens))
        if start_override is not None:
            if start_mask is None:
                raise ValueError("start_override requires start_mask")
            if start_override.shape != x.shape:
                raise ValueError(
                    "start_override %s != encoded tokens %s"
                    % (tuple(start_override.shape), tuple(x.shape))
                )
            if start_mask.shape != x.shape[:2]:
                raise ValueError(
                    "start_mask %s != [B, N] %s"
                    % (tuple(start_mask.shape), tuple(x.shape[:2]))
                )
            x = torch.where(
                start_mask.unsqueeze(-1), self.ln(start_override), x
            )
        for block in self.blocks:
            x = block(x, obs_rep, causal_mask)
        return x

    def forward(self, action, obs_rep, obs, causal_mask):
        x = self.hidden(action, obs_rep, causal_mask)
        logits = self.head(x)
        if self.obs_skip is None:
            return logits
        if obs is None:
            raise ValueError("CommDecoder.forward requires obs (MLP skip to logits)")
        return logits + self.obs_skip(obs)

    def forward_with_start(self, action, obs_rep, obs, causal_mask,
                           start_override, start_mask):
        """`forward`, but each group's chain head starts from `start_override`.

        Used by the group-latent arm: the START token of a group's
        autoregressive decode is Embed(z_g), so every member conditions on the
        same draw. Purely additive — `forward` is unchanged.
        """
        x = self.hidden(action, obs_rep, causal_mask,
                        start_override=start_override, start_mask=start_mask)
        logits = self.head(x)
        if self.obs_skip is None:
            return logits
        if obs is None:
            raise ValueError("forward_with_start requires obs (MLP skip to logits)")
        return logits + self.obs_skip(obs)

    def commit_logits(self, hidden, z_prev_onehot):
        return self.commit_head(torch.cat([hidden, z_prev_onehot], dim=-1))

    def action_logits_given_z(self, hidden, z_onehot, rel_xy=None):
        parts = [hidden, z_onehot]
        if rel_xy is not None:
            parts.append(rel_xy)
        return self.action_given_z(torch.cat(parts, dim=-1))


class CommMultiAgentTransformer(nn.Module):
    """MAT restricted to the environment's communication graph.

    Constructor signature matches MultiAgentTransformer so TransformerPolicy can
    build either one interchangeably.
    """

    def __init__(self, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False,
                 comm_mask_kind="bits", encoder_critic_attn="hard",
                 decoder_attn_gain=0.01, use_commitment=False, n_commit_targets=0,
                 entropy_coef=0.01, commit_entropy_coef=0.01,
                 commit_action_blind=False, commit_critic_z=False,
                 commit_critic_pair=False, use_group_token=False,
                 group_token_entropy_coef=0.01, group_token_go_min=None,
                 use_obs_skip=True, use_group_latent=False, n_latent=3,
                 latent_cond_std=1.0, latent_entropy_coef=0.01,
                 latent_bias_std=1.0, latent_span_init=1,
                 latent_bias_norm=1, latent_bias_scale=1.5,
                 latent_bias_floor=1.0, decoder_self_only=False,
                 critic_global_state=False, norm_raw_obs=True):
        super(CommMultiAgentTransformer, self).__init__()

        if action_type != 'Discrete':
            raise NotImplementedError(
                "mat_comm supports discrete actions only; a continuous variant "
                "would need its own decoder-token convention.")
        if dec_actor:
            raise ValueError(
                "mat_comm is incompatible with --dec_actor: a decentralised "
                "per-agent MLP actor has no cross-agent attention to constrain, "
                "so there is nothing for the comm mask to do.")
        if comm_mask_kind not in ("bits", "pairwise"):
            raise ValueError(
                "comm_mask_kind must be 'bits' or 'pairwise', got %r"
                % (comm_mask_kind,))
        if encoder_critic_attn not in ("hard", "soft_group_bias"):
            raise ValueError(
                "encoder_critic_attn must be 'hard' or 'soft_group_bias', got %r"
                % (encoder_critic_attn,))
        min_tail = n_agent if comm_mask_kind == "bits" else n_agent * n_agent
        if state_dim < min_tail:
            raise ValueError(
                "state_dim (%d) too small for comm_mask_kind=%s (need last %d "
                "entries; n_agent=%d)."
                % (state_dim, comm_mask_kind, min_tail, n_agent))
        if encode_state:
            # Isolation would be void. The wrapper tiles ONE global state across
            # the agent axis, so with encode_state every agent's encoder input
            # row is the identical full state -- all positions, the true K, the
            # lot. That is the same class of leak as the global action shift:
            # the information is in the position's own input, not in something
            # it attends to, so no attention mask can remove it. Measured on a
            # 6-agent setup, perturbing one isolated agent's state fields moved
            # every other agent's value by 0.07-0.30 (0.0 with encode_state off).
            raise ValueError(
                "mat_comm cannot use --encode_state: share_obs is one global "
                "state tiled per agent, so every agent would read the full "
                "state directly and the communication masking would be "
                "meaningless. The encoder must run on per-agent observations.")

        self.n_agent = n_agent
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.comm_mask_kind = comm_mask_kind
        self.encoder_critic_attn = encoder_critic_attn
        self.decoder_attn_gain = float(decoder_attn_gain)
        self.use_group_token = bool(use_group_token)
        self.use_obs_skip = bool(use_obs_skip)
        if self.use_group_token and not self.use_obs_skip:
            raise ValueError("group_token=1 requires use_obs_skip=1")
        self.group_token_entropy_coef = float(group_token_entropy_coef)
        if group_token_go_min is not None:
            raise ValueError(
                "group_token_go_min is removed; GO/NO-GO is learned, not masked"
            )
        self.group_token_go_min = None
        if self.use_group_token and comm_mask_kind != "pairwise":
            raise ValueError("use_group_token requires comm_mask_kind=pairwise")
        # Group-scoped committed-exploration latent (CT-SDE contribution).
        # Mutually exclusive with the pursuit group_token and with commitment:
        # all three write the decoder's START, so only one may own it.
        self.use_group_latent = bool(use_group_latent)
        self.n_latent = int(n_latent)
        self.latent_cond_std = float(latent_cond_std)
        self.latent_entropy_coef = float(latent_entropy_coef)
        self.latent_bias_std = float(latent_bias_std)
        self.latent_span_init = bool(int(latent_span_init))
        self.latent_bias_norm = bool(int(latent_bias_norm))
        self.latent_bias_scale = float(latent_bias_scale)
        self.latent_bias_floor = float(latent_bias_floor)
        # Observation-sharing DE control: the encoder keeps the comm mask (each
        # agent still attends to its neighbours' observations) but the decoder
        # gets an identity mask, so no agent conditions on another's action.
        # Same network and capacity -- the ONLY difference from CT-SDE is the
        # action conditioning.
        self.decoder_self_only = bool(decoder_self_only)
        if self.decoder_self_only and (use_group_latent or use_group_token
                                       or use_commitment):
            raise ValueError("decoder_self_only is the plain DE control; it "
                             "cannot be combined with latent/token/commitment")
        if self.use_group_latent:
            if self.use_group_token:
                raise ValueError("group_latent and group_token are mutually exclusive")
            if comm_mask_kind != "pairwise":
                raise ValueError("group_latent requires comm_mask_kind=pairwise")
            if self.n_latent < 2:
                raise ValueError("n_latent must be >= 2, got %d" % self.n_latent)
        self.use_commitment = bool(use_commitment)
        self.n_commit_targets = int(n_commit_targets)
        if self.use_commitment and self.use_group_token:
            raise ValueError("commitment and group_token are mutually exclusive")
        if self.use_commitment and self.use_group_latent:
            raise ValueError("commitment and group_latent are mutually exclusive")
        self.n_commit = (self.n_commit_targets + 1) if self.use_commitment else 0
        if self.use_commitment and self.n_commit_targets < 1:
            raise ValueError("use_commitment requires n_commit_targets >= 1")
        self.entropy_coef = float(entropy_coef)
        self.commit_entropy_coef = float(commit_entropy_coef)
        self.commit_action_blind = bool(commit_action_blind)
        self.commit_critic_z = bool(commit_critic_z)
        self.commit_critic_pair = bool(commit_critic_pair)
        if self.commit_action_blind and not self.use_commitment:
            raise ValueError("commit_action_blind requires use_commitment")
        if self.commit_critic_z and not self.use_commitment:
            raise ValueError("commit_critic_z requires use_commitment")
        if self.commit_critic_pair and not self.use_commitment:
            raise ValueError("commit_critic_pair requires use_commitment")
        if self.commit_critic_z and self.commit_critic_pair:
            raise ValueError(
                "commit_critic_z and commit_critic_pair are mutually exclusive"
            )
        if self.commit_critic_pair and encoder_critic_attn != "soft_group_bias":
            raise ValueError("commit_critic_pair requires soft_group_bias")
        self.critic_needs_commits = self.commit_critic_z or self.commit_critic_pair
        if self.use_commitment and self.entropy_coef == 0.0 and self.commit_entropy_coef != 0.0:
            raise ValueError(
                "commit_entropy_coef=%s requires entropy_coef > 0 "
                "(H_z is folded into the existing entropy bonus)"
                % self.commit_entropy_coef
            )
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device

        print("[mat_comm] grouping=1-hop neighborhood (no transitive closure)")
        print("[mat_comm] decoder=per-agent (querier neighborhood tokens)")
        if self.use_group_token:
            print(
                "[mat_comm] decoder logits = MLP(obs) + AR(z-as-START, obs_rep)  "
                "group_token=GO/NO-GO learned (no ready mask) "
                "token_entropy_coef=%s token_credit=clique_catch "
                "(obs skip ON, loner START=blank, no z-concat, no recruit START)"
                % (self.group_token_entropy_coef,)
            )
        else:
            if self.use_obs_skip:
                print("[mat_comm] decoder logits = MLP(obs) + AR(token, obs_rep)")
            else:
                print("[mat_comm] decoder logits = AR(token, obs_rep)  (obs skip OFF)")
        if self.use_commitment:
            print("[mat_comm] commitment=on n_targets=%d NULL=%d token_dim=%d "
                  "entropy_coef=%s commit_entropy_coef=%s action_blind=%s "
                  "critic_z=%s critic_pair=%s"
                  % (self.n_commit_targets, self.n_commit_targets,
                     (action_dim + 1) + self.n_commit + 3,
                     self.entropy_coef, self.commit_entropy_coef,
                     int(self.commit_action_blind),
                     int(self.commit_critic_z),
                     int(self.commit_critic_pair)))
        print("[mat_comm] comm_mask_kind=%s encoder_critic_attn=%s decoder_attn_gain=%s state_dim=%d n_agent=%d norm_raw_obs=%s"
              % (comm_mask_kind, encoder_critic_attn, decoder_attn_gain, state_dim, n_agent, norm_raw_obs), flush=True)

        crit_d = 0
        if critic_global_state:
            tail = n_agent * n_agent if comm_mask_kind == "pairwise" else n_agent
            g = state_dim - tail
            if g <= 0 or g % n_agent:
                raise ValueError(
                    "critic_global_state: share_obs minus comm tail (%d) is not "
                    "n_agent=%d equal per-agent chunks" % (g, n_agent))
            crit_d = g // n_agent
            print("[mat_comm] critic reads global state: %d dims per agent "
                  "(actor stays on local obs)" % crit_d, flush=True)
        self.critic_state_dim = crit_d
        self.encoder = CommEncoder(
            state_dim, obs_dim, n_block, n_embd, n_head, encode_state,
            critic_state_dim=crit_d, n_agent=n_agent,
            critic_attn_mode=encoder_critic_attn,
            critic_z_dim=self.n_commit if self.commit_critic_z else 0,
            critic_pair_mlp=self.commit_critic_pair,
            n_commit_targets=self.n_commit_targets if self.commit_critic_pair else 0,
            norm_raw_obs=norm_raw_obs,
        )
        from mat.algorithms.utils.mat_comm_commit import commit_token_dim
        from mat.algorithms.utils.mat_comm_group_token import decoder_token_dim

        if self.use_commitment:
            tok_dim = commit_token_dim(action_dim, self.n_commit_targets)
        elif self.use_group_token:
            tok_dim = decoder_token_dim(action_dim)
        else:
            tok_dim = None
        self.decoder = CommDecoder(
            obs_dim, action_dim, n_block, n_embd, n_head,
            decoder_attn_gain=self.decoder_attn_gain,
            token_dim=tok_dim,
            n_commit=self.n_commit,
            action_z_rel=self.commit_action_blind,
            use_obs_skip=self.use_obs_skip,
            action_z_classes=0,
            norm_raw_obs=norm_raw_obs,
        )
        self.group_token_head = (
            GroupTokenHead(n_embd) if self.use_group_token else None
        )
        if self.use_group_latent:
            from mat.algorithms.utils.mat_comm_group_latent import (
                GroupLatentHead,
                LatentDiscriminator,
            )

            self.group_latent_head = GroupLatentHead(
                n_embd, self.n_latent, cond_std=self.latent_cond_std,
                action_dim=action_dim, bias_std=self.latent_bias_std,
                span_init=self.latent_span_init,
                bias_norm=self.latent_bias_norm,
                bias_scale=self.latent_bias_scale,
                bias_floor=self.latent_bias_floor,
            )
            self.latent_disc = LatentDiscriminator(action_dim, self.n_latent)
        else:
            self.group_latent_head = None
            self.latent_disc = None
        if abs(self.decoder_attn_gain - 0.01) > 1e-12:
            n_lin = 0
            for mod in self.decoder.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.orthogonal_(mod.weight, gain=self.decoder_attn_gain)
                    if mod.bias is not None:
                        nn.init.constant_(mod.bias, 0.0)
                    n_lin += 1
            print("[mat_comm] reinit %d decoder Linear layers gain=%s "
                  "(encoder unchanged; attn-only cannot reach Δlogit>0.1)"
                  % (n_lin, self.decoder_attn_gain), flush=True)
        self.to(device)
        self._dec_rollout_s = 0.0
        self._dec_rollout_n = 0
        self._dec_train_s = 0.0
        self._dec_train_n = 0

    def _sync_if_cuda(self):
        try:
            if torch.device(self.tpdv.get("device", "cpu")).type == "cuda":
                torch.cuda.synchronize()
        except (TypeError, RuntimeError):
            pass

    def pop_decode_times(self):
        """Mean decoder wall-clock (seconds) since last pop. Train vs rollout."""
        out = {
            "dec_rollout_s": (
                self._dec_rollout_s / self._dec_rollout_n
                if self._dec_rollout_n else float("nan")
            ),
            "dec_train_s": (
                self._dec_train_s / self._dec_train_n
                if self._dec_train_n else float("nan")
            ),
            "dec_rollout_calls": float(self._dec_rollout_n),
            "dec_train_calls": float(self._dec_train_n),
        }
        self._dec_rollout_s = 0.0
        self._dec_rollout_n = 0
        self._dec_train_s = 0.0
        self._dec_train_n = 0
        return out

    def zero_std(self):
        # Discrete only; kept so the TransformerPolicy interface stays uniform.
        pass

    # ------------------------------------------------------------------
    # Communication graph
    # ------------------------------------------------------------------

    def _comm_from_state(self, state):
        """Communication graph from the share_obs tail. See comm_from_share_obs."""
        return comm_from_share_obs(state, self.n_agent, self.comm_mask_kind)

    def _masks(self, state):
        group_mask = build_group_mask(self._comm_from_state(state))
        return group_mask, build_group_causal_mask(group_mask)

    def _dec_mask(self, group_mask):
        """Decoder mask: the group mask, or identity for the DE control."""
        if not self.decoder_self_only:
            return group_mask
        n = group_mask.size(-1)
        eye = torch.eye(n, device=group_mask.device, dtype=group_mask.dtype)
        return eye.expand_as(group_mask).contiguous()

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------

    def _commit_geom(self, state):
        return parse_commitment_pack(
            state, self.n_agent, self.n_commit_targets, self.comm_mask_kind
        )

    def _food_blind_obs(self, obs):
        """Zero the leading 3*n_targets food triples. Agent triples stay."""
        n_food_dims = 3 * self.n_commit_targets
        if obs.size(-1) < n_food_dims:
            raise ValueError(
                "obs dim %d too small to blind %d food dims"
                % (obs.size(-1), n_food_dims)
            )
        out = obs.clone()
        out[..., :n_food_dims] = 0
        return out

    def _action_obs_rep(self, state, obs, group_mask, obs_rep):
        if not self.commit_action_blind:
            return obs_rep
        _, obs_rep_a = self.encoder(
            state, self._food_blind_obs(obs), group_mask, need_v=False,
        )
        return obs_rep_a

    def _group_token_fields(self, state):
        return parse_group_token_pack(state, self.n_agent, self.comm_mask_kind)

    def forward(self, state, obs, action, available_actions=None,
                commits=None, z_prev=None, group_z=None):
        """Teacher-forced pass used by the PPO update."""
        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv).long()
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = state.shape[0]
        group_mask, causal_mask = self._masks(state)

        if self.use_commitment:
            if commits is None or z_prev is None:
                raise ValueError("commitment forward needs commits and z_prev")
            commits = check(commits).to(**self.tpdv).long()
            z_prev = check(z_prev).to(**self.tpdv).long()
        if self.use_group_token:
            if group_z is None:
                raise ValueError("group_token forward needs group_z")
            group_z = check(group_z).to(**self.tpdv).long()
        v_loc, obs_rep = self.encoder(
            state, obs, group_mask,
            commits=commits if self.critic_needs_commits else None,
        )
        self._sync_if_cuda()
        t0 = time.perf_counter()
        if self.use_group_token:
            ready, _, is_rep, _ = self._group_token_fields(state)
            if self.entropy_coef == 0.0:
                z_scale = 0.0
            else:
                z_scale = self.group_token_entropy_coef / self.entropy_coef
            action_log, entropy, _, log_z = discrete_parallel_act_group_token(
                self.decoder, self.group_token_head, obs_rep, action, group_z,
                group_mask, ready, is_rep, self.action_dim, self.tpdv,
                available_actions, z_entropy_scale=z_scale, obs=obs,
            )
            self._sync_if_cuda()
            self._dec_train_s += time.perf_counter() - t0
            self._dec_train_n += 1
            return action_log, v_loc, entropy, log_z
        if self.use_group_latent:
            if group_z is None:
                raise ValueError("group_latent forward needs group_z")
            from torch.nn import functional as F

            from mat.algorithms.utils.mat_comm_act_latent import (
                discrete_parallel_act_group_latent,
            )
            from mat.algorithms.utils.mat_comm_group_latent import (
                eval_group_latent,
                latent_mi_terms,
                masked_mean_pool,
                structural_is_rep,
            )

            gz = check(group_z).to(**self.tpdv).long()
            if gz.dim() == 3:
                gz = gz.squeeze(-1)
            is_rep = structural_is_rep(group_mask)
            pooled = masked_mean_pool(obs_rep, group_mask)
            log_z, ent_z = eval_group_latent(
                self.group_latent_head, pooled, is_rep, gz
            )
            action_log, entropy = discrete_parallel_act_group_latent(
                self.decoder, self.group_latent_head, obs_rep, obs, action,
                group_mask, gz, self.action_dim, self.tpdv, available_actions,
            )
            # MI(z ; the group's joint action): keeps different modes meaning
            # different joint behaviours instead of collapsing to one.
            act_flat = action.squeeze(-1).long()
            onehot = F.one_hot(act_flat, num_classes=self.action_dim).to(
                dtype=obs_rep.dtype
            )
            mi_disc, mi_bonus = latent_mi_terms(
                self.latent_disc, onehot, group_mask, is_rep, gz
            )
            self._sync_if_cuda()
            self._dec_train_s += time.perf_counter() - t0
            self._dec_train_n += 1
            return action_log, v_loc, entropy, log_z, ent_z, mi_disc, mi_bonus
        if self.use_commitment:
            agent_xy, target_xy, target_alive, visible = self._commit_geom(state)
            if self.entropy_coef == 0.0:
                z_scale = 0.0
            else:
                z_scale = self.commit_entropy_coef / self.entropy_coef
            action_log, entropy, _, _ = discrete_parallel_act_commit(
                self.decoder, obs_rep, obs, action, commits, z_prev, group_mask,
                agent_xy, target_xy, target_alive, visible,
                self.n_commit_targets, self.action_dim, self.tpdv,
                available_actions, z_entropy_scale=z_scale,
                obs_rep_a=self._action_obs_rep(state, obs, group_mask, obs_rep),
                action_z_rel=self.commit_action_blind,
            )
        else:
            action_log, entropy = discrete_parallel_act(
                self.decoder, obs_rep, obs, action,
                self._dec_mask(group_mask), causal_mask,
                batch_size, self.n_agent, self.action_dim, self.tpdv, available_actions)
        self._sync_if_cuda()
        self._dec_train_s += time.perf_counter() - t0
        self._dec_train_n += 1

        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False,
                    z_prev=None, z_hold=None,
                    forced_actions=None, forced_mask=None):
        if forced_mask is not None and (self.use_group_latent or self.use_commitment
                                        or self.use_group_token):
            raise ValueError("forced-prefix decoding is implemented for the "
                             "plain CT-SDE decode path only")
        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = obs.shape[0]
        group_mask, causal_mask = self._masks(state)

        v_loc, obs_rep = self.encoder(
            state, obs, group_mask, need_v=not self.critic_needs_commits,
        )
        self._sync_if_cuda()
        t0 = time.perf_counter()
        if self.use_group_token:
            ready, _, is_rep, _ = self._group_token_fields(state)
            output_action, output_z, output_log, _, _ = (
                discrete_autoregressive_act_group_token(
                    self.decoder, self.group_token_head, obs_rep, group_mask,
                    ready, is_rep, self.action_dim, self.tpdv,
                    available_actions, deterministic, obs=obs,
                )
            )
            self._sync_if_cuda()
            self._dec_rollout_s += time.perf_counter() - t0
            self._dec_rollout_n += 1
            return output_action, output_log, v_loc, output_z
        if self.use_group_latent:
            from mat.algorithms.utils.mat_comm_act_latent import (
                discrete_autoregressive_act_group_latent,
            )
            from mat.algorithms.utils.mat_comm_group_latent import (
                masked_mean_pool,
                sample_group_latent,
                structural_is_rep,
            )

            is_rep = structural_is_rep(group_mask)
            pooled = masked_mean_pool(obs_rep, group_mask)
            # Callers may hand these in flattened over threads*agents; the
            # transformer's batch dimension is threads, so normalize both to
            # [batch_size, n_agent].
            zp = None
            if z_prev is not None:
                zp = check(z_prev).to(**self.tpdv).long()
                if zp.numel() != batch_size * self.n_agent:
                    raise ValueError(
                        "z_prev has %d elements, expected %d"
                        % (zp.numel(), batch_size * self.n_agent)
                    )
                zp = zp.reshape(batch_size, self.n_agent)
            hold_t = None
            if z_hold is not None:
                hold_t = torch.as_tensor(
                    z_hold, device=self.tpdv["device"]
                ).bool()
                if hold_t.numel() == batch_size:
                    hold_t = hold_t.reshape(batch_size, 1).expand(
                        batch_size, self.n_agent
                    )
                elif hold_t.numel() == batch_size * self.n_agent:
                    hold_t = hold_t.reshape(batch_size, self.n_agent)
                else:
                    raise ValueError(
                        "z_hold has %d elements, expected %d or %d"
                        % (hold_t.numel(), batch_size,
                           batch_size * self.n_agent)
                    )
            output_z, _, _ = sample_group_latent(
                self.group_latent_head, pooled, is_rep, group_mask,
                deterministic=deterministic, z_hold=hold_t, z_prev=zp,
            )
            output_action, output_log = discrete_autoregressive_act_group_latent(
                self.decoder, self.group_latent_head, obs_rep, obs, group_mask,
                output_z, self.action_dim, self.tpdv,
                available_actions, deterministic,
            )
            self._sync_if_cuda()
            self._dec_rollout_s += time.perf_counter() - t0
            self._dec_rollout_n += 1
            return output_action, output_log, v_loc, output_z
        if self.use_commitment:
            if z_prev is None:
                z_prev = torch.full(
                    (batch_size, self.n_agent), self.n_commit_targets,
                    dtype=torch.long, device=self.tpdv["device"],
                )
            else:
                z_prev = check(z_prev).to(**self.tpdv).long()
                if z_prev.dim() == 3:
                    z_prev = z_prev.squeeze(-1)
            agent_xy, target_xy, target_alive, visible = self._commit_geom(state)
            hold_t = None
            if z_hold is not None:
                hold_t = torch.as_tensor(z_hold, device=self.tpdv["device"]).bool().view(-1)
                if hold_t.numel() != batch_size:
                    raise ValueError(
                        "z_hold length %d != batch %d" % (hold_t.numel(), batch_size)
                    )
            output_action, output_z, output_log, _, _ = discrete_autoregressive_act_commit(
                self.decoder, obs_rep, obs, group_mask,
                agent_xy, target_xy, target_alive, visible,
                z_prev, self.n_commit_targets, self.action_dim, self.tpdv,
                available_actions, deterministic,
                obs_rep_a=self._action_obs_rep(state, obs, group_mask, obs_rep),
                action_z_rel=self.commit_action_blind,
                z_hold=hold_t,
            )
            if self.critic_needs_commits:
                v_loc, _ = self.encoder(
                    state, obs, group_mask, need_rep=False, commits=output_z,
                )
            self._sync_if_cuda()
            self._dec_rollout_s += time.perf_counter() - t0
            self._dec_rollout_n += 1
            return output_action, output_log, v_loc, output_z
        output_action, output_action_log = discrete_autoregressive_act(
            self.decoder, obs_rep, obs, self._dec_mask(group_mask),
            causal_mask, batch_size, self.n_agent, self.action_dim, self.tpdv,
            available_actions, deterministic,
            forced_actions=forced_actions, forced_mask=forced_mask)
        self._sync_if_cuda()
        self._dec_rollout_s += time.perf_counter() - t0
        self._dec_rollout_n += 1

        return output_action, output_action_log, v_loc

    def get_values(self, state, obs, available_actions=None, commits=None):
        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if self.critic_needs_commits:
            if commits is None:
                raise ValueError("critic-conditioned get_values needs commits")
            commits = check(commits).to(**self.tpdv).long()

        group_mask, _ = self._masks(state)
        v_tot, _ = self.encoder(
            state, obs, group_mask, need_rep=False,
            commits=commits if self.critic_needs_commits else None,
        )
        return v_tot
