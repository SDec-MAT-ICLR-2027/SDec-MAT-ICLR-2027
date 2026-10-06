"""Recurrent, communication-constrained MAT (`mat_comm_gru`).

`mat_comm` is memoryless: every decision is made from the current observation
alone. That is fine for Aerial Ordering, where the readiness count is fully
observable at the moment it matters, but it does not scale to environments where
an agent has to remember something it saw several steps ago.

This variant inserts a per-agent GRU in front of the encoder:

    h_{i,t-1}  +  obs_{i,t}   ->  GRU  ->  h_{i,t}
                                            |
                          encoder input = [ obs_{i,t} ; h_{i,t} ]

Everything downstream is `mat_comm` unchanged — the encoder and decoder classes
are imported from it rather than re-implemented, so the communication masking
cannot drift between the two variants.

Design points, all deliberate:

  * ONE GRUCell, applied to each agent independently, with a separate hidden
    state per agent. Weight sharing keeps the model team-size independent, which
    is what lets a policy trained at 5 workers run at 8. Per-agent GRU
    parameters would break that and multiply the parameter count by n_agents.

  * The encoder receives [obs ; h], not h alone. The GRU output is a lossy
    summary; the exact-K decision depends on the CURRENT readiness field, and
    routing it through a recurrent bottleneck only to recover it again is a tax
    with no upside. This matches CT-SDE's encoder input on the same benchmark.

  * The GRU never mixes across agents. It sees one agent's own observation
    stream and nothing else, so it does not weaken the isolation guarantee: an
    isolated agent's hidden state remains a function of its own history only.
    It does let an agent remember what it learned while it WAS connected, which
    is correct behaviour for a comm-limited system rather than a leak.

  * masks zero the incoming hidden state at episode boundaries. A training chunk
    can straddle a reset, and without this, memory bleeds across episodes. The
    mask is applied BEFORE the cell, matching MAPPO's convention.

Batch layout. Two shapes reach this model:

    rollout   B = n_rollout_threads, seq_len = None (one step at a time)
    training  B = L * mb, TIME-MAJOR, seq_len = L (a chunk replayed forward)

`recurrent_generator_transformer` in shared_buffer.py produces the time-major
layout; `_gru_forward` folds it back into [L, mb, ...] to run the recurrence.
"""

import torch
import torch.nn as nn

from mat.algorithms.mat.algorithm.mat_comm_transformer import CommDecoder, CommEncoder
from mat.algorithms.utils.mat_comm_act import (
    discrete_autoregressive_act,
    discrete_parallel_act,
)
from mat.algorithms.utils.mat_comm_groups import (
    build_group_causal_mask,
    build_group_mask,
    comm_from_share_obs,
)
from mat.algorithms.utils.util import check


class CommGRUMultiAgentTransformer(nn.Module):
    """mat_comm with per-agent recurrent memory in front of the encoder."""

    def __init__(self, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False,
                 rnn_hidden_dim=64, comm_mask_kind="bits", norm_raw_obs=True):
        super(CommGRUMultiAgentTransformer, self).__init__()

        if action_type != 'Discrete':
            raise NotImplementedError(
                "mat_comm_gru supports discrete actions only.")
        if dec_actor:
            raise ValueError(
                "mat_comm_gru is incompatible with --dec_actor: a per-agent MLP "
                "actor has no cross-agent attention for the comm mask to act on.")
        if comm_mask_kind not in ("bits", "pairwise"):
            raise ValueError(
                "comm_mask_kind must be 'bits' or 'pairwise', got %r"
                % (comm_mask_kind,))
        min_tail = n_agent if comm_mask_kind == "bits" else n_agent * n_agent
        if state_dim < min_tail:
            raise ValueError(
                "state_dim (%d) too small for comm_mask_kind=%s (need last %d "
                "entries; n_agent=%d)."
                % (state_dim, comm_mask_kind, min_tail, n_agent))
        if encode_state:
            # Same reason as mat_comm: share_obs is one global state tiled per
            # agent, so encoding it hands every agent the full state and voids
            # the communication masking entirely.
            raise ValueError(
                "mat_comm_gru cannot use --encode_state: share_obs is one global "
                "state tiled per agent, so every agent would read the full state "
                "directly and the communication masking would be meaningless.")

        self.n_agent = n_agent
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.comm_mask_kind = comm_mask_kind
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device

        print("[mat_comm_gru] comm_mask_kind=%s state_dim=%d n_agent=%d"
              % (comm_mask_kind, state_dim, n_agent), flush=True)

        # Shared weights, per-agent hidden state.
        self.rnn = nn.GRUCell(obs_dim, rnn_hidden_dim)
        self.rnn_norm = nn.LayerNorm(rnn_hidden_dim)

        # Encoder consumes [obs ; h]; the decoder's obs argument is unused on the
        # non-dec_actor path, so it keeps the raw observation width.
        self.encoder = CommEncoder(state_dim, obs_dim + rnn_hidden_dim,
                                   n_block, n_embd, n_head, encode_state=False,
                                   norm_raw_obs=norm_raw_obs)
        self.decoder = CommDecoder(obs_dim, action_dim, n_block, n_embd, n_head,
                                   norm_raw_obs=norm_raw_obs)
        self.to(device)

    def zero_std(self):
        pass

    # ------------------------------------------------------------------
    # Communication graph (identical contract to mat_comm)
    # ------------------------------------------------------------------

    def _comm_from_state(self, state):
        return comm_from_share_obs(state, self.n_agent, self.comm_mask_kind)

    def _masks(self, state):
        group_mask = build_group_mask(self._comm_from_state(state))
        return group_mask, build_group_causal_mask(group_mask)

    # ------------------------------------------------------------------
    # Recurrence
    # ------------------------------------------------------------------

    def _gru_forward(self, obs, rnn_states, masks, seq_len):
        """Run the per-agent GRU and return (encoder_input, final_hidden_states).

        Args:
            obs: [B, N, obs_dim]. When seq_len is not None, B == seq_len * mb and
                the layout is time-major.
            rnn_states: [B, N, H] for a single step, or [mb, N, H] initial states
                for a chunk.
            masks: [B, N, 1]. 0 marks a step whose predecessor terminated.
            seq_len: chunk length L, or None for a single rollout step.

        Returns:
            encoder_input: [B, N, obs_dim + H]
            h_out: [B, N, H] -- for a chunk, the hidden state AFTER each step.
        """
        n_agent, hdim = self.n_agent, self.rnn_hidden_dim

        if seq_len is None:
            # Rollout: one step. Zero the incoming state where the episode ended.
            h_prev = rnn_states * masks
            h = self.rnn(obs.reshape(-1, self.obs_dim),
                         h_prev.reshape(-1, hdim)).view(-1, n_agent, hdim)
            return torch.cat([obs, self.rnn_norm(h)], dim=-1), h

        # Training: replay the chunk. Time varies slowest in the flat batch.
        mb = obs.shape[0] // seq_len
        obs_seq = obs.view(seq_len, mb, n_agent, self.obs_dim)
        mask_seq = masks.view(seq_len, mb, n_agent, 1)

        h = rnn_states.view(mb, n_agent, hdim)
        outs = []
        for t in range(seq_len):
            # Applied before the cell: an episode that ended at t-1 must not
            # carry its memory into t.
            h = h * mask_seq[t]
            h = self.rnn(obs_seq[t].reshape(-1, self.obs_dim),
                         h.reshape(-1, hdim)).view(mb, n_agent, hdim)
            outs.append(h)

        h_all = torch.stack(outs, dim=0).view(seq_len * mb, n_agent, hdim)
        return torch.cat([obs, self.rnn_norm(h_all)], dim=-1), h_all

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------

    def forward(self, state, obs, action, rnn_states, masks, seq_len,
                available_actions=None):
        """Teacher-forced pass over one chunk, used by the PPO update."""
        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = state.shape[0]
        group_mask, causal_mask = self._masks(state)
        encoder_input, _ = self._gru_forward(obs, rnn_states, masks, seq_len)

        v_loc, obs_rep = self.encoder(state, encoder_input, group_mask)
        action_log, entropy = discrete_parallel_act(
            self.decoder, obs_rep, obs, action, group_mask, causal_mask,
            batch_size, self.n_agent, self.action_dim, self.tpdv, available_actions)

        return action_log, v_loc, entropy

    def get_actions(self, state, obs, rnn_states, masks, available_actions=None,
                    deterministic=False):
        """One rollout step; also returns the updated hidden state."""
        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = obs.shape[0]
        group_mask, causal_mask = self._masks(state)
        encoder_input, h_out = self._gru_forward(obs, rnn_states, masks, seq_len=None)

        v_loc, obs_rep = self.encoder(state, encoder_input, group_mask)
        output_action, output_action_log = discrete_autoregressive_act(
            self.decoder, obs_rep, obs, group_mask, causal_mask,
            batch_size, self.n_agent, self.action_dim, self.tpdv,
            available_actions, deterministic)

        return output_action, output_action_log, v_loc, h_out

    def get_values(self, state, obs, rnn_states, masks, seq_len=None,
                   available_actions=None):
        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        group_mask, _ = self._masks(state)
        encoder_input, _ = self._gru_forward(obs, rnn_states, masks, seq_len)
        v_tot, _ = self.encoder(state, encoder_input, group_mask)
        return v_tot
