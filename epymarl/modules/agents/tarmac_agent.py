import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.agents.comm_groups import build_group_mask


class TarMACAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        H = int(getattr(args, "hidden_dim", 64))
        self.H = H

        # Communication-mask convention: "bits" (per-agent connectivity, aerial
        # ordering default) or "adjacency" (pairwise [B,N,N] matrix). comm_symmetric
        # only applies in adjacency mode (undirected graphs).
        self.comm_mask_type = getattr(args, "comm_mask_type", "bits")
        self.comm_symmetric = getattr(args, "comm_symmetric", False)

        self.encoder = nn.Linear(input_shape, H)
        self.rnn = nn.GRUCell(H, H)
        self.q_fc = nn.Linear(H, H)
        self.k_fc = nn.Linear(H, H)
        self.v_fc = nn.Linear(H, H)
        self.head = nn.Linear(H, self.n_actions)

    def init_hidden(self):
        return self.encoder.weight.new(1, self.H).zero_()

    def forward(self, inputs, hidden_state, comm_mask):
        """
        inputs:       [B, N, feat]
        hidden_state: [B, N, H]
        comm_mask:    [B, N] / [B, N, 1] per-agent bits ("bits" mode), or a
                      [B, N, N] pairwise adjacency matrix ("adjacency" mode).
        Returns: logits [B, N, A], new_hidden [B, N, H]
        """
        B, N, _ = inputs.shape
        H = self.H

        e = F.relu(self.encoder(inputs))  # [B, N, H]
        h = self.rnn(
            e.reshape(B * N, H), hidden_state.reshape(B * N, H)
        ).view(B, N, H)

        # Pairwise mask via the shared builder (bits -> outer-AND; adjacency ->
        # matrix used directly). The builder always sets the self-loop, so isolated
        # agents still attend to themselves (avoids NaN softmax) — exactly TarMAC's
        # required convention.
        M = build_group_mask(comm_mask, self.comm_mask_type, self.comm_symmetric)

        q = self.q_fc(h)  # [B, N, H]
        k = self.k_fc(h)
        v = self.v_fc(h)

        scores = th.bmm(q, k.transpose(1, 2)) / math.sqrt(H)  # [B, N, N]

        # Double-mask pattern: mask the logits, then re-mask the weights
        scores = scores.masked_fill(~M, -1e9)
        attn = th.softmax(scores, dim=-1)
        attn = attn * M.float()
        denom = attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        attn = attn / denom

        c = th.bmm(attn, v)  # [B, N, H]
        # TarMAC feeds the aggregated message into the policy GRU, so it persists
        # in the recurrent state and conditions the NEXT step's query (the paper
        # predicts q^{t+1} from h^{t+1}, which already contains c^t) -- that is the
        # mechanism by which targeting depends on what was heard. Returning the
        # pre-communication h would cut it. We keep the message same-step rather
        # than the paper's one-step delay, matching commnet_agent.py. tanh keeps
        # the state bounded for re-entry into the GRU.
        h2 = th.tanh(h + c)
        logits = self.head(h2)  # [B, N, A]

        return logits, h2
