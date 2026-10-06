import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.agents.comm_groups import build_group_mask


class CommNetAgent(nn.Module):
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
        self.msg_fc = nn.Linear(H, H)
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

        # Pairwise connectivity via the shared builder (bits -> outer-AND;
        # adjacency -> matrix used directly). CommNet mean-pools over NEIGHBOURS
        # only, so strip the self-loop the builder adds.
        M = build_group_mask(comm_mask, self.comm_mask_type, self.comm_symmetric)
        eye = th.eye(N, device=M.device, dtype=th.bool).unsqueeze(0)
        M = M & (~eye)

        M_float = M.float()
        # Masked mean pool: rows with no neighbour get c=0 (numerator is also 0)
        c = th.bmm(M_float, h)  # [B, N, H]
        denom = M_float.sum(dim=-1, keepdim=True).clamp(min=1.0)
        c = c / denom

        # Recurrent CommNet: h^{t+1} = sigma(H h^t + C c^t) -- the communication
        # vector IS the next hidden state, which is what gives the recurrent
        # variant its multi-hop propagation over time (paper Sec. 2.2: "replacing
        # the communication step i ... by a time step t"). Returning the
        # pre-communication h instead would make each agent's recurrent state a
        # function of its own observation history alone. tanh (not relu) keeps the
        # state bounded for re-entry into the GRU and matches the paper's sigma.
        h2 = th.tanh(h + self.msg_fc(c))
        logits = self.head(h2)  # [B, N, A]

        return logits, h2
