import torch as th

from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY


class CommMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.comm_mask_type = getattr(args, "comm_mask_type", "bits")
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)   # [B, N, feat]
        avail_actions = ep_batch["avail_actions"][:, t]   # [B, N, A]
        comm_mask = self._get_comm_mask(ep_batch, t)       # [B, N, 1]

        logits, self.hidden_states = self.agent(agent_inputs, self.hidden_states, comm_mask)

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                logits[avail_actions == 0] = -1e10
            logits = th.nn.functional.softmax(logits, dim=-1)

        return logits  # [B, N, A]

    def init_hidden(self, batch_size):
        h = self.agent.init_hidden()  # [1, H]
        self.hidden_states = h.unsqueeze(0).expand(batch_size, self.n_agents, -1).clone()

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage)
        )

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # [B, N, obs_dim]

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])

        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )

        return th.cat(inputs, dim=-1)  # [B, N, feat]

    def _get_comm_mask(self, batch, t):
        comm_mask = batch["comm_mask"][:, t]
        # Adjacency masks are already [B, N, N] and must pass through untouched.
        # Bit masks are normalised to [B, N, 1] (build_group_mask also tolerates [B, N]).
        if self.comm_mask_type != "adjacency" and comm_mask.dim() == 2:
            comm_mask = comm_mask.unsqueeze(-1)
        return comm_mask

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
