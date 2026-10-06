import torch
import numpy as np
from mat.utils.util import update_linear_schedule
from mat.utils.util import get_shape_from_obs_space, get_shape_from_act_space
from mat.algorithms.utils.util import check
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer

# Algorithms that carry a hidden state between timesteps. These take the
# sequential data generator and receive/return real rnn_states; every other
# algorithm keeps the original behaviour, where rnn_states are inert zeros.
RECURRENT_ALGORITHMS = ("mat_comm_gru",)


class TransformerPolicy:
    """
    MAT Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, num_agents, device=torch.device("cpu")):
        self.device = device
        self.algorithm_name = args.algorithm_name
        self.lr = args.lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self._use_policy_active_masks = args.use_policy_active_masks
        if act_space.__class__.__name__ == 'Box':
            self.action_type = 'Continuous'
        else:
            self.action_type = 'Discrete'

        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        self.share_obs_dim = get_shape_from_obs_space(cent_obs_space)[0]
        if self.action_type == 'Discrete':
            self.act_dim = act_space.n
            self.act_num = 1
        else:
            print("act high: ", act_space.high)
            self.act_dim = act_space.shape[0]
            self.act_num = self.act_dim

        print("obs_dim: ", self.obs_dim)
        print("share_obs_dim: ", self.share_obs_dim)
        print("act_dim: ", self.act_dim)

        self.num_agents = num_agents
        self.tpdv = dict(dtype=torch.float32, device=device)

        # Recurrent variants keep a hidden state between steps, so they need the
        # sequential data generator and the rnn_states plumbing below. Everything
        # else gets the original feed-forward path untouched.
        self.is_recurrent = self.algorithm_name in RECURRENT_ALGORITHMS

        if self.algorithm_name in ["mat", "mat_dec", "mappo"]:
            from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer as MAT
        elif self.algorithm_name == "mat_comm":
            from mat.algorithms.mat.algorithm.mat_comm_transformer import \
                CommMultiAgentTransformer as MAT
        elif self.algorithm_name == "mat_comm_gru":
            from mat.algorithms.mat.algorithm.mat_comm_gru_transformer import \
                CommGRUMultiAgentTransformer as MAT
        elif self.algorithm_name == "mat_gru":
            from mat.algorithms.mat.algorithm.mat_gru import MultiAgentGRU as MAT
        elif self.algorithm_name == "mat_decoder":
            from mat.algorithms.mat.algorithm.mat_decoder import MultiAgentDecoder as MAT
        elif self.algorithm_name == "mat_encoder":
            from mat.algorithms.mat.algorithm.mat_encoder import MultiAgentEncoder as MAT
        else:
            raise NotImplementedError

        mat_kwargs = dict(n_block=args.n_block, n_embd=args.n_embd, n_head=args.n_head,
                          encode_state=args.encode_state, device=device,
                          action_type=self.action_type, dec_actor=args.dec_actor,
                          share_actor=args.share_actor)
        # The raw-obs LayerNorm is rank-degenerate on a low-dimensional
        # observation (see Encoder in ma_transformer.py): on Pickup Consensus
        # obs=[K/N, N/n_max] collapses to the same +-1 vector for every K.
        # Honour --no_raw_obs_layernorm on every arm, so a mat_dec baseline can
        # be matched to its mat_comm counterpart.
        if self.algorithm_name in ("mat", "mat_dec", "mappo",
                                   "mat_comm", "mat_comm_gru"):
            mat_kwargs["norm_raw_obs"] = not getattr(
                args, "no_raw_obs_layernorm", False)
        if self.algorithm_name in ("mat_comm", "mat_comm_gru"):
            kind = getattr(args, "comm_mask_kind", None)
            if kind not in ("bits", "pairwise"):
                raise ValueError(
                    "mat_comm requires --comm_mask_kind {bits,pairwise}; "
                    "refusing to guess (bits on DICG concat-obs would silently "
                    "mis-group)."
                )
            mat_kwargs["comm_mask_kind"] = kind
            mode = getattr(args, "encoder_critic_attn", None)
            if self.algorithm_name == "mat_comm":
                if mode not in ("hard", "soft_group_bias"):
                    raise ValueError(
                        "mat_comm requires --encoder_critic_attn {hard,soft_group_bias}; "
                        "got %r" % (mode,)
                    )
                mat_kwargs["encoder_critic_attn"] = mode
                mat_kwargs["decoder_attn_gain"] = float(
                    getattr(args, "decoder_attn_gain", 0.01)
                )
                mat_kwargs["critic_global_state"] = bool(
                    int(getattr(args, "critic_global_state", 0) or 0)
                )
                mat_kwargs["decoder_self_only"] = bool(
                    int(getattr(args, "decoder_self_only", 0) or 0)
                )
                raw_skip = getattr(args, "use_obs_skip", 1)
                if raw_skip is None or int(raw_skip) not in (0, 1):
                    raise ValueError(
                        "use_obs_skip must be 0 or 1, got %r" % raw_skip
                    )
                mat_kwargs["use_obs_skip"] = int(raw_skip) == 1
                use_c = int(getattr(args, "commitment", 0)) == 1
                n_tgt = int(getattr(args, "n_commit_targets", 0))
                if use_c and n_tgt < 1:
                    raise ValueError("commitment=1 requires n_commit_targets >= 1")
                mat_kwargs["use_commitment"] = use_c
                mat_kwargs["n_commit_targets"] = n_tgt
                if use_c:
                    cz = getattr(args, "commit_entropy_coef", None)
                    if cz is None:
                        raise ValueError(
                            "commitment=1 requires --commit_entropy_coef "
                            "(no silent default; Step 1 used the combined 0.01)"
                        )
                    mat_kwargs["entropy_coef"] = float(args.entropy_coef)
                    mat_kwargs["commit_entropy_coef"] = float(cz)
                    blind = getattr(args, "commit_action_blind", 0)
                    if int(blind) not in (0, 1):
                        raise ValueError(
                            "commit_action_blind must be 0 or 1, got %r" % blind
                        )
                    mat_kwargs["commit_action_blind"] = int(blind) == 1
                    czc = int(args.commit_critic_z)
                    if czc not in (0, 1):
                        raise ValueError(
                            "commit_critic_z must be 0 or 1, got %r" % czc
                        )
                    ccp = int(args.commit_critic_pair)
                    if ccp not in (0, 1):
                        raise ValueError(
                            "commit_critic_pair must be 0 or 1, got %r" % ccp
                        )
                    if czc == 1 and ccp == 1:
                        raise ValueError(
                            "commit_critic_z and commit_critic_pair are "
                            "mutually exclusive"
                        )
                    mat_kwargs["commit_critic_z"] = czc == 1
                    mat_kwargs["commit_critic_pair"] = ccp == 1
                    self.commit_critic_z = czc == 1
                    self.commit_critic_pair = ccp == 1
                    self.critic_needs_commits = czc == 1 or ccp == 1
                self.use_commitment = use_c
                self.n_commit_targets = n_tgt
                self.n_commit = n_tgt + 1 if use_c else 0
                raw_gt = getattr(args, "group_token", 0)
                if raw_gt is None:
                    raise ValueError("group_token cannot be None")
                if int(raw_gt) not in (0, 1):
                    raise ValueError("group_token must be 0 or 1, got %r" % raw_gt)
                use_gt = int(raw_gt) == 1
                if use_gt and use_c:
                    raise ValueError("group_token and commitment are mutually exclusive")
                if use_gt:
                    gte = getattr(args, "group_token_entropy_coef", None)
                    if gte is None:
                        raise ValueError(
                            "group_token=1 requires --group_token_entropy_coef "
                            "(no silent default)"
                        )
                    cm = getattr(args, "catch_min", None)
                    if cm is None:
                        raise ValueError("group_token=1 requires catch_min")
                    mat_kwargs["use_group_token"] = True
                    mat_kwargs["group_token_entropy_coef"] = float(gte)
                    mat_kwargs["entropy_coef"] = float(args.entropy_coef)
                    if int(getattr(args, "use_obs_skip", 1)) == 0:
                        raise ValueError("group_token=1 requires --use_obs_skip 1")
                self.use_group_token = use_gt
                # Group-scoped committed-exploration latent (CT-SDE).
                raw_gl = int(getattr(args, "group_latent", 0) or 0)
                if raw_gl not in (0, 1):
                    raise ValueError("group_latent must be 0 or 1, got %r" % raw_gl)
                use_gl = raw_gl == 1
                if use_gl and (use_gt or use_c):
                    raise ValueError(
                        "group_latent is mutually exclusive with group_token "
                        "and commitment (all three own the decoder START)"
                    )
                if use_gl:
                    mat_kwargs["use_group_latent"] = True
                    mat_kwargs["n_latent"] = int(getattr(args, "n_latent", 3))
                    mat_kwargs["latent_cond_std"] = float(
                        getattr(args, "latent_cond_std", 1.0)
                    )
                    mat_kwargs["latent_entropy_coef"] = float(
                        getattr(args, "latent_entropy_coef", 0.01)
                    )
                    mat_kwargs["latent_bias_std"] = float(
                        getattr(args, "latent_bias_std", 1.0)
                    )
                    mat_kwargs["latent_span_init"] = int(
                        getattr(args, "latent_span_init", 1)
                    )
                    mat_kwargs["latent_bias_norm"] = int(
                        getattr(args, "latent_bias_norm", 1)
                    )
                    mat_kwargs["latent_bias_scale"] = float(
                        getattr(args, "latent_bias_scale", 1.5)
                    )
                    mat_kwargs["latent_bias_floor"] = float(
                        getattr(args, "latent_bias_floor", 1.0)
                    )
                    mat_kwargs["entropy_coef"] = float(args.entropy_coef)
                self.use_group_latent = use_gl
                self.n_latent = int(getattr(args, "n_latent", 3))
                self.latent_mi_coef = float(getattr(args, "latent_mi_coef", 0.1))
                self.latent_entropy_coef = float(
                    getattr(args, "latent_entropy_coef", 0.01)
                )
            elif mode not in (None, "hard"):
                raise ValueError(
                    "encoder_critic_attn=%r is not implemented for %s"
                    % (mode, self.algorithm_name)
                )
        if self.is_recurrent:
            # Must match the buffer's rnn_states allocation, which is sized by
            # args.hidden_size.
            mat_kwargs["rnn_hidden_dim"] = args.hidden_size

        if not hasattr(self, "use_group_token"):
            self.use_group_token = False
        if not hasattr(self, "use_group_latent"):
            self.use_group_latent = False
            self.n_latent = 0
            self.latent_mi_coef = 0.0
            self.latent_entropy_coef = 0.0
        if not hasattr(self, "use_commitment"):
            self.use_commitment = False
            self.n_commit_targets = 0
            self.n_commit = 0
        if not hasattr(self, "commit_critic_z"):
            self.commit_critic_z = False
        if not hasattr(self, "commit_critic_pair"):
            self.commit_critic_pair = False
        if not hasattr(self, "critic_needs_commits"):
            self.critic_needs_commits = False
        self.transformer = MAT(self.share_obs_dim, self.obs_dim, self.act_dim, num_agents,
                               **mat_kwargs)
        if args.env_name == "hands":
            self.transformer.zero_std()

        # count the volume of parameters of model
        # Total_params = 0
        # Trainable_params = 0
        # NonTrainable_params = 0
        # for param in self.transformer.parameters():
        #     mulValue = np.prod(param.size())
        #     Total_params += mulValue
        #     if param.requires_grad:
        #         Trainable_params += mulValue
        #     else:
        #         NonTrainable_params += mulValue
        # print(f'Total params: {Total_params}')
        # print(f'Trainable params: {Trainable_params}')
        # print(f'Non-trainable params: {NonTrainable_params}')

        self.optimizer = torch.optim.Adam(self.transformer.parameters(),
                                          lr=self.lr, eps=self.opti_eps,
                                          weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False, z_prev=None, z_hold=None,
                    forced_actions=None, forced_mask=None):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """

        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        if self.is_recurrent:
            rnn_in = check(rnn_states_actor).to(**self.tpdv).reshape(
                -1, self.num_agents, self.transformer.rnn_hidden_dim)
            masks_in = check(masks).to(**self.tpdv).reshape(-1, self.num_agents, 1)

            actions, action_log_probs, values, h_out = self.transformer.get_actions(
                cent_obs, obs, rnn_in, masks_in, available_actions, deterministic)

            # Back to the buffer's [threads*agents, recurrent_N, hidden] layout.
            rnn_states_actor = h_out.reshape(-1, 1, self.transformer.rnn_hidden_dim)
            # One GRU serves both heads in MAT; mirror it so the buffer stays
            # consistent and the critic state is never stale.
            rnn_states_critic = rnn_states_actor.clone()
        else:
            if self.use_commitment:
                if z_prev is None:
                    z_prev = np.full(
                        (cent_obs.shape[0], self.num_agents),
                        self.n_commit_targets,
                        dtype=np.int64,
                    )
                else:
                    z_prev = np.asarray(z_prev).reshape(-1, self.num_agents)
                actions, action_log_probs, values, commits = self.transformer.get_actions(
                    cent_obs, obs, available_actions, deterministic, z_prev=z_prev,
                    z_hold=z_hold,
                )
            elif self.use_group_token:
                actions, action_log_probs, values, commits = self.transformer.get_actions(
                    cent_obs, obs, available_actions, deterministic)
            elif self.use_group_latent:
                if z_prev is not None:
                    z_prev = np.asarray(z_prev).reshape(-1, self.num_agents)
                actions, action_log_probs, values, commits = self.transformer.get_actions(
                    cent_obs, obs, available_actions, deterministic,
                    z_prev=z_prev, z_hold=z_hold,
                )
            else:
                kw = {}
                if forced_mask is not None:
                    dev = self.tpdv["device"]
                    kw["forced_actions"] = torch.as_tensor(
                        np.asarray(forced_actions).reshape(-1, self.num_agents),
                        dtype=torch.long, device=dev)
                    kw["forced_mask"] = torch.as_tensor(
                        np.asarray(forced_mask).reshape(-1, self.num_agents),
                        dtype=torch.bool, device=dev)
                actions, action_log_probs, values = self.transformer.get_actions(
                    cent_obs, obs, available_actions, deterministic, **kw)
                commits = None
            # unused, just for compatibility
            rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
            rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)

        actions = actions.view(-1, self.act_num)
        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)

        if self.use_commitment or self.use_group_token or self.use_group_latent:
            commits = commits.view(-1, 1)
            return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, commits
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, obs, rnn_states_critic, masks, available_actions=None,
                    commits=None):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """

        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)
        if commits is not None:
            commits = np.asarray(commits).reshape(-1, self.num_agents)

        if self.is_recurrent:
            rnn_in = check(rnn_states_critic).to(**self.tpdv).reshape(
                -1, self.num_agents, self.transformer.rnn_hidden_dim)
            masks_in = check(masks).to(**self.tpdv).reshape(-1, self.num_agents, 1)
            # Bootstrap value only: a single step, so seq_len is None.
            values = self.transformer.get_values(cent_obs, obs, rnn_in, masks_in,
                                                 seq_len=None,
                                                 available_actions=available_actions)
        else:
            if self.critic_needs_commits:
                values = self.transformer.get_values(
                    cent_obs, obs, available_actions, commits=commits,
                )
            else:
                values = self.transformer.get_values(
                    cent_obs, obs, available_actions,
                )

        values = values.view(-1, 1)

        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, actions, masks,
                         available_actions=None, active_masks=None, seq_len=None,
                         commits=None, z_prev=None, group_z=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param actions: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        actions = actions.reshape(-1, self.num_agents, self.act_num)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        if self.is_recurrent:
            # rnn_states here is the INITIAL state of each chunk: [mb*N, 1, H].
            rnn_in = check(rnn_states_actor).to(**self.tpdv).reshape(
                -1, self.num_agents, self.transformer.rnn_hidden_dim)
            masks_in = check(masks).to(**self.tpdv).reshape(-1, self.num_agents, 1)
            action_log_probs, values, entropy = self.transformer(
                cent_obs, obs, actions, rnn_in, masks_in, seq_len, available_actions)
        else:
            if self.use_commitment:
                commits = commits.reshape(-1, self.num_agents, 1)
                z_prev = z_prev.reshape(-1, self.num_agents, 1)
                action_log_probs, values, entropy = self.transformer(
                    cent_obs, obs, actions, available_actions,
                    commits=commits, z_prev=z_prev,
                )
            elif self.use_group_token:
                group_z = group_z.reshape(-1, self.num_agents, 1)
                action_log_probs, values, entropy, log_z = self.transformer(
                    cent_obs, obs, actions, available_actions, group_z=group_z,
                )
            elif self.use_group_latent:
                group_z = group_z.reshape(-1, self.num_agents, 1)
                (action_log_probs, values, entropy,
                 log_z, ent_z, mi_disc, mi_bonus) = self.transformer(
                    cent_obs, obs, actions, available_actions, group_z=group_z,
                )
            else:
                action_log_probs, values, entropy = self.transformer(
                    cent_obs, obs, actions, available_actions)

        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)
        entropy = entropy.view(-1, self.act_num)

        if self._use_policy_active_masks and active_masks is not None:
            entropy = (entropy*active_masks).sum()/active_masks.sum()
        else:
            entropy = entropy.mean()

        if self.use_group_token:
            log_z = log_z.reshape(-1, 1)
            return values, action_log_probs, entropy, log_z
        if self.use_group_latent:
            return (values, action_log_probs, entropy,
                    log_z.reshape(-1, 1), ent_z.reshape(-1, 1),
                    mi_disc, mi_bonus.reshape(-1, 1))
        return values, action_log_probs, entropy

    def act(self, cent_obs, obs, rnn_states_actor, masks, available_actions=None, deterministic=True,
            z_prev=None, z_hold=None):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """

        # this function is just a wrapper for compatibility
        rnn_states_critic = np.zeros_like(rnn_states_actor)
        out = self.get_actions(cent_obs,
                               obs,
                               rnn_states_actor,
                               rnn_states_critic,
                               masks,
                               available_actions,
                               deterministic,
                               z_prev=z_prev,
                               z_hold=z_hold)
        if self.use_commitment or self.use_group_token:
            _, actions, _, rnn_states_actor, _, commits = out
            return actions, rnn_states_actor, commits
        if self.use_group_latent:
            # get_actions returns the drawn latent in the 6th slot; eval /
            # probes only need the actions, so drop it.
            _, actions, _, rnn_states_actor, _, _z = out
            return actions, rnn_states_actor
        _, actions, _, rnn_states_actor, _ = out
        return actions, rnn_states_actor

    def save(self, save_dir, episode):
        torch.save(self.transformer.state_dict(), str(save_dir) + "/transformer_" + str(episode) + ".pt")

    def restore(self, model_dir):
        transformer_state_dict = torch.load(model_dir, map_location=self.device)
        self.transformer.load_state_dict(transformer_state_dict)
        # self.transformer.reset_std()

    def train(self):
        self.transformer.train()

    def eval(self):
        self.transformer.eval()

