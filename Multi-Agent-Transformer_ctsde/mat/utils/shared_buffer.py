import torch
import numpy as np
import torch.nn.functional as F
from mat.utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols


class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space, env_name):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.algo = args.algorithm_name
        self.num_agents = num_agents
        self.env_name = env_name

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape),
                                  dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, act_space.n),
                                             dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.use_commitment = (
            args.algorithm_name == "mat_comm"
            and int(getattr(args, "commitment", 0)) == 1
        )
        if self.use_commitment:
            n_tgt = int(getattr(args, "n_commit_targets", 0))
            if n_tgt < 1:
                raise ValueError("commitment=1 requires n_commit_targets >= 1")
            self.n_commit_targets = n_tgt
            self.commitments = np.zeros(
                (self.episode_length, self.n_rollout_threads, num_agents, 1),
                dtype=np.float32,
            )
            self.commitment_prev = np.full(
                (self.episode_length, self.n_rollout_threads, num_agents, 1),
                n_tgt,
                dtype=np.float32,
            )
        else:
            self.n_commit_targets = 0
            self.commitments = None
            self.commitment_prev = None

        self.use_group_token = (
            args.algorithm_name == "mat_comm"
            and int(getattr(args, "group_token", 0)) == 1
        )
        self.use_group_latent = (
            args.algorithm_name == "mat_comm"
            and int(getattr(args, "group_latent", 0)) == 1
        )
        if self.use_group_token and self.use_commitment:
            raise ValueError("group_token and commitment cannot both be on")
        if self.use_group_latent and (self.use_group_token or self.use_commitment):
            raise ValueError(
                "group_latent is mutually exclusive with group_token/commitment"
            )
        if self.use_group_latent:
            # Same slot as group_tokens. No outcome array: unlike the pursuit
            # GO token, the latent is credited with the ordinary team
            # advantage, not a hand-defined task outcome.
            self.group_tokens = np.zeros(
                (self.episode_length, self.n_rollout_threads, num_agents, 1),
                dtype=np.float32,
            )
            self.group_token_outcome = None
        elif self.use_group_token:
            self.group_tokens = np.zeros(
                (self.episode_length, self.n_rollout_threads, num_agents, 1),
                dtype=np.float32,
            )
            self.group_token_outcome = np.zeros(
                (self.episode_length, self.n_rollout_threads, num_agents, 1),
                dtype=np.float32,
            )
        elif not self.use_group_latent:
            self.group_tokens = None
            self.group_token_outcome = None

        self.step = 0

    def insert(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None,
               commitments=None, commitment_prev=None, group_tokens=None,
               group_token_outcome=None):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()
        if self.use_commitment:
            if commitments is None or commitment_prev is None:
                raise ValueError("commitment buffer insert needs commitments and commitment_prev")
            self.commitments[self.step] = np.asarray(commitments, dtype=np.float32).copy()
            self.commitment_prev[self.step] = np.asarray(commitment_prev, dtype=np.float32).copy()
        if self.use_group_latent:
            if group_tokens is None:
                raise ValueError("group_latent buffer insert needs group_tokens")
            self.group_tokens[self.step] = np.asarray(
                group_tokens, dtype=np.float32
            ).copy()
        if self.use_group_token:
            if group_tokens is None:
                raise ValueError("group_token buffer insert needs group_tokens")
            if group_token_outcome is None:
                raise ValueError("group_token buffer insert needs group_token_outcome")
            self.group_tokens[self.step] = np.asarray(group_tokens, dtype=np.float32).copy()
            self.group_token_outcome[self.step] = np.asarray(
                group_token_outcome, dtype=np.float32
            ).copy()

        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        """
        Insert data into the buffer. This insert function is used specifically for Hanabi, which is turn based.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) denotes indicate whether whether true terminal state or due to episode limit
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def chooseafter_update(self):
        """Copy last timestep data to first index. This method is used for Hanabi."""
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            if self._use_popart or self._use_valuenorm:
                delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                    self.value_preds[step + 1]) * self.masks[step + 1] \
                        - value_normalizer.denormalize(self.value_preds[step])
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                # here is a patch for mpe, whose last step is timeout instead of terminate
                if self.env_name == "MPE" and step == self.rewards.shape[0] - 1:
                    gae = 0

                self.advantages[step] = gae
                self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
            else:
                delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * \
                        self.masks[step + 1] - self.value_preds[step]
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                # here is a patch for mpe, whose last step is timeout instead of terminate
                if self.env_name == "MPE" and step == self.rewards.shape[0] - 1:
                    gae = 0

                self.advantages[step] = gae
                self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length,
                          n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grid(batch_size, num_agents)

        # keep (num_agent, dim)
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        share_obs = share_obs[rows, cols]
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs = obs[rows, cols]
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states = rnn_states[rows, cols]
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        rnn_states_critic = rnn_states_critic[rows, cols]
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        actions = actions[rows, cols]
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:])
            available_actions = available_actions[rows, cols]
        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        value_preds = value_preds[rows, cols]
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        returns = returns[rows, cols]
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        masks = masks[rows, cols]
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        active_masks = active_masks[rows, cols]
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        action_log_probs = action_log_probs[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows, cols]
        if self.use_commitment:
            commits = self.commitments.reshape(-1, *self.commitments.shape[2:])
            commits = commits[rows, cols]
            z_prev = self.commitment_prev.reshape(-1, *self.commitment_prev.shape[2:])
            z_prev = z_prev[rows, cols]
        if self.use_group_latent:
            group_z = self.group_tokens.reshape(-1, *self.group_tokens.shape[2:])
            group_z = group_z[rows, cols]
        if self.use_group_token:
            group_z = self.group_tokens.reshape(-1, *self.group_tokens.shape[2:])
            group_z = group_z[rows, cols]
            z_out = self.group_token_outcome.reshape(
                -1, *self.group_token_outcome.shape[2:]
            )
            z_out = z_out[rows, cols]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            share_obs_batch = share_obs[indices].reshape(-1, *share_obs.shape[2:])
            obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
            rnn_states_batch = rnn_states[indices].reshape(-1, *rnn_states.shape[2:])
            rnn_states_critic_batch = rnn_states_critic[indices].reshape(-1, *rnn_states_critic.shape[2:])
            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices].reshape(-1, *available_actions.shape[2:])
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
            active_masks_batch = active_masks[indices].reshape(-1, *active_masks.shape[2:])
            old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[2:])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])

            if self.use_commitment:
                commits_batch = commits[indices].reshape(-1, *commits.shape[2:])
                z_prev_batch = z_prev[indices].reshape(-1, *z_prev.shape[2:])
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                      value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                      adv_targ, available_actions_batch, commits_batch, z_prev_batch
            elif self.use_group_token:
                group_z_batch = group_z[indices].reshape(-1, *group_z.shape[2:])
                z_out_batch = z_out[indices].reshape(-1, *z_out.shape[2:])
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                      value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                      adv_targ, available_actions_batch, group_z_batch, z_out_batch
            elif self.use_group_latent:
                # No z_out: the latent is credited with adv_targ, which is
                # already yielded above.
                group_z_batch = group_z[indices].reshape(-1, *group_z.shape[2:])
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                      value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                      adv_targ, available_actions_batch, group_z_batch
            else:
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                      value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                      adv_targ, available_actions_batch

    def recurrent_generator_transformer(self, advantages, num_mini_batch, data_chunk_length):
        """Sequential counterpart of feed_forward_generator_transformer, for recurrent policies.

        The feed-forward generator draws a random permutation over all
        (timestep, thread) pairs, which destroys temporal order. A GRU cannot be
        trained through that: its state at t is defined by its state at t-1, and
        after shuffling the previous step is somewhere else entirely.

        This generator samples whole CHUNKS instead: `data_chunk_length`
        consecutive timesteps from one rollout thread. The model replays the
        recurrence forward through each chunk, starting from the hidden state
        stored at the chunk's first step.

        Layout of every yielded tensor is TIME-MAJOR and then flattened, i.e.
        built as [L, mb, num_agents, dim] and returned as [L*mb*num_agents, dim].
        The policy's usual `.reshape(-1, num_agents, dim)` therefore recovers
        [L*mb, num_agents, dim] with time varying slowest -- which is exactly
        what the recurrent forward pass needs to fold back into [L, mb, ...].
        The one exception is rnn_states, which is the INITIAL state per chunk and
        so has no time axis: [mb*num_agents, recurrent_N, hidden].

        masks_batch matters here. A chunk may straddle an episode boundary, and
        the model must zero the incoming hidden state at that point or memory
        bleeds from one episode into the next. masks[t] == 0 marks a step whose
        predecessor terminated; the model applies it before each recurrent step.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]

        if episode_length % data_chunk_length != 0:
            raise ValueError(
                f"episode_length ({episode_length}) must be divisible by "
                f"data_chunk_length ({data_chunk_length}); otherwise the tail of "
                f"each rollout would never be sampled.")

        chunks_per_thread = episode_length // data_chunk_length
        total_chunks = n_rollout_threads * chunks_per_thread

        if total_chunks < num_mini_batch:
            raise ValueError(
                f"{n_rollout_threads} threads x {chunks_per_thread} chunks = "
                f"{total_chunks} chunks, fewer than num_mini_batch "
                f"({num_mini_batch}). Lower num_mini_batch, lower "
                f"data_chunk_length, or raise n_rollout_threads.")

        mini_batch_size = total_chunks // num_mini_batch
        rand = torch.randperm(total_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size]
                   for i in range(num_mini_batch)]

        L = data_chunk_length
        offsets = np.arange(L)

        def gather(source, t_idx, b_idx):
            """[T, threads, N, ...] -> [L, mb, N, ...] -> [L*mb*N, ...]."""
            chunk = source[t_idx, b_idx]
            return chunk.reshape(-1, *source.shape[3:])

        for indices in sampler:
            threads = indices // chunks_per_thread
            starts = (indices % chunks_per_thread) * L

            # [L, mb] index grids: time varies along axis 0, chunk along axis 1.
            t_idx = starts[None, :] + offsets[:, None]
            b_idx = np.broadcast_to(threads[None, :], t_idx.shape)

            share_obs_batch = gather(self.share_obs[:-1], t_idx, b_idx)
            obs_batch = gather(self.obs[:-1], t_idx, b_idx)
            actions_batch = gather(self.actions, t_idx, b_idx)
            value_preds_batch = gather(self.value_preds[:-1], t_idx, b_idx)
            return_batch = gather(self.returns[:-1], t_idx, b_idx)
            masks_batch = gather(self.masks[:-1], t_idx, b_idx)
            active_masks_batch = gather(self.active_masks[:-1], t_idx, b_idx)
            old_action_log_probs_batch = gather(self.action_log_probs, t_idx, b_idx)
            adv_targ = None if advantages is None else gather(advantages, t_idx, b_idx)
            if self.available_actions is not None:
                available_actions_batch = gather(self.available_actions[:-1], t_idx, b_idx)
            else:
                available_actions_batch = None

            # Initial recurrent state of each chunk: no time axis.
            rnn_states_batch = self.rnn_states[starts, threads].reshape(
                -1, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = self.rnn_states_critic[starts, threads].reshape(
                -1, *self.rnn_states_critic.shape[3:])

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch
