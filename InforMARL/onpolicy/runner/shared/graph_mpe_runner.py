import time
import numpy as np
from numpy import ndarray as arr
from typing import Tuple
import torch
from onpolicy.runner.shared.base_runner import Runner
from onpolicy.utils.ctsde_switches import (
    random_group_commit,
    arm_behaviour_weights,
)
import wandb
import imageio


def _t2n(x):
    return x.detach().cpu().numpy()


class GMPERunner(Runner):
    """
    Runner class to perform training, evaluation and data
    collection for the MPEs. See parent class for details
    """

    dt = 0.1

    def __init__(self, config):
        super(GMPERunner, self).__init__(config)
        a = self.all_args
        # --- the three CT-SDE switches (see onpolicy/utils/ctsde_switches.py)
        self.commit_groups = str(getattr(a, "commit_groups", "comm"))
        self.group_commit_eps = float(getattr(a, "group_commit_eps", 0.0) or 0.0)
        self.commit_member_prob = float(getattr(a, "commit_member_prob", 1.0))
        self.offpolicy_weight = int(getattr(a, "offpolicy_weight", 0) or 0)
        self.behav_w_mode = str(getattr(a, "behav_w_mode", "marginal"))
        self.behav_w_floor = float(getattr(a, "behav_w_floor", 0.0) or 0.0)
        self._eps0 = self.group_commit_eps
        self._eps_final = float(getattr(a, "group_commit_eps_final", -1.0))
        self._eps_decay_frac = float(getattr(a, "group_commit_decay_frac", 1.0))
        self._behav_w = None
        self._randcommit_groups = 0

        # Same coupling rules the MAT fork enforces, so a command that is
        # rejected there is rejected here for the same reason.
        if self.offpolicy_weight and self.group_commit_eps <= 0.0:
            raise ValueError(
                "offpolicy_weight=1 needs group_commit_eps > 0: with no "
                "exploration the data are on-policy and every weight is 1")
        if self.commit_member_prob < 1.0 and self.behav_w_mode != "arm":
            raise ValueError("commit_member_prob < 1 needs --behav_w_mode arm")
        if self._eps_final >= 0.0 and self._eps0 <= 0.0:
            raise ValueError("group_commit_eps_final needs group_commit_eps > 0")
        if not 0.0 <= self.behav_w_floor <= 1.0:
            raise ValueError("behav_w_floor must be in [0, 1]")
        if self.commit_groups != "comm" and self.group_commit_eps > 0.0:
            raise ValueError("only commit_groups=comm is ported to InforMARL")
        if self.group_commit_eps > 0.0 and getattr(a, "obs_mode", "") != "radius":
            # grouping reads the radius obs visibility flags
            raise ValueError("group_commit_eps > 0 needs --obs_mode radius")

    def _decayed_eps(self, episode, episodes):
        """Linear decay of group_commit_eps, matching the MAT fork."""
        if self._eps_final < 0.0 or self._eps0 <= 0.0:
            return self._eps0
        frac = min(1.0, (episode / max(episodes - 1, 1)) / max(self._eps_decay_frac, 1e-9))
        return self._eps0 + frac * (self._eps_final - self._eps0)

    def _relogprob(self, step, actions):
        """Teacher-forced log-probs of the EXECUTED actions, so the PPO ratio
        starts at 1. Values come from the critic and do not depend on the
        action, so only the log-probs are recomputed."""
        _, logp, _ = self.trainer.policy.evaluate_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.node_obs[step]),
            np.concatenate(self.buffer.adj[step]),
            np.concatenate(self.buffer.agent_id[step]),
            np.concatenate(self.buffer.share_agent_id[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(actions),
            np.concatenate(self.buffer.masks[step]),
        )
        return np.array(np.split(_t2n(logp), self.n_rollout_threads))

    def _store_behaviour_weights(self, step, logp, acts, groups_all, eps):
        if self._behav_w is None:
            self._behav_w = np.ones_like(self.buffer.action_log_probs)
        n_act = self.envs.action_space[0].n
        w = np.ones_like(logp)
        for th in range(self.n_rollout_threads):
            # ar=False: InforMARL's actor is independent, so the behaviour
            # marginal applies (identical to the MAT fork's mappo arm).
            w[th, :, 0] = arm_behaviour_weights(
                logp[th, :, 0], acts[th, :, 0], groups_all[th],
                eps, self.commit_member_prob, n_act, ar=False)
        if self.behav_w_floor > 0.0:
            w = np.maximum(w, self.behav_w_floor)
        self._behav_w[step] = w
        self.buffer.behav_w = self._behav_w

    def run(self):
        self.warmup()

        start = time.time()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        # This is where the episodes are actually run.
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            # exploration strength for this episode (constant unless decaying)
            self._cur_eps = self._decayed_eps(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Obs reward and next obs
                obs, agent_id, node_obs, adj, rewards, dones, infos = self.envs.step(
                    actions_env
                )

                data = (
                    obs,
                    agent_id,
                    node_obs,
                    adj,
                    agent_id,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()

                env_infos = self.process_infos(infos)

                avg_ep_rew = np.mean(self.buffer.rewards) * self.episode_length
                train_infos["average_episode_rewards"] = avg_ep_rew
                # MAT-fork-compatible aliases so InforMARL shares the same
                # wandb charts as mat_comm_soft / mat_comm_dec / mappo.
                # The scenario reward is summed and shared across agents
                # (`MultiAgentGraphEnv.step`), and the MAT env records the same
                # shared value, so this equals MAT's `return_mean` exactly.
                train_infos["return_mean"] = avg_ep_rew
                train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
                if self._eps0 > 0.0:
                    n_eps_done = self.log_interval * self.n_rollout_threads
                    env_infos["random_committed_groups"] = [
                        self._randcommit_groups / max(n_eps_done, 1)]
                    env_infos["group_commit_eps"] = [
                        float(getattr(self, "_cur_eps", self._eps0))]
                    self._randcommit_groups = 0
                print(
                    f"Average episode rewards is {avg_ep_rew:.3f} \t"
                    f"Total timesteps: {total_num_steps} \t "
                    f"Percentage complete {total_num_steps / self.num_env_steps * 100:.3f}"
                )
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, agent_id, node_obs, adj = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            # (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            # (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
            share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
            share_agent_id = np.expand_dims(share_agent_id, 1).repeat(
                self.num_agents, axis=1
            )
        else:
            share_obs = obs
            share_agent_id = agent_id

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.node_obs[0] = node_obs.copy()
        self.buffer.adj[0] = adj.copy()
        self.buffer.agent_id[0] = agent_id.copy()
        self.buffer.share_agent_id[0] = share_agent_id.copy()

    @torch.no_grad()
    def _obs_prefix_len(self):
        """Length of the own-state prefix of a radius observation."""
        return 4 if bool(int(getattr(self.all_args, "obs_abs_pos", 1) or 0)) else 2

    def _census_ablated(self):
        """True when the actor observation no longer shows the true comm graph."""
        a = self.all_args
        r = getattr(a, "obs_radius", None)
        return ((r is not None and float(r) > 0.0
                 and float(r) < float(getattr(a, "sense_radius", 1.0)))
                or not bool(int(getattr(a, "obs_abs_pos", 1) or 0)))

    def _commit_adj(self, step):
        """True N x N agent adjacency for the group-commit scaffold.

        Taken from the env's own distance matrix (`world.cached_dist_mag`,
        carried in buffer.adj) thresholded at sense_radius -- exactly the
        scenario's `agent_radius_adjacency`. Entities are ordered agents,
        then landmarks, then obstacles, so the agent block is the leading nxn.

        This is TRAINING machinery and deliberately does NOT read the actor
        observation: under a census ablation the observation's peer flags no
        longer equal the comm graph, and deriving groups from them would
        silently switch correlated exploration off in exactly the condition
        under test.
        """
        n = self.num_agents
        adj = getattr(self.buffer, "adj", None)
        if adj is None:
            if self._census_ablated():
                raise RuntimeError(
                    "census is ablated but buffer.adj is unavailable, so the "
                    "group-commit scaffold would fall back to the actor "
                    "observation and differ between conditions")
            return None
        arr_ = np.asarray(adj[step])
        if arr_.ndim == 4:        # [threads, agents, nodes, nodes]
            arr_ = arr_[:, 0]
        if arr_.shape[-1] < n or arr_.shape[-2] < n:
            raise RuntimeError("buffer.adj node block %s smaller than n_agents %d"
                               % (arr_.shape, n))
        R = float(getattr(self.all_args, "sense_radius", 1.0))
        return arr_[:, :n, :n] <= R

    def collect(self, step: int) -> Tuple[arr, arr, arr, arr, arr, arr]:
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.node_obs[step]),
            np.concatenate(self.buffer.adj[step]),
            np.concatenate(self.buffer.agent_id[step]),
            np.concatenate(self.buffer.share_agent_id[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )

        # ---- exploration switch: subset commit over radius groups.
        # Identical mechanism to the MAT fork, so every arm explores the same
        # way and any remaining gap is about execution, not exploration.
        eps = getattr(self, "_cur_eps", self.group_commit_eps)
        if eps > 0.0:
            n_act = self.envs.action_space[0].n
            rc_actions, n_r, groups_all, _forced = random_group_commit(
                self.buffer.obs[step], actions, self.num_agents, n_act,
                eps, self.commit_member_prob,
                adj=self._commit_adj(step), off=self._obs_prefix_len())
            if n_r:
                self._randcommit_groups += n_r
                # Re-score the EXECUTED actions so the PPO ratio starts at 1.
                # InforMARL's actor is independent, so (unlike the AR arm)
                # nothing needs re-decoding: overriding one agent's action does
                # not change any other agent's distribution.
                action_log_probs = self._relogprob(step, rc_actions)
                actions = rc_actions
            if self.offpolicy_weight:
                self._store_behaviour_weights(
                    step, action_log_probs, actions, groups_all, eps)
        elif self.offpolicy_weight and self._behav_w is not None:
            # eps decayed to 0: the data are on-policy, so weights must be 1.
            # Never reuse a weight left over from an earlier step.
            self._behav_w[step] = 1.0
            self.buffer.behav_w = self._behav_w

        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == "MultiDiscrete":
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[
                    actions[:, :, i]
                ]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == "Discrete":
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            agent_id,
            node_obs,
            adj,
            agent_id,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # if centralized critic, then shared_obs is concatenation of obs from all agents
        if self.use_centralized_V:
            # TODO stack agent_id as well for agent specific information
            # (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            # (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
            share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
            share_agent_id = np.expand_dims(share_agent_id, 1).repeat(
                self.num_agents, axis=1
            )
        else:
            share_obs = obs
            share_agent_id = agent_id

        self.buffer.insert(
            share_obs,
            obs,
            node_obs,
            adj,
            agent_id,
            share_agent_id,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.node_obs[-1]),
            np.concatenate(self.buffer.adj[-1]),
            np.concatenate(self.buffer.share_agent_id[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    @torch.no_grad()
    def eval(self, total_num_steps: int):
        eval_episode_rewards = []
        eval_obs, eval_agent_id, eval_node_obs, eval_adj = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_node_obs),
                np.concatenate(eval_adj),
                np.concatenate(eval_agent_id),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(
                np.split(_t2n(eval_action), self.n_eval_rollout_threads)
            )
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads)
            )

            if self.eval_envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(
                        self.eval_envs.action_space[0].high[i] + 1
                    )[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate(
                            (eval_actions_env, eval_uc_actions_env), axis=2
                        )
            elif self.eval_envs.action_space[0].__class__.__name__ == "Discrete":
                eval_actions_env = np.squeeze(
                    np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2
                )
            else:
                raise NotImplementedError

            # Obser reward and next obs
            (
                eval_obs,
                eval_agent_id,
                eval_node_obs,
                eval_adj,
                eval_rewards,
                eval_dones,
                eval_infos,
            ) = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), dtype=np.float32
            )

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(
            np.array(eval_episode_rewards), axis=0
        )
        eval_average_episode_rewards = np.mean(
            eval_env_infos["eval_average_episode_rewards"]
        )
        print(
            "eval average episode rewards of agent: "
            + str(eval_average_episode_rewards)
        )
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self, get_metrics: bool = False):
        """
        Visualize the env.
        get_metrics: bool (default=False)
            if True, just return the metrics of the env and don't render.
        """
        envs = self.envs

        all_frames = []
        rewards_arr, success_rates_arr, num_collisions_arr, frac_episode_arr = (
            [],
            [],
            [],
            [],
        )

        for episode in range(self.all_args.render_episodes):
            obs, agent_id, node_obs, adj = envs.reset()
            if not get_metrics:
                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                else:
                    envs.render("human")

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
            )

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(node_obs),
                    np.concatenate(adj),
                    np.concatenate(agent_id),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(
                    np.split(_t2n(rnn_states), self.n_rollout_threads)
                )

                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[
                            actions[:, :, i]
                        ]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate(
                                (actions_env, uc_actions_env), axis=2
                            )
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, agent_id, node_obs, adj, rewards, dones, infos = envs.step(
                    actions_env
                )
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones(
                    (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
                )
                masks[dones == True] = np.zeros(
                    ((dones == True).sum(), 1), dtype=np.float32
                )

                if not get_metrics:
                    if self.all_args.save_gifs:
                        image = envs.render("rgb_array")[0][0]
                        all_frames.append(image)
                        calc_end = time.time()
                        elapsed = calc_end - calc_start
                        if elapsed < self.all_args.ifi:
                            time.sleep(self.all_args.ifi - elapsed)
                    else:
                        envs.render("human")

            env_infos = self.process_infos(infos)
            # print('_'*50)
            num_collisions = self.get_collisions(env_infos)
            frac, success = self.get_fraction_episodes(env_infos)
            rewards_arr.append(np.mean(np.sum(np.array(episode_rewards), axis=0)))
            frac_episode_arr.append(np.mean(frac))
            success_rates_arr.append(success)
            num_collisions_arr.append(num_collisions)
            # print(np.mean(frac), success)
            # print("Average episode rewards is: " +
            # str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        print(rewards_arr)
        print(frac_episode_arr)
        print(success_rates_arr)
        print(num_collisions_arr)

        if not get_metrics:
            if self.all_args.save_gifs:
                imageio.mimsave(
                    str(self.gif_dir) + "/render.gif",
                    all_frames,
                    duration=self.all_args.ifi,
                )
