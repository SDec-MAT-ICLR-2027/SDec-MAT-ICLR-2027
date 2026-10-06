"""InforMARL's GMPERunner, extended for the MAC envs.

Training (collect -> compute -> GR_MAPPO.train) is InforMARL's. The additions:
  * action masks are fed to the policy and stored in the GraphReplayBuffer
    (both already support them; the MPE runner just never passes them);
  * episode returns / env info stats are logged the way epymarl/runners/parallel_runner.py
    logs them (return_mean, <info>_mean, test_return_mean, test_success_mean, ...),
    with env steps as the x-axis, so its curves line up with the EPyMARL baselines;
  * test runs mirror EPyMARL: every `test_interval` env steps, `test_nepisode`
    episodes, one episode per eval env per round. Actions are sampled from the
    policy by default, as the baselines' soft_policies selector does at test time.
"""

import os
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter

from onpolicy.runner.shared.graph_mpe_runner import GMPERunner, _t2n


def _is_number(v):
    return isinstance(v, (bool, int, float, np.number, np.bool_))


class MACGraphRunner(GMPERunner):
    def __init__(self, config):
        super().__init__(config)
        args = self.all_args
        # Replace InforMARL's own writer with one in results/informarl/tb_logs/<token>.
        if not self.use_wandb:
            self.writter.close()
        self.writter = SummaryWriter(config["tb_dir"])
        self.models_dir = config["models_dir"]
        self.common_reward = config["common_reward"]
        self.n_actions = self.envs.action_space[0].n

        self.test_nepisode = args.test_nepisode
        self.test_interval = args.test_interval
        self.runner_log_interval = args.runner_log_interval
        self.learner_log_interval = args.learner_log_interval
        self.save_model_interval = args.save_model_interval
        self.eval_deterministic = args.eval_deterministic

        self.t_env = 0
        self.train_episodes = 0
        self._train_returns, self._train_stats = [], {}
        self._ep_ret = np.zeros(self.n_rollout_threads, dtype=np.float64)
        self._ep_len = np.zeros(self.n_rollout_threads, dtype=np.int64)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self):
        self.warmup()
        start = time.time()
        iterations = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_test_T = -self.test_interval - 1
        last_runner_log_T = 0
        last_learner_log_T = -self.learner_log_interval - 1
        model_save_T = 0

        for it in range(iterations):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(it, iterations)

            for step in range(self.episode_length):
                (values, actions, action_log_probs, rnn_states,
                 rnn_states_critic, actions_env) = self.collect(step)
                (obs, agent_id, node_obs, adj, rewards, dones, infos,
                 avail) = self.envs.step(actions_env)
                self.t_env += self.n_rollout_threads
                self._track_train_episodes(rewards, dones, infos)
                self.insert((obs, agent_id, node_obs, adj, rewards, dones, infos, avail,
                             values, actions, action_log_probs, rnn_states,
                             rnn_states_critic))

            self.compute()
            train_infos = self.train()

            if self.t_env - last_learner_log_T >= self.learner_log_interval:
                for k, v in train_infos.items():
                    self.writter.add_scalar(k, float(v), self.t_env)
                last_learner_log_T = self.t_env

            if (self.t_env - last_runner_log_T >= self.runner_log_interval
                    and self._train_returns):
                self._log_stats("", self._train_returns, self._train_stats)
                self.writter.add_scalar("episode", self.train_episodes, self.t_env)
                last_runner_log_T = self.t_env

            if self.use_eval and (self.t_env - last_test_T) / self.test_interval >= 1.0:
                last_test_T = self.t_env
                elapsed = time.time() - start
                print(f"t_env: {self.t_env} / {int(self.num_env_steps)}  "
                      f"({elapsed / 60:.1f} min elapsed)")
                self.evaluate()

            if self.save_model_interval > 0 and (
                    self.t_env - model_save_T >= self.save_model_interval or model_save_T == 0):
                model_save_T = self.t_env
                self.save_models(os.path.join(self.models_dir, str(self.t_env)))

        self.save_models(os.path.join(self.models_dir, str(self.t_env)))
        self.writter.flush()

    # ------------------------------------------------------------------
    # rollout plumbing (GMPERunner versions + action masks)
    # ------------------------------------------------------------------
    def _share(self, obs, agent_id, n_threads):
        if self.use_centralized_V:
            share_obs = obs.reshape(n_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            share_agent_id = agent_id.reshape(n_threads, -1)
            share_agent_id = np.expand_dims(share_agent_id, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs, share_agent_id = obs, agent_id
        return share_obs, share_agent_id

    def warmup(self):
        obs, agent_id, node_obs, adj, avail = self.envs.reset()
        share_obs, share_agent_id = self._share(obs, agent_id, self.n_rollout_threads)
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.node_obs[0] = node_obs.copy()
        self.buffer.adj[0] = adj.copy()
        self.buffer.agent_id[0] = agent_id.copy()
        self.buffer.share_agent_id[0] = share_agent_id.copy()
        self.buffer.available_actions[0] = avail.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic = \
            self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.node_obs[step]),
                np.concatenate(self.buffer.adj[step]),
                np.concatenate(self.buffer.agent_id[step]),
                np.concatenate(self.buffer.share_agent_id[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]),
                np.concatenate(self.buffer.available_actions[step]),
            )
        T = self.n_rollout_threads
        values = np.array(np.split(_t2n(value), T))
        actions = np.array(np.split(_t2n(action), T))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), T))
        rnn_states = np.array(np.split(_t2n(rnn_states), T))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), T))
        actions_env = np.squeeze(np.eye(self.n_actions)[actions], 2)
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        (obs, agent_id, node_obs, adj, rewards, dones, infos, avail, values, actions,
         action_log_probs, rnn_states, rnn_states_critic) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs, share_agent_id = self._share(obs, agent_id, self.n_rollout_threads)
        self.buffer.insert(share_obs, obs, node_obs, adj, agent_id, share_agent_id,
                           rnn_states, rnn_states_critic, actions, action_log_probs,
                           values, rewards, masks, available_actions=avail)

    # ------------------------------------------------------------------
    # EPyMARL-style episode statistics
    # ------------------------------------------------------------------
    def _team_reward(self, rewards_e):
        return float(rewards_e[0, 0]) if self.common_reward else float(rewards_e[:, 0].sum())

    @staticmethod
    def _accumulate(stats, info, ep_len):
        for k, v in info.items():
            if _is_number(v):
                stats[k] = stats.get(k, 0.0) + float(v)
        stats["n_episodes"] = stats.get("n_episodes", 0) + 1
        stats["ep_length"] = stats.get("ep_length", 0) + int(ep_len)

    def _track_train_episodes(self, rewards, dones, infos):
        for e in range(self.n_rollout_threads):
            self._ep_ret[e] += self._team_reward(rewards[e])
            self._ep_len[e] += 1
            if np.all(dones[e]):
                self._train_returns.append(self._ep_ret[e])
                self._accumulate(self._train_stats, infos[e][0], self._ep_len[e])
                self.train_episodes += 1
                self._ep_ret[e] = 0.0
                self._ep_len[e] = 0

    def _log_stats(self, prefix, returns, stats):
        w, t = self.writter, self.t_env
        w.add_scalar(prefix + "return_mean", float(np.mean(returns)), t)
        w.add_scalar(prefix + "return_std", float(np.std(returns)), t)
        w.add_scalar(prefix + "return_max", float(np.max(returns)), t)
        n = stats.get("n_episodes", len(returns))
        msg = [f"{prefix}return_mean: {np.mean(returns):.4f}"]
        for k, v in stats.items():
            if k != "n_episodes":
                w.add_scalar(prefix + k + "_mean", v / n, t)
                if k == "success":
                    msg.append(f"{prefix}success_mean: {v / n:.4f}")
        if prefix:
            print("  " + "  ".join(msg) + f"  ({n} episodes)")
        returns.clear()
        stats.clear()

    @torch.no_grad()
    def evaluate(self):
        E = self.n_eval_rollout_threads
        returns, stats = [], {}
        for _ in range(max(1, self.test_nepisode // E)):
            obs, agent_id, node_obs, adj, avail = self.eval_envs.reset()
            rnn_states = np.zeros((E, self.num_agents, self.recurrent_N, self.hidden_size),
                                  dtype=np.float32)
            masks = np.ones((E, self.num_agents, 1), dtype=np.float32)
            ep_ret = np.zeros(E)
            ep_len = np.zeros(E, dtype=np.int64)
            finished = np.zeros(E, dtype=bool)
            while not finished.all():
                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs), np.concatenate(node_obs), np.concatenate(adj),
                    np.concatenate(agent_id), np.concatenate(rnn_states),
                    np.concatenate(masks), np.concatenate(avail),
                    deterministic=self.eval_deterministic,
                )
                actions = np.array(np.split(_t2n(action), E))
                rnn_states = np.array(np.split(_t2n(rnn_states), E))
                actions_env = np.squeeze(np.eye(self.n_actions)[actions], 2)
                obs, agent_id, node_obs, adj, rewards, dones, infos, avail = \
                    self.eval_envs.step(actions_env)
                for e in range(E):
                    if finished[e]:
                        continue
                    ep_ret[e] += self._team_reward(rewards[e])
                    ep_len[e] += 1
                    if np.all(dones[e]):
                        finished[e] = True
                        returns.append(ep_ret[e])
                        self._accumulate(stats, infos[e][0], ep_len[e])
                rnn_states[dones == True] = 0.0
                masks = np.ones((E, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = 0.0
        self._log_stats("test_", returns, stats)

    def save_models(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.trainer.policy.actor.state_dict(), os.path.join(path, "actor.pt"))
        torch.save(self.trainer.policy.critic.state_dict(), os.path.join(path, "critic.pt"))
