"""Runner for the (single-group) Pickup Consensus benchmark.

Structurally this is `aerial_runner.py` / `smac_runner.py` — warmup / collect
/ insert / compute / train are unchanged, because the data path is identical.
The benchmark returns one-shot per-episode diagnostics (`n_pickup`, `K`,
`success`, `overshoot`, `undershoot`, `no_attempt`) rather than SMAC's
cumulative battle counters, so — like the aerial runner — this one
accumulates finished episodes as they complete and averages over the ones
seen since the last log.

The metric that matters here is `success_mean` (== solve_rate): the fraction
of episodes where exactly K agents committed. `freeze` (episodes that ran the
full time_limit with zero commitments — `no_attempt`) is the ALL-WAIT
equilibrium this benchmark is built to expose in baselines that cannot break
symmetry.
"""

import json
import os
import time
from functools import reduce

import numpy as np
import torch
import wandb

from mat.envs.pickup_consensus.pickup_consensus_mat_wrapper import DIAG_KEYS
from mat.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class PickupConsensusRunner(Runner):
    """Training / evaluation loop for Pickup Consensus."""

    def __init__(self, config):
        super(PickupConsensusRunner, self).__init__(config)

        # Episodes that finished since the last log point.
        self._finished_returns = []
        self._finished_infos = []
        # Undiscounted return currently accumulating in each rollout thread.
        self._running_returns = np.zeros(self.n_rollout_threads, dtype=np.float32)

        # Append-only metrics log. tensorboardX only flushes its event file, and
        # summary.json is written at the very end -- a killed run loses it. This
        # is one JSON object per log point, readable while training runs.
        # base_runner only defines log_dir on the non-wandb path.
        self.log_dir = getattr(self, "log_dir", None)
        self._metrics_path = os.path.join(self.log_dir, "metrics.jsonl") \
            if self.log_dir else None
        self._t_start = time.time()
        # Greedy (argmax) vs sampled eval. See --eval_deterministic in config.py:
        # on an identical-observation benchmark this decides whether a
        # decentralised shared-parameter actor can score at all, so it must be
        # held identical across the algorithms being compared.
        self._eval_deterministic = bool(
            getattr(self.all_args, "eval_deterministic", True))
        self._print_architecture()

    def _print_architecture(self):
        """One-time banner: what is actually being trained."""
        a = self.all_args
        n_params = sum(p.numel() for p in self.policy.transformer.parameters())
        comm = a.algorithm_name in ("mat_comm", "mat_comm_gru")
        recurrent = getattr(self.policy, "is_recurrent", False)
        print("\n" + "=" * 72)
        print(f" {a.algorithm_name}  |  {a.experiment_name}  |  seed {a.seed}  |  {self.device}")
        print("=" * 72)
        print(f"   env        {self.num_agents} identical agents, "
              f"K in [{a.k_min},{a.k_max}], n_max {a.n_max}, limit {a.time_limit}")
        print(f"   shapes     obs {self.policy.obs_dim}  share_obs {self.policy.share_obs_dim}  "
              f"actions {self.policy.act_dim}")
        print(f"   model      n_block {a.n_block}, n_embd {a.n_embd}, n_head {a.n_head}, "
              f"{n_params:,} params")
        comm_desc = ("ON  - one sequential decode chain over the whole team "
                     "(no real topology: the env is always fully connected)"
                     if comm else "OFF - dense attention over all agents")
        print(f"   comm mask  {comm_desc}")
        if recurrent:
            print(f"   memory     per-agent GRU, hidden {a.hidden_size}, "
                  f"encoder input [obs ; h] = {self.policy.obs_dim + a.hidden_size}")
            print(f"              BPTT over {a.data_chunk_length}-step chunks "
                  f"({self.episode_length // a.data_chunk_length} chunks/thread)")
        else:
            print("   memory     none (feed-forward, one decision per timestep)")
        print(f"   rollout    {self.n_rollout_threads} threads x {self.episode_length} steps, "
              f"ppo_epoch {a.ppo_epoch}, lr {a.lr}, clip {a.clip_param}")
        if a.use_eval:
            print(f"   eval       every {a.eval_interval} updates x {a.eval_episodes} episodes, "
                  f"{'greedy (argmax)' if self._eval_deterministic else 'sampled'}")
        print(f"   logs       {self.log_dir}")
        print("=" * 72 + "\n")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic

                self.insert(data)

            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save(episode)

            if episode % self.log_interval == 0:
                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(
                    lambda x, y: x * y, list(self.buffer.active_masks.shape))
                train_infos.update(self._drain_episode_stats())

                fps = int(total_num_steps / (time.time() - start))
                self._report(episode, episodes, total_num_steps, fps, train_infos)
                self.log_train(train_infos, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        obs, share_obs, available_actions = self.envs.reset()

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]),
                                              np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        self._track_episodes(rewards, dones_env, infos)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # No agent can die in this benchmark, so every agent is always active.
        # Kept for buffer-shape compatibility with the shared MAT trainer.
        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0]
                               for agent_id in range(self.num_agents)] for info in infos])

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, bad_masks,
                           active_masks, available_actions)

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------

    def _track_episodes(self, rewards, dones_env, infos):
        """Accumulate undiscounted returns, harvest diagnostics on termination.

        Every episode here ends on the step that reports done (either a
        commitment resolves it, or it truncates at time_limit), so the
        terminal step's info already carries the whole episode's outcome.
        """
        # Common reward: every agent carries the same scalar, so agent 0 is it.
        self._running_returns += rewards[:, 0, 0]

        for i in np.nonzero(dones_env)[0]:
            self._finished_returns.append(float(self._running_returns[i]))
            self._finished_infos.append({k: float(infos[i][0].get(k, 0.0)) for k in DIAG_KEYS})
            self._running_returns[i] = 0.0

    def _drain_episode_stats(self):
        """Mean stats over episodes finished since the last log, then clear."""
        if not self._finished_returns:
            return {}

        returns = np.array(self._finished_returns, dtype=np.float32)
        stats = {
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()),
            "return_max": float(returns.max()),
            "episodes_completed": float(len(returns)),
        }
        for key in DIAG_KEYS:
            stats[key + "_mean"] = float(np.mean([info[key] for info in self._finished_infos]))
        stats["solve_rate"] = stats["success_mean"]

        self._finished_returns = []
        self._finished_infos = []
        return stats

    def _report(self, episode, episodes, total_num_steps, fps, info):
        """Human-readable block on stdout + one JSON line appended to disk."""
        g = lambda k, d=float('nan'): info.get(k, d)
        mins = (time.time() - self._t_start) / 60.0
        pct = 100.0 * total_num_steps / max(1, int(self.num_env_steps))

        print("-" * 72)
        print(f" {self.algorithm_name} | {self.experiment_name} | update {episode}/{episodes} "
              f"| {total_num_steps}/{int(self.num_env_steps)} steps ({pct:.0f}%) "
              f"| {fps} FPS | {mins:.0f}m")
        if "return_mean" in info:
            print(f"   return     mean {g('return_mean'):+8.2f}   std {g('return_std'):7.2f}   "
                  f"max {g('return_max'):+8.2f}   episodes {int(g('episodes_completed', 0))}")
            print(f"   task       solve_rate {g('solve_rate'):.3f}   "
                  f"K_mean {g('K_mean'):.2f}   n_pickup_mean {g('n_pickup_mean'):.2f}")
            print(f"   outcomes   overshoot {g('overshoot_mean'):.3f}   "
                  f"undershoot {g('undershoot_mean'):.3f}   "
                  f"no_attempt (freeze) {g('no_attempt_mean'):.3f}")
        else:
            print("   (no episode finished since the last log point)")
        print(f"   optim      value_loss {g('value_loss'):.4f}   "
              f"policy_loss {g('policy_loss'):+.4f}   entropy {g('dist_entropy'):.4f}   "
              f"ratio {g('ratio'):.3f}   grad_norm {g('actor_grad_norm'):.2f}")
        print("-" * 72)

        if self._metrics_path is not None:
            row = {"episode": int(episode), "total_num_steps": int(total_num_steps),
                   "fps": int(fps), "minutes": round(mins, 2)}
            row.update({k: float(v) for k, v in info.items() if np.ndim(v) == 0})
            with open(self._metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        eval_episode_infos = []
        one_episode_rewards = [[] for _ in range(self.n_eval_rollout_threads)]

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_share_obs),
                                        np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(eval_available_actions),
                                        deterministic=self._eval_deterministic)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions \
                = self.eval_envs.step(eval_actions)

            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i][0][0])

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),
                dtype=np.float32)

            eval_masks = np.ones(
                (self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards[eval_i]))
                    one_episode_rewards[eval_i] = []
                    eval_episode_infos.append(
                        {k: float(eval_infos[eval_i][0].get(k, 0.0)) for k in DIAG_KEYS})

            if eval_episode >= self.all_args.eval_episodes:
                eval_env_infos = {
                    'eval_return_mean': np.array(eval_episode_rewards),
                }
                self.log_env(eval_env_infos, total_num_steps)

                eval_solve_rate = float(np.mean(
                    [i["success"] for i in eval_episode_infos]))
                eval_scalars = {"eval_solve_rate": eval_solve_rate}
                for key in DIAG_KEYS:
                    eval_scalars["eval_" + key + "_mean"] = float(
                        np.mean([i[key] for i in eval_episode_infos]))

                print("=" * 72)
                print(f" EVAL @ {total_num_steps} steps ({eval_episode} "
                      f"{'greedy' if self._eval_deterministic else 'sampled'} episodes)")
                print(f"   solve_rate {eval_solve_rate:.4f}   "
                      f"return {float(np.mean(eval_episode_rewards)):+.2f} "
                      f"+/- {float(np.std(eval_episode_rewards)):.2f}")
                print(f"   overshoot {eval_scalars['eval_overshoot_mean']:.3f}   "
                      f"undershoot {eval_scalars['eval_undershoot_mean']:.3f}   "
                      f"no_attempt (freeze) {eval_scalars['eval_no_attempt_mean']:.3f}")
                print("=" * 72)

                if self._metrics_path is not None:
                    row = {"total_num_steps": int(total_num_steps), "phase": "eval",
                           "eval_return_mean": float(np.mean(eval_episode_rewards))}
                    row.update({k: float(v) for k, v in eval_scalars.items()})
                    with open(self._metrics_path, "a") as f:
                        f.write(json.dumps(row) + "\n")

                for k, v in eval_scalars.items():
                    if self.use_wandb:
                        wandb.log({k: v}, step=total_num_steps)
                    else:
                        self.writter.add_scalars(k, {k: v}, total_num_steps)
                break
