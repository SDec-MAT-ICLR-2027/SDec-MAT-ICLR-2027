"""Runner for the Team Partition benchmark.

Structurally this is `charging_runner.py` -- warmup / collect / insert / compute
/ train are unchanged, because the data path is identical. Only the logging
differs, because the benchmark's failure modes are different.

Every episode ends in exactly one of three ways, and the three rates sum to 1:

    success_rate       the first round anyone GOes resolves the attempt, and
                       every zone's count equals its quota  (info exact_match)
    wrong_split_rate   an attempt resolved, some count wrong: -wrong_penalty
    miss_rate          no attempt resolved: the gate deadline or the commit
                       window expired, or the horizon ran out  (no_attempt)

`miss_rate` near 1 is the navigate-and-wait floor; `gate_missed` says how much
of it is the quorum never assembling in time (navigation) rather than the
assembled team declining (counting).

What separates navigation from counting, in order of reading:

    regrouped_ever / gate_open_ever   did the team ever reach the decision
    blocked_commits                   GOes absorbed before the gate opened
    zones_matched                     partial credit on the split (0..n_zones)
    n_hist_classes                    distinct observation histories -- the
                                      symmetry break a memory/comm policy uses
    analytic_ceiling / coordinated_cap  the env's own per-episode reference
                                      lines: memoryless symmetric vs joint

`comm` stats read the per-agent comm bits out of the share_obs tail -- the same
groups `mat_comm` builds (one clique of assembled agents plus singletons).

Diagnostics are not a hand-kept list: every scalar the env puts in `info` is
averaged, so a key the env gains later shows up without a change here.
"""

import json
import os
import time
from functools import reduce

import numpy as np
import torch
import wandb

from mat.runner.shared.base_runner import Runner

# Wrapper bookkeeping, not env diagnostics.
_NON_DIAG_KEYS = ("bad_transition", "won")


def _t2n(x):
    return x.detach().cpu().numpy()


def _diag(info):
    """The env's scalar diagnostics out of one agent's info dict."""
    return {k: float(v) for k, v in info.items() if k not in _NON_DIAG_KEYS}


def _mean_over(infos):
    """Per-key mean over episode infos, over the keys each episode carries."""
    keys = sorted({k for i in infos for k in i})
    return {k: float(np.mean([i[k] for i in infos if k in i])) for k in keys}


def _outcome_rates(infos):
    """success / wrong split / miss fractions over a list of episode infos."""
    success = np.array([i.get("exact_match", 0.0) >= 1.0 for i in infos])
    miss = np.array([i.get("no_attempt", 0.0) >= 1.0 for i in infos])
    return {
        "success_rate": float(success.mean()),
        "wrong_split_rate": float((~success & ~miss).mean()),
        "miss_rate": float((~success & miss).mean()),
    }


class TeamPartitionRunner(Runner):
    """Training / evaluation loop for Team Partition."""

    def __init__(self, config):
        super(TeamPartitionRunner, self).__init__(config)

        # Episodes that finished since the last log point.
        self._finished_returns = []
        self._finished_infos = []
        # Undiscounted return currently accumulating in each rollout thread.
        self._running_returns = np.zeros(self.n_rollout_threads, dtype=np.float32)

        # Append-only metrics log, readable while training runs (summary.json
        # is only written at the very end). base_runner defines log_dir only on
        # the non-wandb path.
        self.log_dir = getattr(self, "log_dir", None)
        self._metrics_path = os.path.join(self.log_dir, "metrics.jsonl") \
            if self.log_dir else None
        self._t_start = time.time()
        # Greedy (argmax) vs sampled eval; see --eval_deterministic in config.py.
        self._eval_deterministic = bool(
            getattr(self.all_args, "eval_deterministic", True))
        self._print_architecture()

    def _print_architecture(self):
        """One-time banner: what is actually being trained."""
        a = self.all_args
        kw = getattr(a, "env_kwargs", {})
        g = lambda k: kw.get(k, "?")
        n_params = sum(p.numel() for p in self.policy.transformer.parameters())
        comm = a.algorithm_name in ("mat_comm", "mat_comm_gru")
        recurrent = getattr(self.policy, "is_recurrent", False)
        print("\n" + "=" * 78)
        print(f" {a.algorithm_name}  |  {a.experiment_name}  |  seed {a.seed}  |  {self.device}")
        print("=" * 78)
        print(f"   env        {self.num_agents} agents ({g('n_spare')} spare), "
              f"{g('n_zones')} zones, grid {g('grid_w')}x{g('grid_h')}, "
              f"horizon {getattr(a, 'env_horizon', '?')}, comm_radius {g('comm_radius')}")
        print(f"              waves: near {g('near_wave_size')} @ {g('spawn_distance')}, "
              f"gap {g('wave_gap')}   gate: commit_gate {g('commit_gate')}, "
              f"window {g('commit_window')}, deadline {g('gate_deadline')}")
        print(f"              reward: all {g('all_reward')}, zone {g('zone_reward')}, "
              f"wrong -{g('wrong_penalty')}, miss -{g('miss_penalty')}, "
              f"dodge {g('dodge_penalty')}, step {g('step_cost')}..{g('step_cost_far')}, "
              f"shaping {g('shaping_weight')}")
        print("              defaults from mac/presets.py (paper configuration)")
        print(f"   shapes     obs {self.policy.obs_dim}"
              f"{' (incl. ' + str(self.num_agents) + ' one-hot id)' if a.obs_agent_id else ''}  "
              f"share_obs {self.policy.share_obs_dim} (state + {self.num_agents} comm bits)  "
              f"actions {self.policy.act_dim}")
        print(f"   model      n_block {a.n_block}, n_embd {a.n_embd}, n_head {a.n_head}, "
              f"{n_params:,} params, raw-obs LayerNorm "
              f"{'OFF' if a.no_raw_obs_layernorm else 'on'}")
        if comm:
            kind = getattr(a, "comm_mask_kind", "bits")
            print(f"   comm mask  ON  - {kind} tail, encoder block-diagonal per component, "
                  f"one decode chain per component")
            soft = a.algorithm_name == "mat_comm" and a.encoder_critic_attn == "soft_group_bias"
            print(f"   critic     {'global (soft group bias, second unmasked pass)' if soft else 'group-local (masked encoder)'}")
        elif getattr(a, "dec_actor", False):
            print("   actor      DECENTRALISED - one shared MLP over each agent's own obs, "
                  "no comm, no decode chain")
            print("   critic     obs-only encoder over all agents (state discarded by the "
                  "37-zeros hack)")
        else:
            print("   comm mask  OFF - dense attention over all agents")
            print("   critic     obs-only (state discarded by the 37-zeros hack)")
        if recurrent:
            print(f"   memory     per-agent GRU, hidden {a.hidden_size}, "
                  f"encoder input [obs ; h] = {self.policy.obs_dim + a.hidden_size}")
            print(f"              BPTT over {a.data_chunk_length}-step chunks "
                  f"({self.episode_length // a.data_chunk_length} chunks/thread)")
        else:
            print("   memory     none (feed-forward, one decision per timestep)")
        print(f"   rollout    {self.n_rollout_threads} threads x {self.episode_length} steps, "
              f"ppo_epoch {a.ppo_epoch}, lr {a.lr}, clip {a.clip_param}, "
              f"entropy {a.entropy_coef}")
        if a.use_eval:
            print(f"   eval       every {a.eval_interval} updates x {a.eval_episodes} episodes, "
                  f"{'greedy (argmax)' if self._eval_deterministic else 'sampled'}")
        print(f"   logs       {self.log_dir}")
        print("=" * 78 + "\n")

    def _comm_stats(self):
        """How the team was grouped over the last rollout, as mat_comm sees it.

        The comm bits are the final n_agents entries of share_obs (the wrapper
        appends them), so this needs no extra env calls. Under per-agent bits a
        connected agent's component is every connected agent (one clique) and an
        unconnected agent is a singleton.
        """
        if not self.use_centralized_V:
            return {}
        bits = self.buffer.share_obs[:-1, :, 0, -self.num_agents:] > 0.5  # [T, threads, N]
        n_connected = bits.sum(axis=-1, keepdims=True)                     # [T, threads, 1]
        sizes = np.where(bits, n_connected, 1)                             # per-agent comp size
        stats = {
            "comm_mean_group_size": float(sizes.mean()),
            "comm_isolated_frac": float((~bits).mean()),
            "comm_fully_connected_frac": float((bits.all(axis=-1)).mean()),
        }
        bias = getattr(getattr(self.policy.transformer, "encoder", None), "group_bias", None)
        if bias is not None:
            stats["critic_group_bias"] = float(bias.detach().cpu().item())
        return stats

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
                train_infos.update(self._comm_stats())

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

        # No agent ever leaves the episode early: every agent emits an
        # observation and an action every round until the episode ends.
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

        The env's `info` values are final only on the terminal step, and the vec
        env auto-resets right after returning it, so this has to run on the step
        that reports done.
        """
        # Common reward: every agent carries the same scalar, so agent 0 is it.
        self._running_returns += rewards[:, 0, 0]

        for i in np.nonzero(dones_env)[0]:
            self._finished_returns.append(float(self._running_returns[i]))
            self._finished_infos.append(_diag(infos[i][0]))
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
        stats.update({k + "_mean": v for k, v in _mean_over(self._finished_infos).items()})
        stats.update(_outcome_rates(self._finished_infos))

        self._finished_returns = []
        self._finished_infos = []
        return stats

    def _report(self, episode, episodes, total_num_steps, fps, info):
        """Human-readable block on stdout + one JSON line appended to disk."""
        g = lambda k, d=float('nan'): info.get(k, d)
        mins = (time.time() - self._t_start) / 60.0
        pct = 100.0 * total_num_steps / max(1, int(self.num_env_steps))
        n_zones = int(getattr(self.all_args, "env_kwargs", {}).get("n_zones", 3))

        print("-" * 78)
        print(f" {self.algorithm_name} | {self.experiment_name} | update {episode}/{episodes} "
              f"| {total_num_steps}/{int(self.num_env_steps)} steps ({pct:.0f}%) "
              f"| {fps} FPS | {mins:.0f}m")
        if "return_mean" in info:
            print(f"   return     mean {g('return_mean'):+8.2f}   std {g('return_std'):7.2f}   "
                  f"max {g('return_max'):+8.2f}   episodes {int(g('episodes_completed', 0))}")
            print(f"   outcome    success {g('success_rate'):.3f}   "
                  f"wrong_split {g('wrong_split_rate'):.3f}   "
                  f"miss {g('miss_rate'):.3f} (gate_missed {g('gate_missed_mean'):.3f})   "
                  f"zones_matched {g('zones_matched_mean'):.2f}/{n_zones}")
            print(f"   regroup    regrouped_ever {g('regrouped_ever_mean'):.3f}   "
                  f"gate_open_ever {g('gate_open_ever_mean'):.3f}   "
                  f"gate_round {g('gate_round_mean'):.2f}   "
                  f"blocked_commits {g('blocked_commits_mean'):.2f}   "
                  f"n_present {g('n_present_mean'):.2f}")
            counts = "/".join(f"{g(f'count_{c}_mean'):.2f}" for c in range(n_zones))
            quotas = "/".join(f"{g(f'quota_{c}_mean'):.2f}" for c in range(n_zones))
            print(f"   split      committed {g('n_committed_mean'):.2f}   counts {counts}   "
                  f"quota {quotas}   abs_error {g('abs_error_mean'):.2f}")
            print(f"   symmetry   n_hist_classes {g('n_hist_classes_mean'):.2f}   "
                  f"arrival_spread {g('arrival_spread_mean'):.2f}   "
                  f"ref: memoryless {g('analytic_ceiling_mean'):.4f}  "
                  f"coordinated {g('coordinated_cap_mean'):.4f}")
        else:
            print("   (no episode finished since the last log point)")
        comm_line = (f"   comm       mean_group_size {g('comm_mean_group_size'):.2f}   "
                     f"isolated_frac {g('comm_isolated_frac'):.2f}   "
                     f"fully_connected_frac {g('comm_fully_connected_frac'):.2f}")
        if "critic_group_bias" in info:
            comm_line += f"   critic_group_bias {g('critic_group_bias'):+.3f}"
        print(comm_line)
        print(f"   optim      value_loss {g('value_loss'):.4f}   "
              f"policy_loss {g('policy_loss'):+.4f}   entropy {g('dist_entropy'):.4f}   "
              f"ratio {g('ratio'):.3f}   grad_norm {g('actor_grad_norm'):.2f}")
        print("-" * 78)

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
                    eval_episode_infos.append(_diag(eval_infos[eval_i][0]))

            if eval_episode >= self.all_args.eval_episodes:
                eval_env_infos = {
                    'eval_return_mean': np.array(eval_episode_rewards),
                }
                self.log_env(eval_env_infos, total_num_steps)

                eval_scalars = {"eval_" + k: v
                                for k, v in _outcome_rates(eval_episode_infos).items()}
                for k, v in _mean_over(eval_episode_infos).items():
                    eval_scalars["eval_" + k + "_mean"] = v
                e = lambda k: eval_scalars.get(k, float('nan'))

                print("=" * 78)
                print(f" EVAL @ {total_num_steps} steps ({eval_episode} "
                      f"{'greedy' if self._eval_deterministic else 'sampled'} episodes)")
                print(f"   success {e('eval_success_rate'):.4f}   "
                      f"wrong_split {e('eval_wrong_split_rate'):.4f}   "
                      f"miss {e('eval_miss_rate'):.4f}   "
                      f"return {float(np.mean(eval_episode_rewards)):+.2f} "
                      f"+/- {float(np.std(eval_episode_rewards)):.2f}")
                print(f"   regrouped_ever {e('eval_regrouped_ever_mean'):.3f}   "
                      f"gate_missed {e('eval_gate_missed_mean'):.3f}   "
                      f"zones_matched {e('eval_zones_matched_mean'):.2f}   "
                      f"committed {e('eval_n_committed_mean'):.2f}")
                print("=" * 78)

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
