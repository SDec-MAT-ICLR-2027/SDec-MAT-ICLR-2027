"""Runner for the LBF B vs C test.

Same data path as aerial_runner / smac_runner. Logging is LBF-specific:
normalized return, foods collected by required level, split rate, failed-load
rate, spawned-food histogram, episode length.
"""

import json
import os
import time
from functools import reduce

import numpy as np
import torch
import wandb

from mat.envs.lbf.lbf_mat_wrapper import DIAG_KEYS
from mat.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class LBFRunner(Runner):
    def __init__(self, config):
        super(LBFRunner, self).__init__(config)
        self._finished_returns = []
        self._finished_infos = []
        self._running_returns = np.zeros(self.n_rollout_threads, dtype=np.float32)
        self._running_lengths = np.zeros(self.n_rollout_threads, dtype=np.int64)
        self.log_dir = getattr(self, "log_dir", None)
        self._metrics_path = (
            os.path.join(self.log_dir, "metrics.jsonl") if self.log_dir else None
        )
        self._t_start = time.time()
        self.use_commitment = bool(getattr(self.policy, "use_commitment", False))
        if self.use_commitment:
            self._null_id = int(self.policy.n_commit_targets)
            self._z_prev = np.full(
                (self.n_rollout_threads, self.num_agents, 1),
                self._null_id,
                dtype=np.float32,
            )
            self._z_sample = None
            self._z_hit = 0
            self._z_app = 0
            self._z_null = 0
            self._z_switch = 0
            self._z_n = 0
            self._z_agree = 0
            self._z_agree_n = 0
            self._z_hit_first = 0
            self._z_app_first = 0
            self._z_hit_rest = 0
            self._z_app_rest = 0
            self._commit_horizon = int(getattr(self.all_args, "commit_horizon", 1))
            if self._commit_horizon < 1:
                raise ValueError(
                    "commit_horizon must be >= 1, got %r" % self._commit_horizon
                )
        self._print_architecture()

    def _print_architecture(self):
        a = self.all_args
        n_params = sum(p.numel() for p in self.policy.transformer.parameters())
        comm = a.algorithm_name in ("mat_comm", "mat_comm_gru")
        if a.dec_actor:
            actor = "MAPPO / mat_dec: shared MLP, independent action sampling"
        elif comm:
            actor = "mat_comm: group-masked AR decoder (joint actions inside components)"
        else:
            actor = "MAT: autoregressive decoder over all agents"
        print("\n" + "=" * 72)
        print(
            f" {a.algorithm_name}  |  {a.experiment_name}  |  seed {a.seed}  |  {self.device}"
        )
        print("=" * 72)
        print(
            f"   env        LBF {a.grid_size}x{a.grid_size}  {a.n_agents}p  {a.n_foods}f  "
            f"sight {a.sight}  food_sight {a.food_sight}  "
            f"agent_sight {a.agent_sight}  horizon {a.max_episode_steps}"
        )
        roster = a.player_levels if a.player_levels else (
            f"{a.min_player_level}..{a.max_player_level}"
        )
        food_lv = a.food_levels if a.food_levels else (
            f"{a.min_food_level}..{a.max_food_level}"
        )
        print(
            f"              player_lv {roster}  "
            f"food_lv {food_lv}  "
            f"force_coop {a.force_coop}  penalty {a.penalty}  "
            f"exact_match {a.exact_match}"
        )
        print(f"   actor      {actor}")
        print(
            f"   shapes     obs {self.policy.obs_dim}  share_obs {self.policy.share_obs_dim}  "
            f"actions {self.policy.act_dim}"
        )
        print(
            f"   model      n_block {a.n_block}, n_embd {a.n_embd}, n_head {a.n_head}, "
            f"{n_params:,} params"
        )
        print(
            f"   rollout    {self.n_rollout_threads} threads x {self.episode_length} steps, "
            f"ppo_epoch {a.ppo_epoch}, lr {a.lr}, entropy {a.entropy_coef}"
        )
        if int(getattr(a, "group_obs", 0)) == 1:
            print(
                "   group obs  SECONDARY merge + vis bit  R=%s  max_group=%s  "
                "share_obs raw gym concat"
                % (getattr(a, "comm_radius", None), getattr(a, "max_group_size", None))
            )
        elif comm:
            print("   group obs  OFF (primary CT-SDE: local obs, masked AR)")
        if comm:
            print(
                "   comm mask  pairwise N×N Chebyshev 1-hop (no closure), "
                "kind=%s radius=%s"
                % (getattr(a, "comm_mask_kind", None), getattr(a, "comm_radius", None))
            )
            crit = getattr(a, "encoder_critic_attn", None)
            if crit == "soft_group_bias":
                if int(a.commit_critic_pair) == 1:
                    print(
                        "   critic     soft pair MLP [G_ij, same_target] "
                        "(replaces group_bias)"
                    )
                else:
                    print("   critic     soft group_bias (2nd encoder pass, shared weights)")
            else:
                print("   critic     group-local (masked encoder)")
                if crit not in ("hard", None):
                    raise ValueError("unexpected encoder_critic_attn=%r" % (crit,))
            if int(getattr(a, "commitment", 0)) == 1:
                print(
                    "   commit     discrete z over %d foods + NULL; z then a; "
                    "no switch penalty; H_a=%s H_z=%s action_blind=%s "
                    "horizon=%s critic_z=%s critic_pair=%s"
                    % (int(getattr(a, "n_commit_targets", 0)),
                       a.entropy_coef, a.commit_entropy_coef,
                       int(getattr(a, "commit_action_blind", 0)),
                       int(getattr(a, "commit_horizon", 1)),
                       int(getattr(a, "commit_critic_z", 0)),
                       int(getattr(a, "commit_critic_pair", 0)))
                )
        print(f"   logs       {self.log_dir}")
        print("=" * 72 + "\n")

    @torch.no_grad()
    def compute(self):
        if not self.policy.critic_needs_commits:
            return super().compute()
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.share_obs[-1]),
            np.concatenate(self.buffer.obs[-1]),
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
            np.concatenate(self.buffer.available_actions[-1]),
            commits=np.concatenate(self.buffer.commitments[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def run(self):
        self.warmup()
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = (
                    self.collect(step)
                )
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(
                    actions
                )
                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )
                self.insert(data)

            self.compute()
            train_infos = self.train()
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save(episode)

            if episode % self.log_interval == 0:
                train_infos["dead_ratio"] = 1 - self.buffer.active_masks.sum() / reduce(
                    lambda x, y: x * y, list(self.buffer.active_masks.shape)
                )
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
        if self.use_commitment:
            self._z_prev.fill(self._null_id)

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        kwargs = {}
        if self.use_commitment:
            kwargs["z_prev"] = self._z_prev
            kwargs["z_hold"] = (self._running_lengths % self._commit_horizon) != 0
        out = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            np.concatenate(self.buffer.available_actions[step]),
            **kwargs,
        )
        if self.use_commitment:
            value, action, action_log_prob, rnn_state, rnn_state_critic, commits = out
        else:
            value, action, action_log_prob, rnn_state, rnn_state_critic = out
            commits = None
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_state_critic), self.n_rollout_threads)
        )
        if self.use_commitment:
            self._z_sample = np.array(np.split(_t2n(commits), self.n_rollout_threads))
            self._note_z(
                self.buffer.share_obs[step], actions, self._z_sample, self._z_prev
            )
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        dones_env = np.all(dones, axis=1)
        self._track_episodes(rewards, dones_env, infos)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                *self.buffer.rnn_states_critic.shape[3:],
            ),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )
        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )
        bad_masks = np.array(
            [
                [
                    [0.0] if info[agent_id]["bad_transition"] else [1.0]
                    for agent_id in range(self.num_agents)
                ]
                for info in infos
            ]
        )
        if not self.use_centralized_V:
            share_obs = obs
        insert_kw = {}
        if self.use_commitment:
            insert_kw["commitments"] = self._z_sample
            insert_kw["commitment_prev"] = self._z_prev
        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            bad_masks,
            active_masks,
            available_actions,
            **insert_kw,
        )
        if self.use_commitment:
            self._z_prev = self._z_sample.copy()
            self._z_prev[dones_env] = self._null_id

    def _track_episodes(self, rewards, dones_env, infos):
        self._running_returns += rewards[:, 0, 0]
        self._running_lengths += 1
        for i in np.nonzero(dones_env)[0]:
            self._finished_returns.append(float(self._running_returns[i]))
            self._finished_infos.append(
                {k: float(infos[i][0].get(k, 0.0)) for k in DIAG_KEYS}
            )
            self._running_returns[i] = 0.0
            self._running_lengths[i] = 0

    def _drain_episode_stats(self):
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
            stats[key + "_mean"] = float(
                np.mean([info[key] for info in self._finished_infos])
            )
        stats["solve_rate"] = float(
            np.mean([info["solved"] >= 1.0 for info in self._finished_infos])
        )
        stats.update(self._comm_stats())
        stats.update(self._z_stats())
        self._finished_returns = []
        self._finished_infos = []
        return stats

    def _note_z(self, share_obs, actions, commits, z_prev):
        from mat.algorithms.utils.mat_comm_commit import (
            action_reduces_dist,
            parse_commitment_pack,
            z_agreement_step,
            z_meaning_step,
        )

        st = torch.as_tensor(share_obs)
        n_tgt = int(self.policy.n_commit_targets)
        ax, tx, alive, _ = parse_commitment_pack(
            st, self.num_agents, n_tgt, "pairwise"
        )
        ax = ax.detach().cpu().numpy()
        tx = tx.detach().cpu().numpy()
        alive = alive.detach().cpu().numpy()
        g = int(self.all_args.grid_size)
        for b in range(share_obs.shape[0]):
            hit, n_app, n_null, n = z_meaning_step(
                actions[b, :, 0],
                commits[b, :, 0],
                ax[b],
                tx[b],
                alive[b],
                g,
                "lbf",
            )
            self._z_hit += hit
            self._z_app += n_app
            self._z_null += n_null
            self._z_n += n
            self._z_switch += int(np.sum(commits[b, :, 0] != z_prev[b, :, 0]))
            self._z_agree += z_agreement_step(commits[b, :, 0], n_tgt)
            self._z_agree_n += 1
            for i in range(self.num_agents):
                z = int(commits[b, i, 0])
                if z >= n_tgt or float(alive[b, z]) <= 0.5:
                    continue
                axy = (float(ax[b, i, 0]) * g, float(ax[b, i, 1]) * g)
                txy = (float(tx[b, z, 0]) * g, float(tx[b, z, 1]) * g)
                ok = action_reduces_dist(int(actions[b, i, 0]), axy, txy, "lbf")
                if i == 0:
                    self._z_app_first += 1
                    self._z_hit_first += int(ok)
                else:
                    self._z_app_rest += 1
                    self._z_hit_rest += int(ok)

    def _z_stats(self):
        if not self.use_commitment or self._z_n <= 0:
            return {}
        out = {
            "z_approach_frac": (
                float(self._z_hit) / float(self._z_app)
                if self._z_app > 0
                else float("nan")
            ),
            "z_approach_first": (
                float(self._z_hit_first) / float(self._z_app_first)
                if self._z_app_first > 0
                else float("nan")
            ),
            "z_approach_rest": (
                float(self._z_hit_rest) / float(self._z_app_rest)
                if self._z_app_rest > 0
                else float("nan")
            ),
            "z_switch_rate": float(self._z_switch) / float(self._z_n),
            "z_null_frac": float(self._z_null) / float(self._z_n),
            "z_agree_frac": (
                float(self._z_agree) / float(self._z_agree_n)
                if self._z_agree_n > 0
                else float("nan")
            ),
        }
        self._z_hit = 0
        self._z_app = 0
        self._z_null = 0
        self._z_switch = 0
        self._z_n = 0
        self._z_agree = 0
        self._z_agree_n = 0
        self._z_hit_first = 0
        self._z_app_first = 0
        self._z_hit_rest = 0
        self._z_app_rest = 0
        return out

    def _comm_stats(self):
        if getattr(self.all_args, "algorithm_name", "") not in (
            "mat_comm",
            "mat_comm_gru",
        ):
            return {}
        if getattr(self.all_args, "comm_mask_kind", None) != "pairwise":
            return {}
        n = self.num_agents
        n2 = n * n
        if self.buffer.share_obs.shape[-1] < n2:
            return {}
        tail = self.buffer.share_obs[:-1, :, 0, -n2:]
        adj = tail.reshape(-1, n, n) > 0.5
        deg = adj.sum(axis=-1)
        neigh_max = deg.max(axis=-1)
        tril = np.tril(np.ones((n, n), dtype=bool))
        ar_len = (adj & tril).sum(axis=-1).max(axis=-1)
        from mat.algorithms.utils.mat_comm_groups import one_hop_shift_leak_np

        leak = one_hop_shift_leak_np(adj)
        return {
            "comm_mean_degree": float(deg.mean()),
            "comm_isolated_frac": float((deg <= 1).mean()),
            "comm_fully_connected_frac": float(adj.all(axis=(-2, -1)).mean()),
            "comm_neigh_max_mean": float(neigh_max.mean()),
            "comm_neigh_full_frac": float((neigh_max >= n).mean()),
            "comm_ar_chain_max_mean": float(ar_len.mean()),
            "comm_shift_leak_frac": float(leak.mean()),
        }

    def _report(self, episode, episodes, total_num_steps, fps, info):
        g = lambda k, d=float("nan"): info.get(k, d)
        mins = (time.time() - self._t_start) / 60.0
        pct = 100.0 * total_num_steps / max(1, int(self.num_env_steps))
        print("-" * 72)
        print(
            f" {self.algorithm_name} | {self.experiment_name} | update {episode}/{episodes} "
            f"| {total_num_steps}/{int(self.num_env_steps)} steps ({pct:.0f}%) "
            f"| {fps} FPS | {mins:.0f}m"
        )
        if "return_mean" in info:
            print(
                f"   return     mean {g('return_mean'):+8.3f}   std {g('return_std'):7.3f}   "
                f"max {g('return_max'):+8.3f}   collect {g('collect_return_mean'):+.3f}   "
                f"solve {g('solve_rate'):.3f}   "
                f"ep_len {g('ep_length_mean'):.1f}   n={int(g('episodes_completed', 0))}"
            )
            print(
                f"   collected  lvl1 {g('collected_lvl1_mean'):.2f}   "
                f"lvl2 {g('collected_lvl2_mean'):.2f}   "
                f"lvl3 {g('collected_lvl3_mean'):.2f}   "
                f"lvl4 {g('collected_lvl4_mean'):.2f}   "
                f"lvl3+ {g('collected_lvl3plus_mean'):.2f}"
            )
            print(
                f"   spawned    lvl1 {g('spawned_lvl1_mean'):.2f}   "
                f"lvl2 {g('spawned_lvl2_mean'):.2f}   "
                f"lvl3 {g('spawned_lvl3_mean'):.2f}   "
                f"lvl4 {g('spawned_lvl4_mean'):.2f}   "
                f"lvl3+ {g('spawned_lvl3plus_mean'):.2f}   "
                f"coop_foods {g('n_coop_foods_mean'):.2f}"
            )
            print(
                f"   coord      split_rate {g('split_rate_mean'):.3f}   "
                f"parallel {g('parallel_work_rate_mean'):.3f}   "
                f"par_collect {g('parallel_collect_mean'):.3f}   "
                f"t_first_adj {g('t_first_adj_mean'):.1f}   "
                f"failed_load {g('failed_load_rate_mean'):.3f}   "
                f"overshoot {g('load_overshoot_rate_mean'):.3f}   "
                f"undershoot {g('load_undershoot_rate_mean'):.3f}   "
                f"n_load {g('n_load_actions_mean'):.2f}"
            )
            print(
                f"   coal       H {g('coalition_entropy_mean'):.3f}   "
                f"types {g('n_coalition_types_mean'):.2f}   "
                f"n_col {g('n_collected_mean'):.2f}   "
                f"c31 {g('frac_c31_mean'):.3f}   "
                f"c112 {g('frac_c112_mean'):.3f}   "
                f"c22 {g('frac_c22_mean'):.3f}"
            )
            print(
                f"   alloc      waste_adj {g('wasted_adjacency_mean'):.2f}   "
                f"eff {g('alloc_efficiency_mean'):.3f}"
            )
            if "group_size_mean_mean" in info or "group_frac_1_mean" in info:
                print(
                    f"   group      size {g('group_size_mean_mean'):.2f}   "
                    f"f1 {g('group_frac_1_mean'):.3f}   "
                    f"f2 {g('group_frac_2_mean'):.3f}   "
                    f"f3 {g('group_frac_3_mean'):.3f}   "
                    f"f4 {g('group_frac_4_mean'):.3f}   "
                    f"foods_pre {g('foods_vis_pre_mean'):.2f}   "
                    f"foods_post {g('foods_vis_post_mean'):.2f}"
                )
            if "comm_mean_degree" in info:
                print(
                    f"   comm       deg {g('comm_mean_degree'):.2f}   "
                    f"isolated {g('comm_isolated_frac'):.3f}   "
                    f"full {g('comm_fully_connected_frac'):.3f}   "
                    f"neigh_max {g('comm_neigh_max_mean'):.2f}   "
                    f"neigh_full {g('comm_neigh_full_frac'):.3f}   "
                    f"ar_chain {g('comm_ar_chain_max_mean'):.2f}   "
                    f"shift_leak {g('comm_shift_leak_frac'):.3f}"
                )
            xf = getattr(self.policy, "transformer", None)
            if xf is not None and hasattr(xf, "pop_decode_times"):
                times = xf.pop_decode_times()
                info.update(times)
                print(
                    f"   decode     rollout {1000.0 * times['dec_rollout_s']:.2f} ms/call   "
                    f"train {1000.0 * times['dec_train_s']:.2f} ms/call"
                )
            if "group_memory_reset_rate_mean" in info:
                print(
                    f"   memory     reset_rate {g('group_memory_reset_rate_mean'):.3f}"
                )
            if "z_approach_frac" in info:
                print(
                    f"   commit     approach {g('z_approach_frac'):.3f}   "
                    f"first {g('z_approach_first'):.3f}   "
                    f"rest {g('z_approach_rest'):.3f}   "
                    f"switch {g('z_switch_rate'):.3f}   "
                    f"null {g('z_null_frac'):.3f}   "
                    f"agree {g('z_agree_frac'):.3f}"
                )
            if "food_untouched_mean" in info or "n_foods_left_mean" in info:
                print(
                    f"   foods      left {g('n_foods_left_mean'):.2f}   "
                    f"untouched {g('food_untouched_mean'):.2f}   "
                    f"solo {g('food_solo_mean'):.2f}   "
                    f"async {g('food_async_mean'):.2f}   "
                    f"paired_fail {g('food_paired_fail_mean'):.2f}   "
                    f"cluster {g('food_cluster_mean'):.2f}"
                )
        else:
            print("   (no episode finished since the last log point)")
        print(
            f"   optim      value_loss {g('value_loss'):.4f}   "
            f"policy_loss {g('policy_loss'):+.4f}   entropy {g('dist_entropy'):.4f}"
        )
        print("-" * 72)
        if self._metrics_path is not None:
            row = {
                "episode": int(episode),
                "total_num_steps": int(total_num_steps),
                "fps": int(fps),
                "minutes": round(mins, 2),
            }
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

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        eval_episode_infos = []
        one_episode_rewards = [[] for _ in range(self.n_eval_rollout_threads)]
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        eval_z_prev = None
        eval_ep_len = np.zeros(self.n_eval_rollout_threads, dtype=np.int64)
        eval_agree = 0
        eval_agree_n = 0
        if self.use_commitment:
            eval_z_prev = np.full(
                (self.n_eval_rollout_threads, self.num_agents, 1),
                self._null_id,
                dtype=np.float32,
            )
        while True:
            self.trainer.prep_rollout()
            act_kw = {}
            if self.use_commitment:
                act_kw["z_prev"] = eval_z_prev
                act_kw["z_hold"] = (eval_ep_len % self._commit_horizon) != 0
            act_out = self.trainer.policy.act(
                np.concatenate(eval_share_obs),
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                np.concatenate(eval_available_actions),
                deterministic=True,
                **act_kw,
            )
            if self.use_commitment:
                eval_actions, eval_rnn_states, eval_z = act_out
            else:
                eval_actions, eval_rnn_states = act_out
                eval_z = None
            eval_actions = np.array(
                np.split(_t2n(eval_actions), self.n_eval_rollout_threads)
            )
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads)
            )
            if self.use_commitment:
                from mat.algorithms.utils.mat_comm_commit import z_agreement_step
                eval_z_now = np.array(
                    np.split(_t2n(eval_z), self.n_eval_rollout_threads)
                )
                n_tgt = int(self.policy.n_commit_targets)
                for ei in range(self.n_eval_rollout_threads):
                    eval_agree += z_agreement_step(eval_z_now[ei, :, 0], n_tgt)
                    eval_agree_n += 1
                eval_z_prev = eval_z_now
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i][0][0])
            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = np.zeros(
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.all_args.n_eval_rollout_threads, self.num_agents, 1),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )
            if self.use_commitment:
                eval_z_prev[eval_dones_env == True] = self._null_id
            eval_ep_len += 1
            eval_ep_len[eval_dones_env] = 0
            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards[eval_i]))
                    one_episode_rewards[eval_i] = []
                    eval_episode_infos.append(
                        {k: float(eval_infos[eval_i][0].get(k, 0.0)) for k in DIAG_KEYS}
                    )
            if eval_episode >= self.all_args.eval_episodes:
                eval_scalars = {
                    "eval_return_mean": float(np.mean(eval_episode_rewards)),
                    "eval_solve_rate": float(
                        np.mean([i["solved"] >= 1.0 for i in eval_episode_infos])
                    ),
                }
                for key in DIAG_KEYS:
                    eval_scalars["eval_" + key + "_mean"] = float(
                        np.mean([i[key] for i in eval_episode_infos])
                    )
                print("=" * 72)
                print(f" EVAL @ {total_num_steps} steps ({eval_episode} greedy episodes)")
                print(
                    f"   solve {eval_scalars['eval_solve_rate']:.3f}   "
                    f"return {eval_scalars['eval_return_mean']:+.3f}   "
                    f"n_col {eval_scalars['eval_n_collected_mean']:.2f}   "
                    f"split {eval_scalars['eval_split_rate_mean']:.3f}"
                )
                print(
                    f"   foods      left {eval_scalars['eval_n_foods_left_mean']:.2f}   "
                    f"untouched {eval_scalars['eval_food_untouched_mean']:.2f}   "
                    f"solo {eval_scalars['eval_food_solo_mean']:.2f}   "
                    f"async {eval_scalars['eval_food_async_mean']:.2f}   "
                    f"paired_fail {eval_scalars['eval_food_paired_fail_mean']:.2f}   "
                    f"cluster {eval_scalars['eval_food_cluster_mean']:.2f}"
                )
                if eval_agree_n > 0:
                    eval_scalars["eval_z_agree_frac"] = float(eval_agree) / float(
                        eval_agree_n
                    )
                    print(
                        f"   commit     agree {eval_scalars['eval_z_agree_frac']:.3f}"
                    )
                print("=" * 72)
                for k, v in eval_scalars.items():
                    if self.use_wandb:
                        wandb.log({k: v}, step=total_num_steps)
                    else:
                        self.writter.add_scalars(k, {k: v}, total_num_steps)
                break
