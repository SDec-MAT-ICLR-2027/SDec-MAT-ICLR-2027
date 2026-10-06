"""MAT runner for MACO Pursuit. Collect/insert match SMACRunner; logs return and catches."""

import time

import numpy as np
import torch
import wandb

from mat.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class MACOPursuitRunner(Runner):
    def __init__(self, config):
        super(MACOPursuitRunner, self).__init__(config)
        self.use_group_token = bool(getattr(self.policy, "use_group_token", False))
        self._group_z = None
        self._tok_ready2 = 0
        self._tok_go_ready2 = 0
        self._tok_go_members = 0
        self._tok_honor = 0
        self._print_architecture()

    def _print_architecture(self):
        a = self.all_args
        n_params = sum(p.numel() for p in self.policy.transformer.parameters())
        print("\n" + "=" * 72)
        print(
            " %s  |  %s  |  seed %s  |  %s"
            % (a.algorithm_name, a.experiment_name, a.seed, self.device)
        )
        print("=" * 72)
        print(
            "   env        maco_pursuit %dx%d  agents=%d preys=%d "
            "sight=%d horizon=%d catch=%s fail=%s"
            % (
                a.map_size,
                a.map_size,
                a.n_agents,
                a.n_preys,
                a.sight_range,
                a.episode_limit,
                a.catch_reward,
                a.catch_fail_reward,
            )
        )
        print(
            "   catch_min  %s (success if ≥ this many simultaneous catchers)"
            % getattr(a, "catch_min", 2)
        )
        print(
            "   obs_ready  %s (1=clique size in local obs)"
            % getattr(a, "obs_ready_count", 0)
        )
        switch = int(a.fail_penalty_switch_steps)
        if switch >= 0:
            print(
                "   curriculum fail %s -> %s at global step %d"
                % (a.catch_fail_reward, a.catch_fail_reward_after, switch)
            )
        print(
            "   shapes     obs %d  share_obs %d  actions %d"
            % (self.policy.obs_dim, self.policy.share_obs_dim, self.policy.act_dim)
        )
        print(
            "   model      n_block %d, n_embd %d, n_head %d, %s params  "
            "dec_actor=%s share_actor=%s"
            % (
                a.n_block,
                a.n_embd,
                a.n_head,
                f"{n_params:,}",
                a.dec_actor,
                a.share_actor,
            )
        )
        if a.algorithm_name in ("mat_comm",):
            fg = getattr(a, "fixed_groups", None)
            if fg:
                print(
                    "   comm mask  pairwise N×N fixed_groups=%s assign=%s "
                    "(no radius, no prey clique), kind=%s"
                    % (
                        fg,
                        getattr(a, "fixed_groups_assign", "id"),
                        getattr(a, "comm_mask_kind", None),
                    )
                )
            else:
                print(
                    "   comm mask  pairwise N×N same-prey catcher cliques "
                    "(Manhattan≤1, no comm_radius), kind=%s"
                    % (getattr(a, "comm_mask_kind", None),)
                )
            crit = getattr(a, "encoder_critic_attn", None)
            if crit != "soft_group_bias":
                raise ValueError(
                    "maco_pursuit mat_comm runner expects "
                    "encoder_critic_attn=soft_group_bias, got %r" % (crit,)
                )
            print("   critic     soft group_bias (2nd encoder pass, shared weights)")
            if int(getattr(a, "group_token", 0)) == 1:
                print(
                    "   decoder    MLP(obs)+AR(z-as-START, obs_rep)  group_token=GO/NO-GO "
                    "learned (no ready mask) token_H=%s token_credit=clique_catch "
                    "(obs skip ON, loner START=blank)"
                    % (getattr(a, "group_token_entropy_coef", None),)
                )
            else:
                if int(getattr(a, "use_obs_skip", 1)) == 1:
                    print("   decoder    logits = MLP(obs) + AR(token, obs_rep)")
                else:
                    print("   decoder    logits = AR(token, obs_rep)  (obs skip OFF)")
        else:
            print("   comm mask  OFF - independent MAPPO actors")
        print(
            "   ppo        epoch %d, minibatch %d, lr %s, clip %s, "
            "entropy %s"
            % (a.ppo_epoch, a.num_mini_batch, a.lr, a.clip_param, a.entropy_coef)
        )
        print("=" * 72 + "\n", flush=True)

    def _comm_stats(self):
        if getattr(self.all_args, "comm_mask_kind", None) != "pairwise":
            return {}
        if self.all_args.algorithm_name not in ("mat_comm",):
            return {}
        n = self.num_agents
        n2 = n * n
        if self.buffer.share_obs.shape[-1] < n2:
            return {}
        tail = self.buffer.share_obs[:-1, :, 0, -n2:]
        adj = tail.reshape(-1, n, n) > 0.5
        deg = adj.sum(axis=-1)
        return {
            "comm_mean_degree": float(deg.mean()),
            "comm_isolated_frac": float((deg <= 1).mean()),
        }

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        ep_returns, ep_steps, ep_joint, ep_solo, ep_prey, ep_won = [], [], [], [], [], []
        last_fail = float(self.all_args.catch_fail_reward)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                for thread_info in infos:
                    last_fail = float(thread_info[0]["catch_fail_reward"])
                    if thread_info[0].get("episode_done"):
                        ep_returns.append(thread_info[0]["episode_return"])
                        ep_steps.append(thread_info[0]["episode_steps"])
                        ep_joint.append(thread_info[0]["joint_catches"])
                        ep_solo.append(thread_info[0]["solo_fails"])
                        ep_prey.append(thread_info[0]["prey_left"])
                        ep_won.append(float(thread_info[0]["won"]))

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
                end = time.time()
                n_done = len(ep_returns)
                if n_done > 0:
                    mean_ret = float(np.mean(ep_returns))
                    mean_len = float(np.mean(ep_steps))
                    mean_joint = float(np.mean(ep_joint))
                    mean_solo = float(np.mean(ep_solo))
                    mean_prey = float(np.mean(ep_prey))
                    win_rate = float(np.mean(ep_won))
                else:
                    mean_ret = mean_len = mean_joint = mean_solo = mean_prey = win_rate = float("nan")

                print(
                    "\n Env maco_pursuit Algo %s Exp %s updates %d/%d "
                    "episodes, total num timesteps %d/%d, FPS %d.\n"
                    % (
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )
                print(
                    "  episodes_done=%d  return=%.3f  ep_len=%.1f  "
                    "joint_catches=%.2f  solo_fails=%.2f  prey_left=%.2f  "
                    "win_rate=%.3f  fail_r=%.1f"
                    % (
                        n_done,
                        mean_ret,
                        mean_len,
                        mean_joint,
                        mean_solo,
                        mean_prey,
                        win_rate,
                        last_fail,
                    ),
                    flush=True,
                )
                comm_stats = self._comm_stats()
                if comm_stats:
                    print(
                        "  comm_mean_degree=%.3f  comm_isolated_frac=%.3f"
                        % (
                            comm_stats["comm_mean_degree"],
                            comm_stats["comm_isolated_frac"],
                        ),
                        flush=True,
                    )

                train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
                self.log_train(train_infos, total_num_steps)
                env_log = {
                    "return_mean": np.array([mean_ret]),
                    "ep_length_mean": np.array([mean_len]),
                    "joint_catches_mean": np.array([mean_joint]),
                    "solo_fails_mean": np.array([mean_solo]),
                    "prey_left_mean": np.array([mean_prey]),
                    "win_rate": np.array([win_rate]),
                    "episodes_completed": np.array([float(n_done)]),
                    "catch_fail_reward": np.array([last_fail]),
                }
                for k, v in comm_stats.items():
                    env_log[k] = np.array([v])
                tok_stats = self._flush_token_stats()
                if tok_stats:
                    print(
                        "  P(GO|ready>=2)=%.3f  honor|GO=%.3f  n_ready2=%d"
                        % (
                            tok_stats["p_go_given_ready2"],
                            tok_stats["p_honor_given_go"],
                            tok_stats["n_ready2"],
                        ),
                        flush=True,
                    )
                    for k, v in tok_stats.items():
                        env_log[k] = np.array([float(v)])
                self.log_env(env_log, total_num_steps)
                ep_returns, ep_steps, ep_joint, ep_solo, ep_prey, ep_won = [], [], [], [], [], []

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        obs, share_obs, available_actions = self.envs.reset()
        if not self.use_centralized_V:
            share_obs = obs
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    def _z_outcome_from_step(self, share_obs, group_z, infos):
        from mat.algorithms.utils.mat_comm_groups import comm_from_share_obs
        from mat.algorithms.utils.mat_comm_group_token import (
            clique_z_outcome,
            parse_group_token_pack,
        )

        st = torch.as_tensor(np.asarray(share_obs), dtype=torch.float32)
        if st.dim() != 3:
            raise ValueError(
                "share_obs for z-outcome must be [B,N,D], got %s"
                % (tuple(st.shape),)
            )
        ready, _, is_rep, _ = parse_group_token_pack(
            st, self.num_agents, "pairwise"
        )
        gm = comm_from_share_obs(st, self.num_agents, "pairwise")
        z = np.asarray(group_z).reshape(st.size(0), self.num_agents)
        events = []
        fail_r = None
        for thread_info in infos:
            if not thread_info or "catch_events" not in thread_info[0]:
                raise ValueError(
                    "group_token clique credit needs info catch_events"
                )
            events.append(thread_info[0]["catch_events"])
            fail_r = float(thread_info[0]["catch_fail_reward"])
        if fail_r is None:
            raise ValueError("catch_fail_reward missing from infos")
        out = clique_z_outcome(
            is_rep.detach().cpu().numpy(),
            ready.detach().cpu().numpy(),
            z,
            gm.detach().cpu().numpy(),
            events,
            float(self.all_args.catch_reward),
            fail_r,
        )
        return out.reshape(st.size(0), self.num_agents, 1)

    def _flush_token_stats(self):
        if not self.use_group_token:
            return {}
        n2 = self._tok_ready2
        ngo = self._tok_go_members
        out = {
            "p_go_given_ready2": (
                float(self._tok_go_ready2) / float(n2) if n2 else float("nan")
            ),
            "p_honor_given_go": (
                float(self._tok_honor) / float(ngo) if ngo else float("nan")
            ),
            "n_ready2": n2,
            "n_go_members": ngo,
        }
        self._tok_ready2 = 0
        self._tok_go_ready2 = 0
        self._tok_go_members = 0
        self._tok_honor = 0
        return out

    def _note_token_step(self, share_obs, actions, group_z):
        from mat.algorithms.utils.mat_comm_group_token import (
            HONOR_NONE,
            Z_GO,
            parse_group_token_pack,
        )

        if group_z is None:
            return
        st = torch.as_tensor(np.asarray(share_obs), dtype=torch.float32)
        if st.dim() != 3:
            raise ValueError("share_obs for token stats must be [B,N,D], got %s" % (tuple(st.shape),))
        z = np.asarray(group_z).reshape(st.size(0), self.num_agents)
        acts = np.asarray(actions).reshape(st.size(0), self.num_agents)
        ready, honor, is_rep, _ = parse_group_token_pack(
            st, self.num_agents, "pairwise"
        )
        ready = ready.cpu().numpy()
        honor = honor.cpu().numpy()
        is_rep = is_rep.cpu().numpy()
        for b in range(z.shape[0]):
            for i in range(self.num_agents):
                if is_rep[b, i] < 0.5:
                    continue
                if ready[b, i] + 1e-6 >= 2.0:
                    self._tok_ready2 += 1
                    if int(z[b, i]) == Z_GO:
                        self._tok_go_ready2 += 1
        go = z == Z_GO
        adjacent = honor != float(HONOR_NONE)
        self._tok_go_members += int((go & adjacent).sum())
        self._tok_honor += int((go & adjacent & (acts == honor.astype(np.int64))).sum())

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        out = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            np.concatenate(self.buffer.available_actions[step]),
        )
        if self.use_group_token:
            value, action, action_log_prob, rnn_state, rnn_state_critic, group_z = out
            self._group_z = np.array(np.split(_t2n(group_z), self.n_rollout_threads))
            self._note_token_step(
                self.buffer.share_obs[step],
                _t2n(action),
                _t2n(group_z),
            )
        else:
            value, action, action_log_prob, rnn_state, rnn_state_critic = out
            self._group_z = None
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
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

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]),
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
        if self.use_group_token:
            if self._group_z is None:
                raise ValueError("group_token collect did not store group_z")
            insert_kw["group_tokens"] = self._group_z
            insert_kw["group_token_outcome"] = self._z_outcome_from_step(
                self.buffer.share_obs[self.buffer.step],
                self._group_z,
                infos,
            )
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
            **insert_kw
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        act_names = [
            "catch+y",
            "catch+x",
            "catch-y",
            "catch-x",
            "move+y",
            "move+x",
            "move-y",
            "move-x",
            "stop",
        ]
        eval_won = 0
        eval_episode = 0
        eval_returns = []
        eval_joint = []
        eval_solo = []
        eval_prey = []
        joint_dump = []
        boards_dumped = 0
        t_in_ep = np.zeros(self.n_eval_rollout_threads, dtype=np.int32)
        running = np.zeros(self.n_eval_rollout_threads, dtype=np.float32)

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            eval_out = self.trainer.policy.act(
                np.concatenate(eval_share_obs),
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                np.concatenate(eval_available_actions),
                deterministic=True,
            )
            if self.use_group_token:
                eval_actions, eval_rnn_states, _ = eval_out
            else:
                eval_actions, eval_rnn_states = eval_out
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = (
                self.eval_envs.step(eval_actions)
            )
            running += eval_rewards[:, 0, 0]
            t_in_ep += 1

            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            for eval_i in range(self.n_eval_rollout_threads):
                info0 = eval_infos[eval_i][0]
                for ev in info0.get("catch_events") or []:
                    kind = ev["kind"]
                    agents = [int(a) for a in ev["agents"]]
                    acts = [int(a) for a in ev["acts"]]
                    act_s = ",".join(act_names[a] for a in acts)
                    rec = (
                        "ep=%d t=%d %s prey=%s agents=%s acts=[%s] "
                        "prey_xy=%s agent_xy=%s"
                        % (
                            eval_episode if not eval_dones_env[eval_i] else eval_episode,
                            int(t_in_ep[eval_i]),
                            kind,
                            ev["prey"],
                            agents,
                            act_s,
                            tuple(ev["prey_xy"]),
                            [tuple(xy) for xy in ev["agent_xy"]],
                        )
                    )
                    if kind == "joint":
                        joint_dump.append(rec)
                        board = ev.get("board") or ""
                        if board and boards_dumped < 3:
                            print("\nJOINT BOARD %s\n%s\n" % (rec, board), flush=True)
                            boards_dumped += 1

                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_returns.append(float(info0["episode_return"]))
                    eval_joint.append(float(info0["joint_catches"]))
                    eval_solo.append(float(info0["solo_fails"]))
                    eval_prey.append(float(info0["prey_left"]))
                    running[eval_i] = 0.0
                    t_in_ep[eval_i] = 0
                    if info0["won"]:
                        eval_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                win_rate = eval_won / float(eval_episode)
                n_with_joint = sum(1 for j in eval_joint if j > 0)
                print(
                    "eval n=%d win_rate=%.3f return=%.3f joint=%.3f "
                    "solo=%.3f prey_left=%.3f episodes_with_joint=%d/%d"
                    % (
                        eval_episode,
                        win_rate,
                        float(np.mean(eval_returns)),
                        float(np.mean(eval_joint)),
                        float(np.mean(eval_solo)),
                        float(np.mean(eval_prey)),
                        n_with_joint,
                        eval_episode,
                    ),
                    flush=True,
                )
                print(
                    "eval joint_catches per ep: min=%.0f p50=%.1f max=%.0f"
                    % (
                        float(np.min(eval_joint)),
                        float(np.median(eval_joint)),
                        float(np.max(eval_joint)),
                    ),
                    flush=True,
                )
                print("eval joint events (%d total), first 20:" % len(joint_dump), flush=True)
                for line in joint_dump[:20]:
                    print("  " + line, flush=True)
                if self.use_wandb:
                    wandb.log(
                        {
                            "eval_win_rate": win_rate,
                            "eval_return_mean": float(np.mean(eval_returns)),
                            "eval_joint_catches_mean": float(np.mean(eval_joint)),
                            "eval_solo_fails_mean": float(np.mean(eval_solo)),
                            "eval_prey_left_mean": float(np.mean(eval_prey)),
                        },
                        step=total_num_steps,
                    )
                break
