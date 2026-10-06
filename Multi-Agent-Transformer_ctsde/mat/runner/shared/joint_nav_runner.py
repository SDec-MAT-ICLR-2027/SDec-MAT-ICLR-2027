"""Runner for joint-capture navigation.

Deliberately plain: no group-token bookkeeping, no obs-skip, no z-outcome
credit. The CT-SDE arm here is mat_comm with a soft_group_bias critic and
the ordinary learned START token per group, so there is nothing extra to
log beyond the comm graph itself.

Headline metrics:
  joint_captures  targets felled by >= k_req simultaneous capturers
  undershoots     capture attempts that fell short of k_req (the tax)
  targets_left    how much of the board survived the episode
  win_rate        fraction of episodes that cleared every target
"""
import time

import numpy as np
import torch

from mat.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class JointNavRunner(Runner):
    # local-obs layout: [vel(2), pos(2), rel_target(2), in_capture_range(1),
    # any_target_left(1), ...]. Absolute target = pos + rel_target, which lets
    # us group agents by the target they are actually on, using obs only — so
    # this works identically for mappo (no comm tail) and mat_comm_soft.
    POS = slice(2, 4)
    REL_TARGET = slice(4, 6)
    IN_RANGE = 6
    CAPTURE_ACTION = 5

    def __init__(self, config):
        super(JointNavRunner, self).__init__(config)
        self.explore_eps = float(
            getattr(self.all_args, "explore_group_capture_eps", 0.0)
        )
        self.k_req = int(self.all_args.k_req)
        self._forced_groups = 0
        self.use_group_latent = bool(
            int(getattr(self.all_args, "group_latent", 0) or 0)
        )
        self.latent_hold = bool(int(getattr(self.all_args, "latent_hold", 1)))
        self.n_latent = int(getattr(self.all_args, "n_latent", 3))
        self._z_prev = None
        self._prev_adj = None
        self._z_cur = None
        self._latent_counts = None
        self.bias_start = float(
            getattr(self.all_args, "latent_bias_scale_start", -1.0)
        )
        self.bias_end = float(
            getattr(self.all_args, "latent_bias_scale_end", -1.0)
        )
        self.bias_anneal_frac = float(
            getattr(self.all_args, "latent_bias_anneal_frac", 0.5)
        )
        self.bias_anneal = (
            self.use_group_latent and self.bias_start >= 0 and self.bias_end >= 0
        )
        self._cur_bias_scale = None
        self.latent_commit_eps = float(
            getattr(self.all_args, "latent_commit_eps", 0.0)
        )
        self._committed_groups = 0
        self.group_commit_eps = float(
            getattr(self.all_args, "group_commit_eps", 0.0)
        )
        self.commit_groups = str(getattr(self.all_args, "commit_groups",
                                         "capture"))
        if (str(getattr(self.all_args, "obs_mode", "")) == "radius"
                and self.commit_groups != "comm"):
            raise ValueError(
                "radius env has ONE grouping: the R-graph (commit_groups=comm); "
                "got commit_groups=%r" % self.commit_groups)
        # decoupled-PPO weighting of exploration-overridden samples
        self.offpolicy_weight = int(getattr(self.all_args, "offpolicy_weight", 0))
        if self.offpolicy_weight and int(getattr(self.all_args, "group_latent", 0)):
            raise ValueError("offpolicy_weight is implemented for the plain "
                             "(non-latent) collect path only")
        self._behav_w = None
        self._commit_eligible = None
        # subset commit: each member of a committing group joins w.p. q
        self.commit_member_prob = float(
            getattr(self.all_args, "commit_member_prob", 1.0))
        if not 0.0 < self.commit_member_prob <= 1.0:
            raise ValueError("commit_member_prob must be in (0, 1]")
        self.behav_w_mode = str(getattr(self.all_args, "behav_w_mode", "marginal"))
        self.behav_w_floor = float(getattr(self.all_args, "behav_w_floor", 0.0) or 0.0)
        if not 0.0 <= self.behav_w_floor <= 1.0:
            raise ValueError("behav_w_floor must be in [0, 1]")
        # autoregressive actor (action conditioning): mat_comm without the
        # self-only decoder. mappo / mat_comm_dec actors are independent.
        self._ar_actor = (self.all_args.algorithm_name == "mat_comm"
                          and not int(getattr(self.all_args, "decoder_self_only", 0)))
        self._commit_groups_step = None   # radius groups per thread, this step
        self._forced_mask = None          # [threads, agents] forced this step
        # optional linear decay of group_commit_eps (exploration schedule)
        self._eps0 = self.group_commit_eps
        self._eps_final = float(getattr(self.all_args, "group_commit_eps_final", -1.0))
        self._eps_decay_frac = float(getattr(self.all_args, "group_commit_decay_frac", 1.0))
        if self._eps_final >= 0.0 and self._eps0 <= 0.0:
            raise ValueError("group_commit_eps_final needs group_commit_eps > 0")
        self._randcommit_groups = 0
        self._print_architecture()

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

        Read from the env-appended comm tail of share_obs, NOT from the actor
        observation. The scaffold is TRAINING machinery: it must be identical
        in every cell of an observation ablation, otherwise changing what the
        actor sees silently changes how much correlated exploration it gets
        and the arms stop being comparable. Returns None when the env does not
        append the tail.
        """
        n = self.num_agents
        so = self.buffer.share_obs[step]
        if so.ndim < 3 or so.shape[-1] < n * n:
            return None
        if getattr(self.all_args, "comm_mask_kind", None) != "pairwise":
            return None
        return self._comm_adj(so)

    def _random_group_commit(self, obs, actions, step=None):
        """Arm-agnostic correlated exploration: the FAIR control.

        With probability `group_commit_eps` per group, every member takes the
        same uniformly random action. It uses no latent and ignores the
        policy, so it can be applied identically to MAPPO, plain CT-SDE and
        latent CT-SDE. Holding exploration fixed this way isolates execution:
        any gap that remains between arms comes from how they select actions,
        not from how they explore.

        Groups are agents inside capture range of the same target, recovered
        from obs (abs target = pos + rel_target) so MAPPO, which has no comm
        tail, groups exactly as the CT-SDE arms do at group_radius=capture.
        """
        if self.group_commit_eps <= 0.0:
            return actions, 0
        if self.commit_groups == "comm":
            return self._random_group_commit_comm(obs, actions, step)
        n_act = int(self.buffer.available_actions.shape[-1])
        acts = actions.copy()
        committed = 0
        elig = np.zeros((self.n_rollout_threads, self.num_agents), dtype=bool)
        for th in range(self.n_rollout_threads):
            o = obs[th]
            in_range = o[:, self.IN_RANGE] > 0.5
            idx = np.nonzero(in_range)[0]
            if len(idx) < 2:
                continue
            tgt = o[:, self.POS] + o[:, self.REL_TARGET]
            claimed = np.zeros(self.num_agents, dtype=bool)
            for i in idx:
                if claimed[i]:
                    continue
                same = idx[
                    (np.abs(tgt[idx] - tgt[i]).max(axis=1) < 1e-4) & ~claimed[idx]
                ]
                claimed[same] = True
                if len(same) >= 2:
                    elig[th, same] = True
                if len(same) >= 2 and np.random.rand() < self.group_commit_eps:
                    acts[th, same, 0] = np.random.randint(n_act)
                    committed += 1
        self._commit_eligible = elig
        return acts, committed

    def _capture_groups(self, o):
        """Agents in capture range of the same target (shared/local obs layout);
        the same grouping the capture-mode commit uses."""
        in_range = o[:, self.IN_RANGE] > 0.5
        idx = np.nonzero(in_range)[0]
        groups = []
        if len(idx) < 2:
            return groups
        tgt = o[:, self.POS] + o[:, self.REL_TARGET]
        claimed = np.zeros(self.num_agents, dtype=bool)
        for i in idx:
            if claimed[i]:
                continue
            same = idx[(np.abs(tgt[idx] - tgt[i]).max(axis=1) < 1e-4) & ~claimed[idx]]
            claimed[same] = True
            if len(same) >= 2:
                groups.append([int(x) for x in same])
        return groups

    @staticmethod
    def groups_from_adj(adj):
        """Connected components of size >= 2 of a boolean N x N adjacency."""
        n = adj.shape[0]
        seen = np.zeros(n, dtype=bool)
        groups = []
        for s in range(n):
            if seen[s]:
                continue
            comp, stack = [], [s]
            seen[s] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in np.nonzero(adj[u])[0]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            if len(comp) >= 2:
                groups.append(sorted(comp))
        return groups

    @staticmethod
    def radius_comm_groups(o, n_agents, off=4):
        """THE group structure of the radius env, from one env's radius obs.

        Agents are linked iff within sense_radius R (the same graph CT-SDE
        communicates over); groups are its connected components of size >= 2
        (a partition, which a group commitment needs). Read from the obs
        visibility flags: slot k of agent i is the k-th other agent in id
        order, flag at off + 3k + 2, where `off` is the length of the own-state
        prefix: 4 for [vel(2), pos(2)], 2 when obs_abs_pos=0 drops the
        position. Pure distance, no task knowledge.

        ONLY valid when the observation's peer visibility equals the true comm
        graph. Under a census ablation (obs_radius < sense_radius) it does not,
        and `_commit_adj` must be used instead.
        """
        n = n_agents
        adj = np.eye(n, dtype=bool)
        for i in range(n):
            others = [j for j in range(n) if j != i]
            for k, j in enumerate(others):
                if o[i, off + 3 * k + 2] > 0.5:
                    adj[i, j] = adj[j, i] = True
        seen = np.zeros(n, dtype=bool)
        groups = []
        for s in range(n):
            if seen[s]:
                continue
            comp, stack = [], [s]
            seen[s] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in np.nonzero(adj[u])[0]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            if len(comp) >= 2:
                groups.append(sorted(comp))
        return groups

    def _random_group_commit_comm(self, obs, actions, step=None):
        """Random group commitment over connected components of the
        agent-radius graph. Task-agnostic: grouping is pure distance, so it is
        identical for MAPPO and every CT-SDE arm.

        The adjacency comes from the env-appended comm tail of share_obs when
        that is available, and only otherwise from the observation visibility
        flags. Those flags equal the comm graph ONLY when the actor sees every
        peer inside sense_radius; under a census ablation they do not, which
        would silently switch the exploration scaffold off and confound the
        comparison. That case is refused rather than approximated.
        """
        n = self.num_agents
        n_act = int(self.buffer.available_actions.shape[-1])
        q = self.commit_member_prob
        acts = actions.copy()
        committed = 0
        elig = np.zeros((self.n_rollout_threads, n), dtype=bool)
        forced = np.zeros((self.n_rollout_threads, n), dtype=bool)
        groups_all = []
        adj_all = self._commit_adj(step) if step is not None else None
        if adj_all is None and self._census_ablated():
            raise RuntimeError(
                "group commit would be derived from the actor observation, but "
                "the census is ablated (obs_radius=%s < sense_radius=%s, "
                "obs_abs_pos=%s) so those flags no longer equal the comm graph. "
                "The exploration scaffold must read the env-appended comm tail; "
                "this arm does not provide one."
                % (getattr(self.all_args, "obs_radius", None),
                   getattr(self.all_args, "sense_radius", None),
                   getattr(self.all_args, "obs_abs_pos", 1)))
        off = self._obs_prefix_len()
        for th in range(self.n_rollout_threads):
            if adj_all is not None:
                groups = self.groups_from_adj(adj_all[th])
            else:
                groups = self.radius_comm_groups(obs[th], n, off)
            groups_all.append(groups)
            for comp in groups:
                elig[th, comp] = True
                if np.random.rand() < self.group_commit_eps:
                    c = np.random.randint(n_act)
                    members = [i for i in comp if q >= 1.0 or np.random.rand() < q]
                    acts[th, members, 0] = c
                    forced[th, members] = True
                    committed += 1
        self._commit_eligible = elig
        self._commit_groups_step = groups_all
        self._forced_mask = forced
        return acts, committed

    def _redecode_forced(self, step, acts):
        """Autoregressive actor + subset commit: re-decode the joint action
        with the forced members fixed, so every non-forced member conditions
        on the actions actually executed before it (not on the pre-override
        sample). Independent actors do not need this."""
        forced = self._forced_mask
        out = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            np.concatenate(self.buffer.available_actions[step]),
            forced_actions=acts[..., 0], forced_mask=forced,
        )
        new = np.array(np.split(_t2n(out[1]), self.n_rollout_threads))
        if not np.array_equal(new[forced], acts[forced]):
            raise RuntimeError("forced-prefix decode changed a forced action")
        return new

    def _latent_commit_override(self, z, actions):
        """Group-correlated exploration: commit a whole group to its mode.

        With probability `latent_commit_eps`, every member of a group takes
        the action its shared latent favours. This is the generalized form of
        the eps-override that broke the collapse: that one forced CAPTURE at
        groups with >= k_req agents in capture range -- naming the action, the
        readiness condition and the threshold. Here the action comes from the
        learned mode table, the group is just a connected component of the
        comm graph, and there is no threshold. Nothing task-specific remains.

        Why sparse rather than a logit bias: a bias distorts every logit on
        every step and trades directly against navigation. Measured across
        nine variants, degree and capture_acts were perfectly anti-correlated
        -- every setting that kept navigation (degree ~2.5) had the capture
        action extinct, every setting that produced firing (acts 33-94) had
        degree ~1.1. The override never had that problem because it left the
        policy untouched on 1-eps of steps.
        """
        if self.latent_commit_eps <= 0.0 or z is None:
            return actions, 0
        acts = actions.copy()
        head = self.trainer.policy.transformer.group_latent_head
        with torch.no_grad():
            mode_act = head.mode_action(
                torch.as_tensor(z, device=self.trainer.policy.device)
            )
        if mode_act is None:
            return actions, 0
        mode_act = mode_act.cpu().numpy()
        committed = 0
        for th in range(self.n_rollout_threads):
            adj = self._prev_adj[th] if self._prev_adj is not None else None
            if adj is None:
                continue
            seen = np.zeros(self.num_agents, dtype=bool)
            for i in range(self.num_agents):
                if seen[i]:
                    continue
                members = np.nonzero(adj[i])[0]
                seen[members] = True
                if len(members) < 2:
                    continue
                if np.random.rand() < self.latent_commit_eps:
                    acts[th, members, 0] = mode_act[th, i]
                    committed += 1
        return acts, committed

    def _anneal_bias(self, episode, episodes):
        """Linearly move the latent's bias strength during training.

        Navigation and correlation want opposite strengths: a weak bias keeps
        READY events high (7.40/ep) but gives zero correlation, a strong bias
        gives P(all fire|READY)=0.25 but collapses READY to 0.13/ep. Annealing
        lets the policy learn one before the other.
        """
        if not self.bias_anneal:
            return
        span = max(self.bias_anneal_frac * episodes, 1.0)
        f = min(1.0, episode / span)
        scale = self.bias_start + (self.bias_end - self.bias_start) * f
        self.trainer.policy.transformer.group_latent_head.bias_scale = scale
        self._cur_bias_scale = scale

    def _comm_adj(self, share_obs_step):
        """Decode the N x N group adjacency from the share_obs comm tail."""
        n = self.num_agents
        n2 = n * n
        tail = share_obs_step[:, 0, -n2:]
        return (tail.reshape(-1, n, n) > 0.5)

    def _latent_hold_flags(self, adj):
        """[threads, n_agents] bool: keep z where the group is unchanged.

        This is what makes the exploration COMMITTED over time rather than a
        fresh draw every step: a group keeps its mode for as long as it exists.
        """
        if not self.latent_hold or self._prev_adj is None or self._z_prev is None:
            return None
        if self._prev_adj.shape != adj.shape:
            return None
        return ~(self._prev_adj != adj).any(axis=-1)

    def _correlated_capture_override(self, obs, actions):
        """
        DIAGNOSTIC ONLY. With probability `explore_eps`, force every agent that
        is in capture range of the SAME target to emit CAPTURE together, when
        at least k_req of them are there.

        This is the crudest possible form of committed/correlated exploration.
        Independent exploration discovers a k-agent simultaneous action with
        probability ~p^k (~0.005 at k=3, p=1/6), which is why every run so far
        saw joint_captures=0.00 at every logged point and extinguished the
        action before observing a single success. This override answers one
        question: if a success is ever observed, can the learner reproduce it?

        Log-probs are recomputed for the overridden actions by the caller, so
        the PPO ratio stays valid.
        """
        if self.explore_eps <= 0.0:
            return actions, 0

        acts = actions.copy()
        forced = 0
        for th in range(self.n_rollout_threads):
            o = obs[th]
            in_range = o[:, self.IN_RANGE] > 0.5
            if int(in_range.sum()) < self.k_req:
                continue
            # absolute target position each agent is heading for
            tgt = o[:, self.POS] + o[:, self.REL_TARGET]
            idx = np.nonzero(in_range)[0]
            claimed = np.zeros(self.num_agents, dtype=bool)
            for i in idx:
                if claimed[i]:
                    continue
                same = idx[
                    (np.abs(tgt[idx] - tgt[i]).max(axis=1) < 1e-4) & ~claimed[idx]
                ]
                if len(same) >= self.k_req and np.random.rand() < self.explore_eps:
                    acts[th, same, 0] = self.CAPTURE_ACTION
                    forced += 1
                claimed[same] = True
        return acts, forced

    def _relogprob(self, step, actions, group_z=None):
        """Log-probs of `actions` under the current policy (PPO ratio must be 1).

        The latent path needs the drawn z, otherwise evaluate_actions cannot
        reconstruct the same decode the actions were sampled under.
        """
        out = self.trainer.policy.evaluate_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            actions.reshape(-1, actions.shape[-1]),
            np.concatenate(self.buffer.masks[step]),
            np.concatenate(self.buffer.available_actions[step]),
            group_z=(None if group_z is None
                     else group_z.reshape(-1, 1)),
        )
        log_probs = out[1]
        return np.array(np.split(_t2n(log_probs), self.n_rollout_threads))

    def _print_architecture(self):
        a = self.all_args
        arm = getattr(a, "arm_name", a.algorithm_name)
        print("=" * 72)
        print("[joint_nav] arm=%s (algorithm_name=%s)" % (arm, a.algorithm_name))
        if a.algorithm_name in ("mat_comm",):
            print(
                "  encoder: group-masked (target cliques), "
                "critic: %s" % getattr(a, "encoder_critic_attn", "?")
            )
            if int(getattr(a, "decoder_self_only", 0)):
                print("  decoder: SELF-ONLY (no action conditioning) -- "
                      "observation-sharing DE control")
            else:
                print(
                    "  decoder: per-querier AR over the group, ordinary "
                    "learned START token"
                )
            print(
                "  group_token=%s use_obs_skip=%s  (both off = plain CT-SDE)"
                % (getattr(a, "group_token", 0), getattr(a, "use_obs_skip", 0))
            )
        else:
            print("  independent actors (dec_actor=%s), no joint action" % a.dec_actor)
        print(
            "  obs_mode=%s  k_req=%s  num_targets=%s"
            % (a.obs_mode, a.k_req, a.num_targets)
        )
        print("=" * 72, flush=True)

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
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        ep_ret, ep_len, ep_cap, ep_under, ep_left, ep_won, ep_acts, ep_over = (
            [], [], [], [], [], [], [], []
        )

        for episode in range(episodes):
            if self._eps_final >= 0.0:
                span = max(1.0, self._eps_decay_frac * episodes)
                frac = min(1.0, episode / span)
                self.group_commit_eps = (
                    self._eps0 + (self._eps_final - self._eps0) * frac)
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            self._anneal_bias(episode, episodes)

            for step in range(self.episode_length):
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)
                obs, share_obs, rewards, dones, infos, available_actions = (
                    self.envs.step(actions)
                )

                for thread_info in infos:
                    if thread_info[0].get("episode_done"):
                        ep_ret.append(thread_info[0]["episode_return"])
                        ep_len.append(thread_info[0]["episode_steps"])
                        ep_cap.append(thread_info[0]["joint_captures"])
                        ep_under.append(thread_info[0]["undershoots"])
                        ep_over.append(thread_info[0].get("overkills", 0))
                        ep_left.append(thread_info[0]["targets_left"])
                        ep_acts.append(thread_info[0]["capture_actions"])
                        ep_won.append(float(thread_info[0]["won"]))

                self.insert(
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
                    )
                )

            self.compute()
            train_infos = self.train()

            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save(episode)

            if episode % self.log_interval == 0:
                end = time.time()
                n_done = len(ep_ret)
                if n_done > 0:
                    m_ret = float(np.mean(ep_ret))
                    m_len = float(np.mean(ep_len))
                    m_cap = float(np.mean(ep_cap))
                    m_und = float(np.mean(ep_under))
                    m_left = float(np.mean(ep_left))
                    m_acts = float(np.mean(ep_acts))
                    win = float(np.mean(ep_won))
                else:
                    m_ret = m_len = m_cap = m_und = m_left = m_acts = win = float("nan")

                arm = getattr(self.all_args, "arm_name", self.algorithm_name)
                print(
                    "\n Env joint_nav Arm %s Exp %s updates %d/%d episodes, "
                    "total num timesteps %d/%d, FPS %d.\n"
                    % (
                        arm,
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
                    "joint_captures=%.2f  undershoots=%.2f  capture_acts=%.2f  "
                    "targets_left=%.2f  win_rate=%.3f"
                    % (n_done, m_ret, m_len, m_cap, m_und, m_acts, m_left, win),
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
                    "return_mean": np.array([m_ret]),
                    "ep_length_mean": np.array([m_len]),
                    "joint_captures_mean": np.array([m_cap]),
                    "undershoots_mean": np.array([m_und]),
                    "capture_actions_mean": np.array([m_acts]),
                    "targets_left_mean": np.array([m_left]),
                    "win_rate": np.array([win]),
                    "episodes_completed": np.array([float(n_done)]),
                }
                if getattr(self.all_args, "k_max", None) is not None:
                    m_over = float(np.mean(ep_over)) if n_done else float("nan")
                    print("  overkills=%.2f" % m_over, flush=True)
                    env_log["overkills_mean"] = np.array([m_over])
                if self.latent_commit_eps > 0.0:
                    per_ep = self._committed_groups / max(n_done, 1)
                    print("  committed_groups=%.2f per episode (eps=%.2f)"
                          % (per_ep, self.latent_commit_eps), flush=True)
                    env_log["committed_groups"] = np.array([float(per_ep)])
                    self._committed_groups = 0
                if self.group_commit_eps > 0.0:
                    per_ep = self._randcommit_groups / max(n_done, 1)
                    print("  random_committed_groups=%.2f per episode (eps=%.2f)"
                          % (per_ep, self.group_commit_eps), flush=True)
                    env_log["random_committed_groups"] = np.array([float(per_ep)])
                    self._randcommit_groups = 0
                env_log["group_commit_eps"] = np.array(
                    [float(self.group_commit_eps)])
                if self._cur_bias_scale is not None:
                    print("  latent_bias_scale=%.3f" % self._cur_bias_scale,
                          flush=True)
                    env_log["latent_bias_scale"] = np.array(
                        [float(self._cur_bias_scale)]
                    )
                if self.use_group_latent and self._latent_counts is not None:
                    tot = max(int(self._latent_counts.sum()), 1)
                    frac = self._latent_counts / tot
                    print("  latent_usage=%s (collapse if one mode ~1.0)"
                          % np.round(frac, 3).tolist(), flush=True)
                    for i, f in enumerate(frac):
                        env_log["latent_usage_%d" % i] = np.array([float(f)])
                    self._latent_counts = np.zeros(self.n_latent, dtype=np.int64)
                if self.explore_eps > 0.0:
                    per_ep = self._forced_groups / max(n_done, 1)
                    print(
                        "  forced_capture_groups=%d (%.2f per episode) eps=%.3f"
                        % (self._forced_groups, per_ep, self.explore_eps),
                        flush=True,
                    )
                    env_log["forced_capture_groups"] = np.array([float(per_ep)])
                    self._forced_groups = 0
                for k, v in comm_stats.items():
                    env_log[k] = np.array([v])
                self.log_env(env_log, total_num_steps)
                ep_ret, ep_len, ep_cap, ep_under, ep_left, ep_won, ep_acts, ep_over = (
                    [], [], [], [], [], [], [], []
                )

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        self._z_prev = None
        self._prev_adj = None
        obs, share_obs, available_actions = self.envs.reset()
        if not self.use_centralized_V:
            share_obs = obs
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        if self.use_group_latent:
            adj = self._comm_adj(self.buffer.share_obs[step])
            hold = self._latent_hold_flags(adj)
            out = self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]),
                np.concatenate(self.buffer.available_actions[step]),
                z_prev=(None if self._z_prev is None
                        else self._z_prev.reshape(-1)),
                z_hold=(None if hold is None else hold.reshape(-1)),
            )
            value, action, action_log_prob, rnn_state, rnn_state_critic, z = out
            split = lambda x: np.array(np.split(_t2n(x), self.n_rollout_threads))
            z_np = split(z).reshape(self.n_rollout_threads, self.num_agents)
            self._z_cur = z_np
            self._z_prev = z_np
            self._prev_adj = adj
            if self._latent_counts is None:
                self._latent_counts = np.zeros(self.n_latent, dtype=np.int64)
            vals, cnts = np.unique(z_np, return_counts=True)
            for v, c in zip(vals, cnts):
                if 0 <= int(v) < self.n_latent:
                    self._latent_counts[int(v)] += int(c)
            values = split(value)
            actions = split(action)
            logp = split(action_log_prob)
            if self.latent_commit_eps > 0.0:
                committed_actions, n_c = self._latent_commit_override(
                    z_np, actions
                )
                if n_c:
                    self._committed_groups += n_c
                    # values come from the critic and do not depend on the
                    # action; only the log-probs need recomputing so the PPO
                    # ratio starts at 1
                    logp = self._relogprob(step, committed_actions, z_np)
                    actions = committed_actions
            if self.group_commit_eps > 0.0:
                rc_actions, n_r = self._random_group_commit(
                    self.buffer.obs[step], actions, step
                )
                if n_r:
                    self._randcommit_groups += n_r
                    logp = self._relogprob(step, rc_actions, z_np)
                    actions = rc_actions
            return (values, actions, logp,
                    split(rnn_state), split(rnn_state_critic))
        (
            value,
            action,
            action_log_prob,
            rnn_state,
            rnn_state_critic,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            np.concatenate(self.buffer.available_actions[step]),
        )
        split = lambda x: np.array(np.split(_t2n(x), self.n_rollout_threads))
        values = split(value)
        actions = split(action)
        action_log_probs = split(action_log_prob)
        rnn_states = split(rnn_state)
        rnn_states_critic = split(rnn_state_critic)

        if self.explore_eps > 0.0:
            forced_actions, n_forced = self._correlated_capture_override(
                self.buffer.obs[step], actions
            )
            if n_forced:
                self._forced_groups += n_forced
                # values come from the critic and do not depend on the action,
                # so only the log-probs need recomputing
                action_log_probs = self._relogprob(step, forced_actions)
                actions = forced_actions

        if self.group_commit_eps > 0.0:
            rc_actions, n_r = self._random_group_commit(
                self.buffer.obs[step], actions, step
            )
            if n_r:
                self._randcommit_groups += n_r
                if (self._ar_actor and self.commit_member_prob < 1.0
                        and self.commit_groups == "comm"):
                    rc_actions = self._redecode_forced(step, rc_actions)
                action_log_probs = self._relogprob(step, rc_actions)
                actions = rc_actions
            if self.offpolicy_weight:
                self._store_behaviour_weights(step, action_log_probs, actions)
        elif self.offpolicy_weight and self._behav_w is not None:
            # eps decayed to 0: data is fully on-policy, weights must be 1
            # (never reuse weights left over from an earlier step)
            self._behav_w[step] = 1.0

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    @staticmethod
    def behaviour_weights(logp, eligible, eps, n_act):
        """Decoupled-PPO behaviour weight w = min(1, pi / mu).

        The data were collected by the mixture mu = (1-eps)*pi + eps/|A| for
        agents that were eligible for a group commitment this step (a
        commitment replaces the group's actions with one uniformly random
        action), and by pi itself otherwise. Uses only eps, |A| and the
        policy's own probabilities -- nothing task-specific.
        """
        pi = np.exp(logp)
        mu = np.where(eligible, (1.0 - eps) * pi + eps / float(n_act), pi)
        return np.minimum(1.0, pi / np.maximum(mu, 1e-12))

    @staticmethod
    def arm_behaviour_weights(logp, acts, groups, eps, q, n_act, ar):
        """Decoupled-PPO weight w = min(1, pi / mu) with mu matched to the
        arm's own factorization, under the subset-commit mixture (w.p. eps a
        group draws a shared action c uniformly; each member independently
        takes it w.p. q, the rest act from the policy).

        logp, acts [N] (one env): executed actions and their teacher-forced
        log-probs. groups: radius components (sorted ids = decode order).
        ar=False (independent actors): marginal
            mu_i = (1 - eps*q) * pi_i + eps*q/|A|.
        ar=True (autoregressive actor, pi_i = pi(a_i | executed a_<i)):
        exact conditional mu_i = P(a_{S+i}) / P(a_S), S = group members
        before i, with
            P(a_S) = (1-eps) prod_S pi_j
                     + eps * mean_c prod_S [q 1(a_j=c) + (1-q) pi_j].
        Agents outside every group are on-policy (w = 1).
        """
        pi = np.exp(np.asarray(logp, dtype=np.float64).reshape(-1))
        acts = np.asarray(acts).reshape(-1)
        w = np.ones_like(pi)
        for g in groups:
            if not ar:
                mu = (1.0 - eps * q) * pi[g] + eps * q / float(n_act)
                w[g] = np.minimum(1.0, pi[g] / np.maximum(mu, 1e-300))
                continue
            a_nc = 1.0                        # prod pi over the prefix
            b = np.ones(n_act)                # per-c commit-branch product
            p_prev = 1.0
            for i in g:
                a_nc *= pi[i]
                onehot = np.zeros(n_act)
                onehot[int(acts[i])] = 1.0
                b = b * (q * onehot + (1.0 - q) * pi[i])
                p_new = (1.0 - eps) * a_nc + eps * b.mean()
                mu = p_new / max(p_prev, 1e-300)
                w[i] = min(1.0, pi[i] / max(mu, 1e-300))
                p_prev = p_new
        return w

    def _store_behaviour_weights(self, step, logp, acts=None):
        if self._behav_w is None:
            self._behav_w = np.ones_like(self.buffer.action_log_probs)
        n_act = int(self.buffer.available_actions.shape[-1])
        if self.behav_w_mode == "arm" and self.commit_groups == "comm":
            w = np.ones_like(logp)
            for th in range(self.n_rollout_threads):
                w[th, :, 0] = self.arm_behaviour_weights(
                    logp[th, :, 0], acts[th, :, 0], self._commit_groups_step[th],
                    self.group_commit_eps, self.commit_member_prob, n_act,
                    self._ar_actor)
            self._behav_w[step] = w
        else:
            if self.commit_member_prob < 1.0:
                raise ValueError("commit_member_prob < 1 needs --behav_w_mode arm")
            elig = self._commit_eligible[..., None]
            self._behav_w[step] = self.behaviour_weights(
                logp, elig, self.group_commit_eps, n_act)
        if self.behav_w_floor > 0.0:
            self._behav_w[step] = np.maximum(self._behav_w[step], self.behav_w_floor)
        self.buffer.behav_w = self._behav_w

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
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
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

        active_masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
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
        if self.use_group_latent:
            if self._z_cur is None:
                raise ValueError("group_latent collect did not store z")
            insert_kw["group_tokens"] = self._z_cur.reshape(
                self.n_rollout_threads, self.num_agents, 1
            ).astype(np.float32)
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
        """Greedy (deterministic) evaluation over `n_eval_rollout_threads`."""
        self.trainer.prep_rollout()
        eval_obs, eval_share_obs, eval_avail = self.eval_envs.reset()
        n_thr = self.n_eval_rollout_threads

        eval_rnn_states = np.zeros(
            (n_thr, self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        eval_masks = np.ones((n_thr, self.num_agents, 1), dtype=np.float32)

        caps, unders, lefts, wons, rets = [], [], [], [], []
        for _ in range(self.episode_length):
            action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_share_obs),
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                np.concatenate(eval_avail),
                deterministic=True,
            )
            actions = np.array(np.split(_t2n(action), n_thr))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), n_thr))

            eval_obs, eval_share_obs, _, eval_dones, eval_infos, eval_avail = (
                self.eval_envs.step(actions)
            )
            for thread_info in eval_infos:
                if thread_info[0].get("episode_done"):
                    caps.append(thread_info[0]["joint_captures"])
                    unders.append(thread_info[0]["undershoots"])
                    lefts.append(thread_info[0]["targets_left"])
                    rets.append(thread_info[0]["episode_return"])
                    wons.append(float(thread_info[0]["won"]))

            dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[dones_env == True] = 0.0
            eval_masks = np.ones((n_thr, self.num_agents, 1), dtype=np.float32)
            eval_masks[dones_env == True] = 0.0

        if not caps:
            print("  [eval] no episode completed in the eval window", flush=True)
            return
        stats = {
            "test_joint_captures_mean": np.array([float(np.mean(caps))]),
            "test_undershoots_mean": np.array([float(np.mean(unders))]),
            "test_targets_left_mean": np.array([float(np.mean(lefts))]),
            "test_return_mean": np.array([float(np.mean(rets))]),
            "test_win_rate": np.array([float(np.mean(wons))]),
        }
        print(
            "  [eval] episodes=%d  joint_captures=%.2f  undershoots=%.2f  "
            "targets_left=%.2f  win_rate=%.3f"
            % (
                len(caps),
                float(np.mean(caps)),
                float(np.mean(unders)),
                float(np.mean(lefts)),
                float(np.mean(wons)),
            ),
            flush=True,
        )
        self.log_env(stats, total_num_steps)
