import numpy as np
import torch
import torch.nn as nn
from mat.utils.util import get_gard_norm, huber_loss, mse_loss
from mat.utils.valuenorm import ValueNorm
from mat.algorithms.utils.util import check


class MATTrainer:
    """
    Trainer class for MAT to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 num_agents,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = num_agents

        self.clip_param = args.clip_param
        # decoupled PPO (Hilton et al. 2021): advantages are multiplied by the
        # behaviour weight min(1, pi/mu) the runner stores in buffer.behav_w
        self.offpolicy_weight = int(getattr(args, "offpolicy_weight", 0) or 0)
        self._w_calls = 0
        # OptiMAPPO-style optimistic advantages (Zhao et al. 2024)
        self.optimistic_adv = int(getattr(args, "optimistic_adv", 0) or 0)
        self.opt_leaky_rate = float(getattr(args, "opt_leaky_rate", 0.0) or 0.0)
        if self.optimistic_adv and not (0.0 <= self.opt_leaky_rate <= 1.0):
            raise ValueError("opt_leaky_rate must be in [0, 1]")
        self.gamma = float(getattr(args, "gamma", 0.99))
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.dec_actor = args.dec_actor
        self.z_loss_ready_balance = int(getattr(args, "z_loss_ready_balance", 0))
        if self.z_loss_ready_balance not in (0, 1):
            raise ValueError(
                "z_loss_ready_balance must be 0 or 1, got %r"
                % self.z_loss_ready_balance
            )
        self.catch_min = int(getattr(args, "catch_min", 0) or 0)

        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)

        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        # if self._use_value_active_masks and not self.dec_actor:
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    @staticmethod
    def optimistic_transform(adv, leaky_rate):
        """OptiMAPPO (Zhao et al. 2024): A <- max(leaky_rate * A, A).
        Positive advantages unchanged; negative ones scaled by leaky_rate."""
        return np.maximum(leaky_rate * adv, adv)

    @staticmethod
    def ppo_surrogate(log_ratio, adv, clip):
        """Per-element PPO clipped surrogate; returns (ratio, objective)."""
        ratio = torch.exp(log_ratio)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
        return ratio, torch.min(surr1, surr2)

    def ppo_update(self, sample):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        commits_batch = None
        z_prev_batch = None
        group_z_batch = None
        z_outcome_batch = None
        if getattr(self.policy, "use_commitment", False):
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, commits_batch, z_prev_batch = sample
        elif getattr(self.policy, "use_group_token", False):
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, group_z_batch, z_outcome_batch = sample
        elif getattr(self.policy, "use_group_latent", False):
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, group_z_batch = sample
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        # Recurrent policies need the chunk length to fold the time-major batch
        # back into [L, mb, ...] and replay the GRU through it.
        seq_len = self.data_chunk_length if self.policy.is_recurrent else None

        log_z = None
        eval_out = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            seq_len,
            commits=commits_batch,
            z_prev=z_prev_batch,
            group_z=group_z_batch,
        )
        ent_z = None
        mi_disc = None
        mi_bonus = None
        if getattr(self.policy, "use_group_token", False):
            values, action_log_probs, dist_entropy, log_z = eval_out
        elif getattr(self.policy, "use_group_latent", False):
            (values, action_log_probs, dist_entropy,
             log_z, ent_z, mi_disc, mi_bonus) = eval_out
        else:
            values, action_log_probs, dist_entropy = eval_out
        # actor update
        log_ratio = action_log_probs - old_action_log_probs_batch
        if self.offpolicy_weight:
            # a forced action can have a tiny proximal prob; its weight is
            # tiny too, but exp() must not overflow to inf before the product
            log_ratio = torch.clamp(log_ratio, max=20.0)
        imp_weights, surr = self.ppo_surrogate(
            log_ratio, adv_targ, self.clip_param)

        if self._use_policy_active_masks:
            policy_loss = (-torch.sum(surr,
                                      dim=-1,
                                      keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.sum(surr, dim=-1, keepdim=True).mean()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        z_loss = action_log_probs.new_zeros(())
        # group_token uses a hand-defined outcome; the group latent uses the
        # ordinary team advantage and is handled separately below.
        if log_z is not None and getattr(self.policy, "use_group_token", False):
            if z_outcome_batch is None:
                raise ValueError("group_token PPO needs z_outcome_batch")
            z_out = check(z_outcome_batch).to(**self.tpdv).reshape_as(log_z)
            if self.z_loss_ready_balance == 1:
                if self.catch_min < 2:
                    raise ValueError(
                        "z_loss_ready_balance=1 requires catch_min >= 2"
                    )
                from mat.algorithms.utils.mat_comm_group_token import (
                    balance_z_outcome_by_ready,
                    parse_group_token_pack,
                )
                so = check(share_obs_batch).to(**self.tpdv)
                if so.dim() != 2:
                    raise ValueError(
                        "share_obs_batch must be [B*N, D], got %s"
                        % (tuple(so.shape),)
                    )
                n_ag = int(self.num_agents)
                if so.size(0) % n_ag != 0:
                    raise ValueError(
                        "share_obs rows %d not divisible by n_agents %d"
                        % (so.size(0), n_ag)
                    )
                so_t = so.reshape(-1, n_ag, so.size(-1))
                ready, _, _, _ = parse_group_token_pack(
                    so_t, n_ag, "pairwise"
                )
                ready = ready.reshape_as(z_out)
                z_out = balance_z_outcome_by_ready(z_out, ready, self.catch_min)
            z_loss = -(log_z * z_out).sum() / active_masks_batch.sum()

        if mi_disc is not None:
            # Group latent: credited with the SAME team advantage as actions,
            # not a task-defined outcome. Entropy stops it collapsing early;
            # the MI term keeps different modes meaning different joint
            # behaviours (minimizing the discriminator's cross-entropy
            # maximizes the variational lower bound on MI(z ; joint action)).
            adv_z = check(adv_targ).to(**self.tpdv).reshape_as(log_z)
            denom = active_masks_batch.sum()
            # The MI bonus is an intrinsic REWARD on z (detached), added to the
            # team advantage. Adding the discriminator CE straight to the loss
            # instead does nothing to the head: actions are discrete samples,
            # so there is no gradient path from the CE back to the latent.
            # Variational MI bonus: I(z;tau) = E[log q(z|tau)] - E[log p(z)].
            # `mi_bonus` is only the first term. Using it alone rewards
            # PREDICTABILITY, and the most predictable z is whichever is
            # already most used — a rich-get-richer loop. Measured at
            # ent=0.05: mi=0.0 gave usage [0.814, 0.127, 0.059] while mi=0.5
            # and mi=5.0 both collapsed outright ([~0,0,0.998] / [0,1,0]).
            # Subtracting log p(z) (the latent's own log-prob) restores the
            # entropy term and penalizes over-used modes. This is the
            # DIAYN/MAVEN form of the bound.
            mi_term = mi_bonus.detach() - log_z.detach()
            z_adv = adv_z.detach() + self.policy.latent_mi_coef * mi_term
            z_loss = (
                -(log_z * z_adv).sum() / denom
                - ent_z.sum() / denom * self.policy.latent_entropy_coef
                + mi_disc
            )

        loss = (
            policy_loss
            - dist_entropy * self.entropy_coef
            + value_loss * self.value_loss_coef
            + z_loss
        )

        self.policy.optimizer.zero_grad()
        loss.backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.transformer.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.policy.transformer.parameters())

        self.policy.optimizer.step()

        return value_loss, grad_norm, policy_loss, dist_entropy, grad_norm, imp_weights

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        adv_raw = buffer.advantages
        adv_pos_frac = float((adv_raw >= 0).mean())
        if self.optimistic_adv:
            # OptiMAPPO order: leaky transform first, then standardize
            adv_raw = self.optimistic_transform(adv_raw, self.opt_leaky_rate)
        advantages_copy = adv_raw.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (adv_raw - mean_advantages) / (std_advantages + 1e-5)
        w_stats = None
        if self.offpolicy_weight:
            w = getattr(buffer, "behav_w", None)
            if w is None:
                raise ValueError("offpolicy_weight=1 but the runner stored no "
                                 "behaviour weights (is group_commit_eps > 0?)")
            if w.shape != advantages.shape:
                raise ValueError("behav_w shape %s != advantages %s"
                                 % (w.shape, advantages.shape))
            advantages = advantages * w
            w_stats = (float(w.mean()), float((w < 0.5).mean()))
            self._w_calls += 1
            if self._w_calls % 5 == 1:
                print("  [offpolicy_weight] mean w %.3f  share w<0.5 %.4f"
                      % w_stats, flush=True)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self.policy.is_recurrent:
                # Contiguous chunks: shuffled single timesteps would give the GRU
                # no predecessor to carry state from.
                data_generator = buffer.recurrent_generator_transformer(
                    advantages, self.num_mini_batch, self.data_chunk_length)
            else:
                data_generator = buffer.feed_forward_generator_transformer(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
        if w_stats is not None:
            train_info['behav_w_mean'], train_info['behav_w_low_frac'] = w_stats
        train_info['adv_pos_frac'] = adv_pos_frac

        return train_info

    def prep_training(self):
        self.policy.train()

    def prep_rollout(self):
        self.policy.eval()
