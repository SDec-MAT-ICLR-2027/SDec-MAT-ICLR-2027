"""MAT runner for Stag Hunt (explicit catch). Collection is SMACRunner's."""

import time

import numpy as np
import wandb

from mat.runner.shared.dicg_pp_runner import DICGPredatorPreyRunner


class StagHuntRunner(DICGPredatorPreyRunner):
    def _print_architecture(self):
        a = self.all_args
        n_params = sum(p.numel() for p in self.policy.transformer.parameters())
        comm = a.algorithm_name in ("mat_comm", "mat_comm_gru")
        print("\n" + "=" * 72)
        print(" %s  |  %s  |  seed %s  |  %s"
              % (a.algorithm_name, a.experiment_name, a.seed, self.device))
        print("=" * 72)
        print("   env        stag_hunt %dx%d  agents=%d stags=%d comm_radius=%d"
              % (a.grid_size, a.grid_size, a.n_agents, a.n_stags, a.comm_radius))
        print("   shapes     obs %d  share_obs %d  actions %d"
              % (self.policy.obs_dim, self.policy.share_obs_dim, self.policy.act_dim))
        print("   model      n_block %d, n_embd %d, n_head %d, %s params"
              % (a.n_block, a.n_embd, a.n_head, f"{n_params:,}"))
        if comm:
            print("   comm mask  pairwise N×N from env.get_comm_mask, "
                  "kind=%s encoder_critic_attn=%s"
                  % (getattr(a, "comm_mask_kind", None),
                     getattr(a, "encoder_critic_attn", "hard")))
        else:
            print("   comm mask  OFF - dense MAT attention")
        print("   ppo        epoch %d, minibatch %d, lr %s, entropy %s"
              % (a.ppo_epoch, a.num_mini_batch, a.lr, a.entropy_coef))
        print("=" * 72 + "\n")

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        catch_id = 5

        ep_captures, ep_steps, ep_returns, ep_misc = [], [], [], []
        catch_avail_n = 0
        catch_taken_n = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                avail = self.buffer.available_actions[step]
                acts = np.asarray(actions).reshape(self.n_rollout_threads, self.num_agents)
                catch_legal = avail[..., catch_id] > 0.5
                catch_avail_n += int(catch_legal.sum())
                catch_taken_n += int(((acts == catch_id) & catch_legal).sum())

                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                for thread_info in infos:
                    if thread_info[0]["episode_done"]:
                        ep_captures.append(thread_info[0]["joint_captures"])
                        ep_steps.append(thread_info[0]["episode_steps"])
                        ep_returns.append(thread_info[0]["episode_return"])
                        ep_misc.append(thread_info[0]["miscaptures"])

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
                end = time.time()
                n_done = len(ep_steps)
                if n_done > 0:
                    mean_caps = float(np.mean(ep_captures))
                    mean_len = float(np.mean(ep_steps))
                    mean_ret = float(np.mean(ep_returns))
                    mean_misc = float(np.mean(ep_misc))
                    cap_rate = float(np.mean(
                        [c / max(1, s) for c, s in zip(ep_captures, ep_steps)]
                    ))
                else:
                    mean_caps = mean_len = mean_ret = mean_misc = cap_rate = float("nan")
                p_catch = (float(catch_taken_n) / float(catch_avail_n)
                           if catch_avail_n > 0 else float("nan"))

                print(
                    f"\n Env stag_hunt Algo {self.algorithm_name} Exp {self.experiment_name} "
                    f"updates {episode}/{episodes} episodes, total num timesteps "
                    f"{total_num_steps}/{self.num_env_steps}, FPS "
                    f"{int(total_num_steps / (end - start))}.\n"
                )
                print(
                    f"  episodes_done={n_done}  caught_stags={mean_caps:.2f}/~4 "
                    f"miscaptures={mean_misc:.2f}  captures/step={cap_rate:.4f} "
                    f"P(catch|avail)={p_catch:.4f}  ep_len={mean_len:.1f}  "
                    f"return={mean_ret:.2f}"
                )

                env_metrics = {
                    "caught_stags_mean": mean_caps,
                    "miscaptures_mean": mean_misc,
                    "captures_per_step": cap_rate,
                    "p_catch_given_avail": p_catch,
                    "ep_length_mean": mean_len,
                    "return_mean": mean_ret,
                    "episodes_completed": float(n_done),
                }
                env_metrics.update(self._comm_stats())
                if self.use_wandb:
                    wandb.log(env_metrics, step=total_num_steps)
                else:
                    for k, v in env_metrics.items():
                        self.writter.add_scalars(k, {k: v}, total_num_steps)
                self.log_train(train_infos, total_num_steps)

                ep_captures, ep_steps, ep_returns, ep_misc = [], [], [], []
                catch_avail_n = 0
                catch_taken_n = 0

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
