import argparse


def _str2bool(v):
    """argparse type for flags that take an explicit true/false value."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if v.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("expected true/false, got %r" % (v,))


def get_config():
    """
    The configuration parser for hyper-parameters of all environment.
    Please reach each `scripts/train/<env>_runner.py` file to find private hyper-parameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["mat", "mat_dec"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch 
        --cuda
            by default True, will use GPU to train; or else will use CPU; 
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.
    
    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or else will use concatenated local obs.
    
    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer. 
    
    Network parameters:
        --share_policy
            by default True, all agents will share the same network; set to make training agents use different policies. 
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards. 
        --use_valuenorm
            by default True, use running mean and std to normalize rewards. 
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs. 
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.
    
    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)
    
    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss 
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm 
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.  
        --huber_delta <float>
            coefficient of huber loss.  
    
    PPG parameters:
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)
    
    Run parameters：
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate
    
    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.
    
    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.
    
    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.
    
    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    # `mat_comm_soft` is an arm name, not a separate network: train_joint_nav.py
    # normalizes it to `mat_comm` with encoder_critic_attn=soft_group_bias and
    # group_token/use_obs_skip forced off (plain CT-SDE).
    parser.add_argument("--algorithm_name", type=str,
                        default='mat', choices=["mat", "mat_dec", "mappo", "mat_encoder", "mat_decoder", "mat_gru",
                                                "mat_comm", "mat_comm_gru", "mat_comm_soft",
                                                "mat_comm_dec"])

    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=32,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name", type=str, default='xxx',help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", action='store_false', default=False, help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")

    # env parameters
    parser.add_argument("--env_name", type=str, default='StarCraft2', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument("--share_policy", action='store_false',
                        default=True, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks") 
    parser.add_argument("--layer_N", type=int, default=2,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=True, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_true',
                        default=False, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        "--commit_entropy_coef", type=float, default=None,
        help="Entropy coefficient on the commitment head H_z. Required when "
             "--commitment 1 (no silent default). Action entropy still uses "
             "--entropy_coef. 0 disables the H_z bonus.",
    )
    parser.add_argument(
        "--commit_action_blind", type=int, default=0,
        help="mat_comm + commitment only. 1=action decoder encodes food-blind "
             "obs (first 3*n_targets dims zeroed) and the action head gets "
             "own-z rel_pos. z head still sees full obs. 0=off.",
    )
    parser.add_argument(
        "--commit_horizon", type=int, default=1,
        help="Resample z every k env steps; hold it in between. 1=every step "
             "(default). k>1 is a hard commitment horizon. Requires "
             "--commitment 1.",
    )
    parser.add_argument(
        "--commit_critic_z", type=int, default=0,
        help="mat_comm + commitment only. 1=concat each agent's detached "
             "one-hot z onto the critic token. Actor stream unchanged. 0=off.",
    )
    parser.add_argument(
        "--commit_critic_pair", type=int, default=0,
        help="mat_comm + commitment only. 1=replace scalar group_bias with a "
             "small MLP over [G_ij, same_target(i,j)]. Mutually exclusive "
             "with --commit_critic_z. 0=off.",
    )
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=100, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=5, help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25, help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")


    # add for transformer
    parser.add_argument("--encode_state", action='store_true', default=False)
    parser.add_argument("--n_block", type=int, default=1)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--dec_actor", action='store_true', default=False)
    parser.add_argument("--share_actor", action='store_true', default=False)
    parser.add_argument(
        "--obs_agent_id", type=_str2bool, default=False, metavar="{true,false}",
        help="false (default): the wrapper passes the env's observation through "
             "untouched. true: append an n_agents-wide one-hot agent index to "
             "every observation, matching pymarl's obs_agent_id for the "
             "mappo/commnet/tarmac baselines. Only obs is touched -- share_obs "
             "is left alone, because mat_comm reads the comm bits by slicing "
             "its tail. NOTE this deliberately BREAKS Pickup Consensus: a "
             "shared-parameter policy can then learn 'lift iff my index < K' "
             "and score ~1.0, which is the whole point of the ID ablation. "
             "Changing it changes obs_dim, so ID-on and ID-off checkpoints are "
             "not interchangeable via --model_dir.")
    parser.add_argument(
        "--eval_deterministic", type=_str2bool, default=True,
        metavar="{true,false}",
        help="true (default, MAT's original behaviour): eval takes the argmax "
             "action. false: eval samples from the policy, as pymarl's "
             "soft_policies selector does in test_mode. This matters a great "
             "deal for a DECENTRALISED shared-parameter actor (mat_dec) on a "
             "benchmark where every agent sees the same observation: argmax "
             "makes all N agents pick the same action, so an exact-K task can "
             "only ever score 0. An autoregressive decoder (mat_comm) breaks "
             "that tie itself and scores fine either way. Keep it identical "
             "across the algorithms you intend to compare.")
    parser.add_argument(
        "--no_raw_obs_layernorm", action="store_true", default=False,
        help="mat_comm/mat_comm_gru (and mat/mat_dec): drop the LayerNorm "
             "applied to the RAW observation at the front of the encoder. That "
             "LayerNorm is rank-degenerate for a low-dimensional obs: on a "
             "vector holding only two distinct values it divides out the gap "
             "magnitude and keeps only its sign, so on Pickup Consensus "
             "(obs=[K/N, N/n_max]) every K collapses to the same +-1 pattern "
             "and K never reaches the decoder. The encoder's LayerNorm(n_embd) "
             "still runs right after the projection. Off by default: MAT's "
             "original behaviour.")
    parser.add_argument(
        "--comm_mask_kind", type=str, default="bits",
        choices=["bits", "pairwise"],
        help="How mat_comm reads the graph from share_obs: Aerial per-agent "
             "bits (default) or a flattened N×N adjacency (DICG predator-prey). "
             "train_dicg_pp.py forces pairwise for mat_comm; do not override.",
    )
    parser.add_argument(
        "--encoder_critic_attn", type=str, default="hard",
        choices=["hard", "soft_group_bias"],
        help="mat_comm encoder attention for the value head. hard (default): "
             "one group-masked pass shared by actor and critic. "
             "soft_group_bias: second encoder pass with the same weights, "
             "unmasked + learned in-group logit bias; decoder still hard. "
             "Not implemented for mat_comm_gru.",
    )
    parser.add_argument(
        "--group_latent", type=int, default=0,
        help="mat_comm only. 1=group-scoped committed-exploration latent. "
             "Each group draws z ~ Categorical(n_latent) from its own pooled "
             "encoder representation; Embed(z) seeds the group's AR chain AND "
             "conditions every member's decoder query, so members are "
             "correlated within a step and (via group-lifetime holding) "
             "across steps. Trained by PG on the team advantage + entropy + "
             "MI(z; joint action). Task-agnostic: the head's only input is the "
             "pooled representation. Mutually exclusive with group_token.",
    )
    parser.add_argument(
        "--n_latent", type=int, default=3,
        help="Number of latent joint-behaviour modes (K). K=2 reduces to a "
             "learned GO/NO-GO with no imposed meaning.",
    )
    parser.add_argument(
        "--latent_mi_coef", type=float, default=0.1,
        help="Weight on MI(z; joint action). 0 risks mode collapse, where "
             "every z produces identical behaviour and the latent is inert.",
    )
    parser.add_argument(
        "--latent_entropy_coef", type=float, default=0.01,
        help="Entropy bonus on the latent, to delay premature collapse.",
    )
    parser.add_argument(
        "--latent_cond_std", type=float, default=1.0,
        help="Init std of the per-member conditioning embedding. The decoder "
             "LayerNorms right after rep_enc+cond so the effect saturates near "
             "1.0; smaller values make the latent nearly inert at init.",
    )
    parser.add_argument(
        "--latent_bias_std", type=float, default=1.0,
        help="Init std of the per-mode action-logit bias. This is what gives "
             "the latent real influence: measured P(3 agents in a group pick "
             "the same action) is 0.0278 without it (= the independent "
             "baseline) and 0.136 with it. 0 disables the bias.",
    )
    parser.add_argument(
        "--latent_span_init", type=int, default=1,
        help="1 (default): init mode_bias so mode k prefers action k, which "
             "guarantees the K modes SPAN the action space. 0: random normal "
             "init, where K draws over A actions can miss the action that "
             "matters -- measured K=3/A=6/std=8 gave mode argmax [4,2,0] with "
             "P(capture)~0 for every mode, so it was never explored. Requires "
             "n_latent >= n_actions for full coverage.",
    )
    parser.add_argument(
        "--latent_bias_norm", type=int, default=1,
        help="1 (default): scale the mode bias by the action logits' own std, "
             "so it stays proportionally decisive as the policy's logits grow. "
             "An absolute bias cannot: measured, 2.0 was too weak to flip an "
             "already-suppressed action while 8.0 froze the policy and killed "
             "navigation. 0 = absolute bias of size --latent_bias_std.",
    )
    parser.add_argument(
        "--latent_bias_scale", type=float, default=1.5,
        help="Bias size as a multiple of (logit std + --latent_bias_floor). "
             "Dimensionless, so it should transfer across tasks and training "
             "stages rather than needing retuning.",
    )
    parser.add_argument(
        "--latent_bias_floor", type=float, default=1.0,
        help="Added to the logit std before scaling, so the bias still bites "
             "at init where logits are ~0.01 and a purely relative bias would "
             "vanish.",
    )
    parser.add_argument(
        "--latent_bias_scale_start", type=float, default=-1.0,
        help="If >= 0, anneal --latent_bias_scale linearly from this value to "
             "--latent_bias_scale_end over --latent_bias_anneal_frac of "
             "training. Rationale: navigation and correlation want opposite "
             "strengths -- measured, scale 1.5 gives P(all fire|READY)=0.25 "
             "but READY drops to 0.13/ep, while a weak bias gives READY "
             "7.40/ep and P(all fire)=0.00. Annealing lets one be learned "
             "before the other. -1 = no anneal (use --latent_bias_scale).",
    )
    parser.add_argument("--latent_bias_scale_end", type=float, default=-1.0)
    parser.add_argument("--latent_bias_anneal_frac", type=float, default=0.5,
                        help="Fraction of training over which to anneal.")
    parser.add_argument(
        "--latent_commit_eps", type=float, default=0.0,
        help="Group-correlated exploration. With this probability per group "
             "per step, every member commits to the action its shared latent "
             "mode favours. Generalizes the eps-override that broke the "
             "collapse: the action is learned rather than named, any group "
             "qualifies, and there is no k_req threshold. Independent "
             "eps-greedy explores a k-agent joint action at rate p^k; this "
             "explores it at eps*P(mode), independent of k. 0 = off.",
    )
    parser.add_argument(
        "--group_commit_eps_final", type=float, default=-1.0,
        help="Linear decay of --group_commit_eps to this value over "
             "--group_commit_decay_frac of training (exploration schedule, "
             "reward unchanged). < 0 = no decay (constant eps).",
    )
    parser.add_argument(
        "--group_commit_decay_frac", type=float, default=1.0,
        help="Fraction of training updates over which group_commit_eps "
             "decays to --group_commit_eps_final; constant afterwards.",
    )
    parser.add_argument(
        "--optimistic_adv", type=int, default=0,
        help="1 = OptiMAPPO-style optimistic advantages (Zhao et al. 2024): "
             "A <- max(opt_leaky_rate * A, A) BEFORE the batch standardization, "
             "so negative advantages (usually caused by teammates' exploratory "
             "actions) are shrunk while positive ones are kept. 0 = off.",
    )
    parser.add_argument(
        "--opt_leaky_rate", type=float, default=0.0,
        help="Slope for negative advantages under --optimistic_adv "
             "(OptiMAPPO's leaky_rate; 0 = drop negatives, 1 = no change).",
    )
    parser.add_argument(
        "--offpolicy_weight", type=int, default=0,
        help="1 = decoupled PPO (Hilton et al. 2021) for exploration "
             "overrides: each sample's advantage is multiplied by "
             "min(1, pi/mu), where mu is the probability the behaviour "
             "(policy + group commitment) actually produced the action and "
             "pi the policy's own probability; see --behav_w_mode for the "
             "exact form. Forced actions the policy would rarely take get "
             "small weight; on-policy samples keep weight ~1. Required "
             "whenever --group_commit_eps > 0. Task-agnostic. 0 = off.",
    )
    parser.add_argument(
        "--behav_w_mode", type=str, default="marginal",
        choices=["marginal", "arm"],
        help="Behaviour prob mu in the --offpolicy_weight. marginal: "
             "mu_i = (1-eps*q)*pi_i + eps*q/|A| for every arm (the original "
             "formula). arm: each arm's own factorization -- the marginal for "
             "independent actors (mappo, mat_comm_dec), and for the "
             "autoregressive mat_comm_soft the exact conditional "
             "mu(a_i | executed actions of the group members decoded before i) "
             "under the group-commit mixture. Uses only eps, q, |A| and the "
             "policy's probs.",
    )
    parser.add_argument(
        "--behav_w_floor", type=float, default=0.0,
        help="Lower bound on the --offpolicy_weight: w = max(floor, min(1, pi/mu)). "
             "Importance weighting can only shrink, so forced exploration "
             "successes the policy gives ~0 probability are otherwise erased "
             "(probe_explore_signal.py). A small floor lets them count while the "
             "cap at 1 still prevents blow-ups. Same for every arm; task-agnostic. "
             "0 = off (no floor).",
    )
    parser.add_argument(
        "--commit_member_prob", type=float, default=1.0,
        help="q for --group_commit_eps: when a group commits, each member "
             "independently joins with probability q and takes the group's "
             "shared random action; the others keep their policy action "
             "(re-decoded after the forced ones for the autoregressive arm). "
             "1 = every member forced to the same action (the original "
             "commit).",
    )
    parser.add_argument(
        "--critic_global_state", type=int, default=0,
        help="1 = mat_comm critic tokens also read each agent's full-view "
             "state chunk from share_obs (centralized critic). The actor "
             "stream never sees it. train_joint_nav.py turns it on whenever "
             "obs_mode is not 'shared'.",
    )
    parser.add_argument(
        "--decoder_self_only", type=int, default=0,
        help="1 = observation-sharing DE control: comm-masked encoder, but the "
             "decoder sees only the agent's own START (no action "
             "conditioning). Set automatically by --algorithm_name "
             "mat_comm_dec in train_joint_nav.py.",
    )
    parser.add_argument(
        "--commit_groups", type=str, default="capture",
        choices=["capture", "comm"],
        help="Groups for --group_commit_eps. capture: agents in capture range "
             "of the same target (needs local/shared obs layout). comm: "
             "connected components of the agent-radius graph, read from the "
             "radius obs visibility flags (task-agnostic, same for all arms).",
    )
    parser.add_argument(
        "--group_commit_eps", type=float, default=0.0,
        help="Arm-agnostic correlated exploration (the fair control). With "
             "this probability per group per step, every member takes the "
             "same uniformly RANDOM action. Uses no latent and ignores the "
             "policy, so it applies identically to mappo / plain CT-SDE / "
             "latent CT-SDE and isolates execution. 0 = off.",
    )
    parser.add_argument(
        "--latent_hold", type=int, default=1,
        help="1=hold z for a group's lifetime (resample only when membership "
             "changes) — this is the 'committed' part. 0=resample every step.",
    )
    parser.add_argument(
        "--group_token", type=int, default=0,
        help="maco_pursuit mat_comm only. 1=per-clique GO/NO-GO token "
             "replaces the obs-skip MLP. 0=off.",
    )
    parser.add_argument(
        "--group_token_entropy_coef", type=float, default=None,
        help="Entropy coefficient on the group token. Required when "
             "--group_token 1 (no silent default).",
    )
    parser.add_argument(
        "--z_loss_ready_balance", type=int, default=0,
        help="1=inverse-frequency weight nonzero z outcomes by ready "
             "bucket (ready < catch_min vs >= catch_min). 0=off. "
             "Does not drop pair rows.",
    )
    parser.add_argument(
        "--fixed_groups", type=str, default=None,
        help="mat_comm only. Fixed partition sizes, e.g. 3,3,4. "
             "Replaces same-prey / radius masks. Empty/None = env default. "
             "Refused with --group_token 1.",
    )
    parser.add_argument(
        "--fixed_groups_assign", type=str, default="id",
        choices=["id", "nearest"],
        help="How --fixed_groups is filled. id=consecutive agent ids "
             "(default). nearest=closest agents at episode reset, then "
             "frozen until the next reset. Ignored unless --fixed_groups.",
    )
    parser.add_argument(
        "--use_obs_skip", type=int, default=1,
        help="mat_comm decoder MLP(obs) skip to action logits. 1=on "
             "(default, current skip recipe). 0=AR logits only. "
             "Refused with --group_token 1.",
    )
    parser.add_argument(
        "--commitment", type=int, default=0,
        help="mat_comm only. 1=discrete commitment head z over visible "
             "target slots + NULL, sampled before the action. 0=off. "
             "Refused on MAPPO/MAT.",
    )
    parser.add_argument(
        "--n_commit_targets", type=int, default=0,
        help="Target-slot count for --commitment 1 (foods / prey / enemies). "
             "Train scripts set this from the env; do not guess.",
    )
    parser.add_argument(
        "--decoder_attn_gain", type=float, default=0.01,
        help="Orthogonal init gain for mat_comm decoder. Stock MAT is 0.01 "
             "(attention projections only). Values other than 0.01 re-init "
             "every decoder Linear (attn, action encoder, MLP, head) with "
             "that gain; encoder is unchanged. Attn-only scaling cannot "
             "reach Δlogit>0.1 on this architecture.",
    )

    # add for online multi-task
    parser.add_argument("--train_maps", type=str, nargs='+', default=None)
    parser.add_argument("--eval_maps", type=str, nargs='+', default=None)

    return parser
