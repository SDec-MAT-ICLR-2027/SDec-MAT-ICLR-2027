#!/usr/bin/env python
"""Train MAPPO or mat_comm_soft (CT-SDE) on joint-capture navigation.

Arms
––––
  mappo          independent actors. Pair with `--obs_mode shared` for the
                 observation-sharing decentralized baseline: it sees every
                 agent and every target, it just cannot correlate actions.
  mat_comm_soft  CT-SDE, plain form. Group-masked encoder/decoder over target
                 cliques, soft_group_bias critic, ordinary learned START token
                 per group. No group-token GO head, no obs-skip, no head-agent
                 token prediction. This script forces those off.

`mat_comm_soft` is the arm name; it is normalized to algorithm_name
`mat_comm` (the network dispatch key) with encoder_critic_attn forced to
soft_group_bias, so the run is self-documenting and cannot be launched
"soft" without the soft critic.

There is no reward curriculum here on purpose: the dense assembly shaping
gives the capture action a standing gradient, so the failure tax can be
constant from step 0.
"""

import os
import socket
import sys
from pathlib import Path

import numpy as np
import setproctitle
import torch
import wandb

sys.path.append(str(Path(os.path.abspath(__file__)).parents[3]))

from mat.config import get_config
from mat.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
from mat.envs.joint_nav import JointNavMATEnv
from mat.runner.shared.joint_nav_runner import JointNavRunner as Runner

MAPPO_ALGOS = ("mappo", "mat_dec")
# mat_comm_dec: observation-sharing DE control (comm-masked encoder,
# self-only decoder) -- the same network as mat_comm_soft minus action
# conditioning.
COMM_ARMS = ("mat_comm_soft", "mat_comm_dec")
ALLOWED_ARMS = MAPPO_ALGOS + COMM_ARMS


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "joint_nav":
                raise NotImplementedError("unsupported env: %s" % all_args.env_name)
            env = JointNavMATEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv(
        [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
    )


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = JointNavMATEnv(all_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv(
        [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
    )


def parse_args(args, parser):
    parser.add_argument("--map_name", type=str, default="joint_nav_6a3t")
    parser.add_argument("--n_agents", type=int, default=6)
    parser.add_argument("--num_targets", type=int, default=3)
    parser.add_argument(
        "--k_req",
        type=int,
        default=3,
        help="Simultaneous capturers needed to fell a target. THE "
        "coordination knob: sweep 1..4 to trace the DE/SDE boundary.",
    )
    parser.add_argument("--capture_radius", type=float, default=0.3)
    parser.add_argument(
        "--group_radius",
        type=float,
        default=0.6,
        help="Radius for FORMING joint-decision groups (>= capture_radius). "
        "Coupling starts on the approach; capture itself always uses "
        "capture_radius. Too tight and the cliques are all singletons.",
    )
    parser.add_argument("--capture_rew", type=float, default=10.0)
    parser.add_argument(
        "--fail_rew",
        type=float,
        default=-1.0,
        help="Team reward when 1..k_req-1 agents capture. Constant from step "
        "0; there is no curriculum on this task.",
    )
    parser.add_argument(
        "--assembly_rew",
        type=float,
        default=0.05,
        help="Dense per-step shaping for co-locating at an active target. "
        "Individually climbable, which is what removes the need for a "
        "failure-penalty curriculum.",
    )
    parser.add_argument("--dist_rew_scale", type=float, default=1.0)
    parser.add_argument(
        "--potential_shaping",
        type=int,
        default=1,
        help="1 (default): dense shaping as a potential difference "
        "gamma*Phi(s')-Phi(s), which cannot be farmed and provably preserves "
        "the optimal policy. 0: legacy per-step rate, which makes capturing "
        "the target you stand on WORSE than camping on it. 0 is kept only to "
        "reproduce that pathology; do not use it for real runs.",
    )
    parser.add_argument(
        "--shaping_gamma",
        type=float,
        default=0.99,
        help="Discount used in the shaping potential. Should match --gamma.",
    )
    parser.add_argument("--collision_rew", type=float, default=0.0)
    parser.add_argument("--world_size", type=float, default=2.0)
    parser.add_argument("--num_obstacles", type=int, default=0)
    parser.add_argument("--max_speed", type=float, default=2.0)
    parser.add_argument(
        "--obs_mode",
        type=str,
        default="shared",
        choices=["local", "shared", "radius"],
        help="shared (default): every agent sees every other agent and every "
        "target. This is the observation-sharing setting, so any gap between "
        "arms is about how they act, not what they know.",
    )
    parser.add_argument(
        "--sense_radius",
        type=float,
        default=1.0,
        help="obs_mode=radius: agents perceive agents/targets only within "
        "this distance. With comm_mode=agent_radius it is also the comm "
        "radius (one R for sensing and communication). Must be >= "
        "2*capture_radius so a ready group is always connected.",
    )
    parser.add_argument(
        "--obs_radius",
        type=float,
        default=None,
        help="CENSUS ABLATION. How far an agent sees OTHER AGENTS. Defaults to "
        "sense_radius (previous behaviour). Targets are still gated by "
        "sense_radius, and the comm graph is unchanged, so lowering this "
        "removes the census without touching the tally or navigation.",
    )
    parser.add_argument(
        "--obs_abs_pos",
        type=int,
        default=1,
        help="CENSUS ABLATION. 1 (default) keeps the agent's absolute own "
        "position in its observation. 0 removes it, so every observation is "
        "in the agent's own frame and pooling a neighbour's observation "
        "cannot restore the census that --obs_radius removed.",
    )
    parser.add_argument(
        "--shuffle_targets",
        type=int,
        default=0,
        help="1 = randomly permute placed target positions over target ids at "
        "each reset (removes the id<->placement bias of sequential rejection "
        "sampling).",
    )
    parser.add_argument(
        "--target_order",
        type=str,
        default="id",
        choices=["id", "distance"],
        help="radius obs target slots: by target id, or nearest-first "
        "(live visible, then visible captured, then hidden).",
    )
    parser.add_argument(
        "--k_max",
        type=int,
        default=None,
        help="Exact-k capture (E4): more than k_max in-range capturers on a "
        "target is an overkill (no capture, fail_rew). Default None = the "
        "original >= k_req rule.",
    )
    parser.add_argument(
        "--step_cost",
        type=float,
        default=0.0,
        help="Per-step TEAM cost charged until the episode ends. Without it "
        "passivity is free (shaping telescopes, off-target CAPTURE is a "
        "no-op), so 'never fire' is a free absorbing strategy. Aerial, Pickup "
        "Consensus and Team Partition all charge for time. Default 0.0 = off "
        "(reproduces every run made before this flag).",
    )
    parser.add_argument(
        "--min_target_sep",
        type=float,
        default=None,
        help="Minimum distance between targets (default 2*group_radius). "
        "For the radius env, sep > sense_radius + 2*capture_radius "
        "guarantees two groups on different targets never see each other.",
    )
    parser.add_argument(
        "--comm_mode",
        type=str,
        default="target_clique",
        choices=["target_clique", "agent_radius"],
        help="target_clique: grouped iff within group_radius of the same live "
        "target (uses task structure). agent_radius: grouped iff within "
        "sense_radius of each other (pure distance, task-agnostic).",
    )
    parser.add_argument(
        "--obs_ready_count",
        type=int,
        default=0,
        help="1=append the in-range count at the agent's nearest active "
        "target to its local obs (information control). 0=off.",
    )
    parser.add_argument(
        "--explore_group_capture_eps",
        type=float,
        default=0.0,
        help="DIAGNOSTIC. Probability per in-range group per step of forcing "
        "all its members to CAPTURE together during ROLLOUT only. Crude "
        "correlated exploration: independent exploration finds a k-agent "
        "simultaneous action with probability ~p^k, so the reward is never "
        "observed and the action is extinguished first. 0 = off (normal run).",
    )
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="jointnav")
    parser.add_argument("--wandb_on", action="store_true", default=False)
    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
        help="Load --model_dir and run greedy eval, then exit. Does not train.",
    )
    parser.set_defaults(episode_length=50)
    return parser.parse_known_args(args)[0]


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    all_args.env_name = "joint_nav"

    arm = str(all_args.algorithm_name)
    if arm not in ALLOWED_ARMS:
        raise ValueError(
            "train_joint_nav.py allows %s, got %r" % (ALLOWED_ARMS, arm)
        )
    all_args.arm_name = arm
    if all_args.obs_mode == "radius":
        # one grouping in the radius env: the R-graph CT-SDE communicates
        # over; the exploration commitment uses its connected components
        if getattr(all_args, "commit_groups", "capture") != "comm":
            print("[train_joint_nav] obs_mode=radius -> commit_groups=comm "
                  "(was %r)" % getattr(all_args, "commit_groups", None))
        all_args.commit_groups = "comm"

    if arm in MAPPO_ALGOS:
        all_args.dec_actor = True
        all_args.share_actor = True
        # MAPPO: the actor is an obs-only MLP; the critic is the encoder. When
        # obs is partial, the critic must read the global state (share_obs,
        # tiled per agent) -- standard CTDE. In shared mode obs is already the
        # full view, so earlier runs are unchanged.
        all_args.encode_state = all_args.obs_mode != "shared"
        if getattr(all_args, "fixed_groups", None):
            raise ValueError("fixed_groups is a mat_comm option")
        if int(getattr(all_args, "group_token", 0)) != 0:
            raise ValueError("group_token is a mat_comm option")
    else:
        # normalize the arm name to the network dispatch key and pin the
        # plain CT-SDE configuration
        all_args.algorithm_name = "mat_comm"
        all_args.decoder_self_only = 1 if arm == "mat_comm_dec" else 0
        all_args.comm_mask_kind = "pairwise"
        all_args.encoder_critic_attn = "soft_group_bias"
        # MAT computes the value from the obs encoder. When obs is not the
        # full view, the critic must also read the global state (CTDE); in
        # shared mode obs already is the full view, so nothing changes there.
        all_args.critic_global_state = (
            0 if all_args.obs_mode == "shared" else 1
        )
        if all_args.dec_actor:
            raise ValueError("mat_comm_soft is incompatible with --dec_actor")
        if getattr(all_args, "comm_radius", None) is not None:
            raise ValueError(
                "do not pass comm_radius; groups are target cliques"
            )
        if getattr(all_args, "fixed_groups", None):
            raise ValueError(
                "fixed_groups is refused: this arm groups by target clique"
            )
        # the plain form: none of the pursuit-specific scaffolding
        for flag, want in (("group_token", 0), ("use_obs_skip", 0)):
            got = int(getattr(all_args, flag, want))
            if got != want:
                raise ValueError(
                    "mat_comm_soft is the plain CT-SDE arm; --%s must be %d, "
                    "got %d" % (flag, want, got)
                )
            setattr(all_args, flag, want)
        if int(getattr(all_args, "commitment", 0)) != 0:
            raise ValueError("commitment is not wired for joint_nav")
        if int(getattr(all_args, "z_loss_ready_balance", 0)) != 0:
            raise ValueError("z_loss_ready_balance requires group_token=1")
        print(
            "[train_joint_nav] %s: comm_mask_kind=pairwise comm_mode=%s "
            "(sense_radius=%s, group_radius=%s) encoder_critic_attn=soft_group_bias "
            "group_token=0 use_obs_skip=0 (plain CT-SDE, no head-agent token)"
            % (arm, getattr(all_args, "comm_mode", "target_clique"),
               getattr(all_args, "sense_radius", None),
               getattr(all_args, "group_radius", None)),
            flush=True,
        )

    # ---- env sanity ----
    if int(all_args.n_agents) < 1:
        raise ValueError("n_agents must be >= 1, got %r" % all_args.n_agents)
    if int(all_args.num_targets) < 1:
        raise ValueError("num_targets must be >= 1, got %r" % all_args.num_targets)
    k = int(all_args.k_req)
    if k < 1:
        raise ValueError("k_req must be >= 1, got %r" % k)
    if k > int(all_args.n_agents):
        raise ValueError(
            "k_req (%d) > n_agents (%d): no target could ever be captured"
            % (k, all_args.n_agents)
        )
    if float(all_args.capture_radius) <= 0:
        raise ValueError(
            "capture_radius must be > 0, got %r" % all_args.capture_radius
        )
    if float(all_args.fail_rew) > 0:
        raise ValueError(
            "fail_rew must be <= 0 (it is the undershoot tax), got %r"
            % all_args.fail_rew
        )
    raw_ready = getattr(all_args, "obs_ready_count", None)
    if raw_ready is None or int(raw_ready) not in (0, 1):
        raise ValueError("obs_ready_count must be 0 or 1, got %r" % raw_ready)
    all_args.obs_ready_count = bool(int(raw_ready))
    if int(all_args.episode_length) < 1:
        raise ValueError(
            "episode_length must be >= 1, got %r" % all_args.episode_length
        )
    if all_args.encode_state and all_args.algorithm_name not in MAPPO_ALGOS:
        # mat_comm refuses encode_state itself (it would leak the tiled global
        # state to every agent). For MAPPO, ma_transformer keeps the real state
        # when encode_state is on; it only reaches the critic, because the
        # dec_actor MLP reads obs alone (test_mappo_global_critic.py).
        raise ValueError(
            "--encode_state is only allowed for the MAPPO arm here"
        )

    if all_args.eval_only:
        if not all_args.model_dir:
            raise ValueError("--eval_only requires --model_dir")
        if not os.path.isfile(all_args.model_dir):
            raise ValueError("--model_dir is not a file: %s" % all_args.model_dir)
        all_args.use_eval = True
        if all_args.wandb_on:
            raise ValueError("--eval_only refuses --wandb_on")

    if all_args.wandb_on:
        all_args.use_wandb = True
        if all_args.user_name == "xxx":
            raise ValueError(
                "--wandb_on requires --user_name (default xxx is refused)"
            )

    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    print("=" * 72)
    print("RESOLVED CONFIG (post-merge, this is what the run actually uses)")
    for key in sorted(vars(all_args)):
        print("  %s = %s" % (key, getattr(all_args, key)))
    print("=" * 72)
    sys.stdout.flush()

    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.map_name
        / all_args.arm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        # Discriminative run name: arm + the knobs that actually differ between
        # cells, so runs are distinguishable in the wandb list without opening
        # the config. `experiment_name` must NOT repeat the arm name.
        # experiment_name is included because sweeps often vary only the
        # method config (bias scale, anneal, hold, K), which the env-level
        # fields below do not capture -- without it every variant in a sweep
        # collides under one wandb name.
        run_name = "%s_%s_k%d_cr%g_fr%g_%s_ar%g_seed%s" % (
            all_args.arm_name,
            all_args.experiment_name,
            int(all_args.k_req),
            float(all_args.capture_rew),
            float(all_args.fail_rew),
            all_args.obs_mode,
            float(all_args.assembly_rew),
            all_args.seed,
        )
        tags = [
            "joint_nav",
            "arm=%s" % all_args.arm_name,
            "k_req=%d" % int(all_args.k_req),
            "capture_rew=%g" % float(all_args.capture_rew),
            "fail_rew=%g" % float(all_args.fail_rew),
            "assembly_rew=%g" % float(all_args.assembly_rew),
            "obs_mode=%s" % all_args.obs_mode,
            "n_agents=%d" % int(all_args.n_agents),
            "num_targets=%d" % int(all_args.num_targets),
            "seed=%s" % all_args.seed,
            "no_curriculum",
        ]
        if int(getattr(all_args, "group_latent", 0)) == 1:
            tags += [
                "group_latent",
                "n_latent=%d" % int(all_args.n_latent),
                "bias_scale=%g" % float(all_args.latent_bias_scale),
                "bias_norm=%d" % int(all_args.latent_bias_norm),
                "latent_hold=%d" % int(all_args.latent_hold),
                "mi=%g" % float(all_args.latent_mi_coef),
                "lat_ent=%g" % float(all_args.latent_entropy_coef),
                "group_radius=%g" % float(all_args.group_radius),
            ]
            if float(getattr(all_args, "latent_bias_scale_start", -1)) >= 0:
                tags.append("anneal=%g->%g@%g" % (
                    float(all_args.latent_bias_scale_start),
                    float(all_args.latent_bias_scale_end),
                    float(all_args.latent_bias_anneal_frac)))
        else:
            tags.append("no_latent")
        run = wandb.init(
            config=all_args,
            project=all_args.wandb_project,
            entity=all_args.user_name,
            notes="%s | %s" % (socket.gethostname(), all_args.experiment_name),
            name=run_name,
            group=all_args.map_name,
            tags=tags,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )
        print("[train_joint_nav] wandb run name=%s tags=%s" % (run_name, tags),
              flush=True)
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        curr_run = (
            "run1" if len(exst_run_nums) == 0 else "run%i" % (max(exst_run_nums) + 1)
        )
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        "%s-joint_nav-%s@%s"
        % (all_args.arm_name, all_args.experiment_name, all_args.user_name)
    )

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    all_args.run_dir = run_dir
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.n_agents,
        "device": device,
        "run_dir": run_dir,
    }

    runner = Runner(config)
    if all_args.eval_only:
        print("[train_joint_nav] eval_only model_dir=%s" % all_args.model_dir, flush=True)
        runner.eval(0)
    else:
        runner.run()

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
