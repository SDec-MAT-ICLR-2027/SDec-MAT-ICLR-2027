#!/usr/bin/env python
"""Train MAPPO (mat_dec) or soft-critic mat_comm on CASEC MACO Pursuit.

Paper yaml defaults: 10 agents, 5 preys, 10x10, horizon 50, sight 2,
catch +1 / solo -1.

MAPPO: no comm tail. Independent actors (dec_actor=True).

mat_comm: groups from the Chebyshev agent-ID crop already in local obs
(sight_range). No comm_radius. This script forces --comm_mask_kind pairwise
and --encoder_critic_attn soft_group_bias. Refuses --dec_actor.
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
from mat.envs.maco_pursuit import MACOPursuitMATEnv
from mat.runner.shared.maco_pursuit_runner import MACOPursuitRunner as Runner

MAPPO_ALGOS = ("mappo", "mat_dec")
COMM_ALGOS = ("mat_comm",)
ALLOWED_ALGOS = MAPPO_ALGOS + COMM_ALGOS


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "maco_pursuit":
                raise NotImplementedError("unsupported env: %s" % all_args.env_name)
            env = MACOPursuitMATEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = MACOPursuitMATEnv(all_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv(
        [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
    )


def parse_args(args, parser):
    parser.add_argument("--map_name", type=str, default="pursuit_10x10")
    parser.add_argument("--n_agents", type=int, default=10)
    parser.add_argument("--n_preys", type=int, default=5)
    parser.add_argument("--episode_limit", type=int, default=50)
    parser.add_argument("--map_size", type=int, default=10)
    parser.add_argument("--catch_reward", type=float, default=1.0)
    parser.add_argument(
        "--catch_min",
        type=int,
        default=2,
        help="Simultaneous catchers needed to kill a prey (CASEC paper=2). "
        "1..catch_min-1 is a fail. Must be 2..4.",
    )
    parser.add_argument("--catch_fail_reward", type=float, default=-1.0)
    parser.add_argument(
        "--fail_penalty_switch_steps",
        type=int,
        default=-1,
        help="-1=off. Else global env-step when catch_fail_reward becomes "
        "--catch_fail_reward_after. Must divide n_rollout_threads.",
    )
    parser.add_argument(
        "--catch_fail_reward_after",
        type=float,
        default=None,
        help="Required when fail_penalty_switch_steps >= 0. Solo-fail reward "
        "after the switch (paper tax is -1).",
    )
    parser.add_argument("--sight_range", type=int, default=2)
    parser.add_argument(
        "--obs_ready_count",
        type=int,
        default=0,
        help="1=append same-prey Manhattan-1 clique size to local obs "
        "(MAPPO/mat_dec fair-count control). 0=off. Refused with group_token.",
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

    all_args.env_name = "maco_pursuit"

    if all_args.algorithm_name not in ALLOWED_ALGOS:
        raise ValueError(
            "train_maco_pursuit.py allows mappo/mat_dec/mat_comm, got %r"
            % all_args.algorithm_name
        )
    if all_args.algorithm_name in MAPPO_ALGOS:
        all_args.dec_actor = True
        all_args.share_actor = True
        if getattr(all_args, "fixed_groups", None):
            raise ValueError("fixed_groups is mat_comm only")
    else:
        all_args.comm_mask_kind = "pairwise"
        all_args.encoder_critic_attn = "soft_group_bias"
        from mat.algorithms.utils.mat_comm_groups import parse_fixed_group_sizes

        fg_sizes = parse_fixed_group_sizes(
            getattr(all_args, "fixed_groups", None), int(all_args.n_agents)
        )
        raw_skip = getattr(all_args, "use_obs_skip", 1)
        if raw_skip is None or int(raw_skip) not in (0, 1):
            raise ValueError("use_obs_skip must be 0 or 1, got %r" % raw_skip)
        print(
            "[train_maco_pursuit] mat_comm: comm_mask_kind=pairwise "
            "groups=%s encoder_critic_attn=soft_group_bias use_obs_skip=%s"
            % (
                (
                    "fixed_"
                    + "_".join(str(s) for s in fg_sizes)
                    + ":"
                    + str(getattr(all_args, "fixed_groups_assign", "id"))
                    if fg_sizes is not None
                    else "same-prey Manhattan-1 catcher cliques"
                ),
                int(raw_skip),
            ),
            flush=True,
        )
        if all_args.dec_actor:
            raise ValueError("mat_comm is incompatible with --dec_actor")
        if int(getattr(all_args, "commitment", 0)) != 0:
            raise ValueError("commitment is not wired for maco_pursuit")
        raw_gt = getattr(all_args, "group_token", 0)
        if raw_gt is None or int(raw_gt) not in (0, 1):
            raise ValueError("group_token must be 0 or 1, got %r" % raw_gt)
        zb = int(getattr(all_args, "z_loss_ready_balance", 0))
        if zb not in (0, 1):
            raise ValueError(
                "z_loss_ready_balance must be 0 or 1, got %r" % zb
            )
        if zb == 1 and int(raw_gt) != 1:
            raise ValueError("z_loss_ready_balance=1 requires group_token=1")
        if fg_sizes is None and str(getattr(all_args, "fixed_groups_assign", "id")) not in (
            "id",
        ):
            raise ValueError(
                "fixed_groups_assign=%r requires --fixed_groups"
                % getattr(all_args, "fixed_groups_assign", None)
            )
        if fg_sizes is not None and int(raw_gt) == 1:
            raise ValueError("fixed_groups and group_token are mutually exclusive")
        if int(raw_gt) == 1 and int(raw_skip) == 0:
            raise ValueError("group_token=1 requires --use_obs_skip 1")
        if int(raw_gt) == 1:
            gte = getattr(all_args, "group_token_entropy_coef", None)
            if gte is None:
                raise ValueError(
                    "group_token=1 requires --group_token_entropy_coef"
                )
            print(
                "[train_maco_pursuit] group_token=1 learned GO/NO-GO "
                "(no ready mask) token_entropy_coef=%s token_credit=clique_catch "
                "(obs skip ON, loner START=blank) z_loss_ready_balance=%s"
                % (gte, zb),
                flush=True,
            )
        if getattr(all_args, "comm_radius", None) is not None:
            raise ValueError(
                "do not pass comm_radius; use same-prey cliques or --fixed_groups"
            )

    if int(all_args.n_agents) < 1:
        raise ValueError("n_agents must be >= 1, got %r" % all_args.n_agents)
    if int(all_args.n_preys) < 1:
        raise ValueError("n_preys must be >= 1, got %r" % all_args.n_preys)
    if int(all_args.map_size) < 1:
        raise ValueError("map_size must be >= 1, got %r" % all_args.map_size)
    if int(all_args.sight_range) < 0:
        raise ValueError("sight_range must be >= 0, got %r" % all_args.sight_range)
    cm = int(all_args.catch_min)
    if cm < 2 or cm > 4:
        raise ValueError("catch_min must be in 2..4, got %r" % all_args.catch_min)
    raw_ready = getattr(all_args, "obs_ready_count", None)
    if raw_ready is None or int(raw_ready) not in (0, 1):
        raise ValueError("obs_ready_count must be 0 or 1, got %r" % raw_ready)
    all_args.obs_ready_count = int(raw_ready)
    if all_args.obs_ready_count == 1 and int(getattr(all_args, "group_token", 0)) == 1:
        raise ValueError("obs_ready_count=1 is refused with group_token (ready already in pack)")
    if all_args.obs_ready_count == 1:
        print(
            "[train_maco_pursuit] obs_ready_count=1 (same-prey Manhattan-1 "
            "clique size appended to local obs)",
            flush=True,
        )
    if int(all_args.episode_limit) < 1:
        raise ValueError("episode_limit must be >= 1, got %r" % all_args.episode_limit)
    if int(all_args.episode_length) != int(all_args.episode_limit):
        raise ValueError(
            "episode_length (%r) must equal episode_limit (%r)"
            % (all_args.episode_length, all_args.episode_limit)
        )
    switch = int(all_args.fail_penalty_switch_steps)
    if switch < -1:
        raise ValueError(
            "fail_penalty_switch_steps must be -1 or >= 0, got %r" % switch
        )
    if switch == -1:
        if all_args.catch_fail_reward_after is not None:
            raise ValueError(
                "catch_fail_reward_after=%r requires fail_penalty_switch_steps >= 0"
                % all_args.catch_fail_reward_after
            )
    else:
        if all_args.catch_fail_reward_after is None:
            raise ValueError(
                "fail_penalty_switch_steps=%r requires --catch_fail_reward_after"
                % switch
            )
        after = float(all_args.catch_fail_reward_after)
        start = float(all_args.catch_fail_reward)
        if after == start:
            raise ValueError(
                "catch_fail_reward_after (%r) equals catch_fail_reward; "
                "curriculum would be a no-op" % after
            )
        nthr = int(all_args.n_rollout_threads)
        nsteps = int(all_args.num_env_steps)
        if switch % nthr != 0:
            raise ValueError(
                "fail_penalty_switch_steps (%r) must be divisible by "
                "n_rollout_threads (%r)" % (switch, nthr)
            )
        if switch >= nsteps:
            raise ValueError(
                "fail_penalty_switch_steps (%r) must be < num_env_steps (%r)"
                % (switch, nsteps)
            )
        print(
            "[train_maco_pursuit] fail-penalty curriculum: %s until %d steps, "
            "then %s" % (start, switch, after),
            flush=True,
        )
    if all_args.encode_state:
        raise ValueError(
            "--encode_state is unsafe: vanilla MAT zeros the state, so the "
            "encoder would see only zeros."
        )

    if all_args.eval_only:
        if not all_args.model_dir:
            raise ValueError("--eval_only requires --model_dir (path to transformer_*.pt)")
        if not os.path.isfile(all_args.model_dir):
            raise ValueError("--model_dir is not a file: %s" % all_args.model_dir)
        all_args.use_eval = True
        if all_args.wandb_on:
            raise ValueError("--eval_only refuses --wandb_on")

    if all_args.wandb_on:
        all_args.use_wandb = True
        if all_args.user_name == "xxx":
            raise ValueError("--wandb_on requires --user_name (default xxx is refused)")

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
    for k in sorted(vars(all_args)):
        print("  %s = %s" % (k, getattr(all_args, k)))
    print("=" * 72)
    sys.stdout.flush()

    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.map_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.wandb_project,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name="%s_%s_seed%s"
            % (all_args.algorithm_name, all_args.experiment_name, all_args.seed),
            group=all_args.map_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
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
        "%s-maco_pursuit-%s@%s"
        % (all_args.algorithm_name, all_args.experiment_name, all_args.user_name)
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
        print("[train_maco_pursuit] eval_only model_dir=%s" % all_args.model_dir, flush=True)
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
