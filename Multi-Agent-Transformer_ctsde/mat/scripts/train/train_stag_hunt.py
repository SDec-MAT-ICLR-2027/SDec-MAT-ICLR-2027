#!/usr/bin/env python
"""Train MAT / mat_comm on Stag Hunt (explicit catch action).

Env knobs match src/config/envs/stag_hunt.yaml except CLI overrides.
Fair vs dicg_pp mat_comm: same MAT PPO defaults + --entropy_coef 0.1.
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
from mat.envs.stag_hunt import StagHuntMATEnv
from mat.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
from mat.runner.shared.stag_hunt_runner import StagHuntRunner as Runner

COMM_ALGOS = ("mat_comm", "mat_comm_gru")


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = StagHuntMATEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = StagHuntMATEnv(all_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--map_name", type=str, default="stag_hunt")
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--n_agents", type=int, default=8)
    parser.add_argument("--n_stags", type=int, default=8)
    parser.add_argument("--n_hare", type=int, default=0)
    parser.add_argument("--episode_limit", type=int, default=200)
    parser.add_argument("--comm_radius", type=int, default=3)
    parser.add_argument("--miscapture_punishment", type=float, default=-2.0)
    parser.add_argument("--capture_reward", type=float, default=10.0)
    parser.add_argument("--step_cost", type=float, default=0.0)
    parser.add_argument("--dicg_style_obs", type=int, default=0)
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="jointnav")
    parser.add_argument("--wandb_on", action="store_true", default=False)
    return parser.parse_known_args(args)[0]


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args.env_name = "stag_hunt"

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True

    if all_args.algorithm_name in COMM_ALGOS:
        all_args.comm_mask_kind = "pairwise"
        print("[train_stag_hunt] comm_mask_kind forced to pairwise for %s"
              % all_args.algorithm_name, flush=True)
    if all_args.encode_state:
        raise ValueError("--encode_state is unsafe for mat_comm share_obs plumbing")

    if all_args.wandb_on:
        all_args.use_wandb = True

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
        print(f"  {k} = {getattr(all_args, k)}")
    print("=" * 72)
    sys.stdout.flush()

    run_dir = Path(
        os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
        + "/results") / all_args.env_name / all_args.map_name / all_args.algorithm_name \
        / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.wandb_project,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=f"{all_args.algorithm_name}_{all_args.experiment_name}"
                              f"_seed{all_args.seed}",
                         group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [int(str(folder.name).split("run")[1]) for folder in run_dir.iterdir()
                             if str(folder.name).startswith("run")]
            curr_run = "run1" if len(exst_run_nums) == 0 else "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        f"{all_args.algorithm_name}-stag_hunt-{all_args.experiment_name}@{all_args.user_name}")

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
