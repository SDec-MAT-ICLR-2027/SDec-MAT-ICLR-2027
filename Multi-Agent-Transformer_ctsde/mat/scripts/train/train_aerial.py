#!/usr/bin/env python
"""Train script for the Aerial Ordering benchmark.

Derived from `train_smac.py`. Two differences worth knowing:

  * SMAC reads its team size from a map table; here it is `1 + n_workers`, so
    the env constructor arguments have to be parsed before `num_agents` is
    known. `num_agents_from` does that without building an env.
  * The env-arg defaults are the paper configuration (mac/presets.py), so a
    bare run reproduces the paper's task. Override any of them on the command line.

Run directory layout: mat/scripts/results/<env>/<scenario>/<algo>/<exp>/runN
"""

import os
import socket
import sys
from pathlib import Path

import numpy as np
import setproctitle
import torch
import wandb

# <repo>/Multi-Agent-Transformer/mat/scripts/train/this_file.py -> the dir that
# holds the `mat` package, so `import mat.…` works from anywhere.
sys.path.append(str(Path(os.path.abspath(__file__)).parents[3]))

from mat.config import get_config
from mat.envs.aerial.aerial_ordering_mat_wrapper import (
    AerialOrderingMATWrapper,
    ENV_ARG_DEFAULTS,
    num_agents_from,
)
from mat.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from mat.runner.shared.aerial_runner import AerialRunner as Runner


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "Aerial":
                raise NotImplementedError(
                    "Can not support the " + all_args.env_name + " environment.")
            env = AerialOrderingMATWrapper(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "Aerial":
                raise NotImplementedError(
                    "Can not support the " + all_args.env_name + " environment.")
            env = AerialOrderingMATWrapper(all_args)
            # Disjoint from the training seeds so eval episodes are unseen.
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='aerial_ordering',
                        help="Name used for the results directory")
    parser.add_argument('--run_dir', type=str, default='')

    # AerialOrderingEnv constructor arguments. Defaults are the paper
    # configuration (mac/presets.py).
    parser.add_argument('--obs_mode', type=str, default=ENV_ARG_DEFAULTS["obs_mode"],
                        choices=["simple", "extended"],
                        help="simple: 5D obs / 3 actions. extended: 10D obs / 6 actions")
    parser.add_argument('--grid_size', type=int, default=ENV_ARG_DEFAULTS["grid_size"])
    parser.add_argument('--n_workers', type=int, default=ENV_ARG_DEFAULTS["n_workers"],
                        help="Team size; n_agents is this plus one sensor")
    parser.add_argument('--n_targets', type=int, default=ENV_ARG_DEFAULTS["n_targets"])
    parser.add_argument('--k_min', type=int, default=ENV_ARG_DEFAULTS["k_min"],
                        help="Inclusive lower bound on the hidden requirement K")
    parser.add_argument('--k_max', type=int, default=ENV_ARG_DEFAULTS["k_max"],
                        help="Inclusive upper bound on the hidden requirement K")
    parser.add_argument('--comm_radius', type=int, default=ENV_ARG_DEFAULTS["comm_radius"],
                        help="Manhattan radius around the sensor for the comm/readiness gate")
    parser.add_argument('--sense_radius', type=int, default=ENV_ARG_DEFAULTS["sense_radius"])
    parser.add_argument('--wind_prob', type=float, default=ENV_ARG_DEFAULTS["wind_prob"])
    parser.add_argument('--time_limit', type=int, default=ENV_ARG_DEFAULTS["time_limit"],
                        help="Episode limit; keep --episode_length equal to this")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True

    if all_args.encode_state:
        # Unsafe for every algorithm here, for two different reasons:
        #   mat/mat_dec/...: ma_transformer.py replaces the state with
        #     np.zeros(..., 37), so the encoder would see only zeros.
        #   mat_comm: share_obs is one global state tiled per agent, so
        #     encoding it hands every agent the full state and voids the
        #     communication masking (enforced again in the model constructor).
        raise ValueError(
            "--encode_state is unsafe in this setup. For {} the encoder must "
            "run on per-agent observations.".format(all_args.algorithm_name))

    if all_args.episode_length != all_args.time_limit:
        print("[warn] episode_length ({}) != time_limit ({}); rollouts will not "
              "align with episode boundaries."
              .format(all_args.episode_length, all_args.time_limit))

    # cuda
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

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") \
        / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir()
                         if str(folder.name).startswith('run')]
        curr_run = 'run1' if len(exst_run_nums) == 0 else 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" +
        str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    num_agents = num_agents_from(all_args)
    all_args.run_dir = run_dir
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
