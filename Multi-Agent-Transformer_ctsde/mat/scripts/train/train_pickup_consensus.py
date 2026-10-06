#!/usr/bin/env python
"""Train script for the (single-group) Pickup Consensus benchmark.

Derived from `train_aerial.py`. The env is built from the paper
configuration (mac/presets.py) with its reward constants untouched (see
`mat/envs/pickup_consensus/pickup_consensus_mat_wrapper.py`), so a run here is
dynamics- and reward-equivalent to the EPyMARL and InforMARL baselines.

  * `--comm_mask_kind bits` is REQUIRED for mat_comm / mat_comm_gru (the
    wrapper appends n_agents comm bits, always 1, to the state tail — the env
    itself has no topology). `assert_comm_mask_kind` refuses to launch if it
    is wrong. Unlike Aerial Ordering's star graph or Multiple Pickup's
    disjoint groups, this always partitions into exactly ONE group covering
    every agent, so mat_comm's value here is the single sequential decode
    chain across the whole identical team, not a communication constraint.
  * Episodes truncate at `time_limit` (no navigation, so nothing before that
    is a real terminal unless someone commits), so `--episode_length` should
    equal `--time_limit`.

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
from mat.envs.pickup_consensus.pickup_consensus_mat_wrapper import (
    ENV_ARG_DEFAULTS,
    PickupConsensusMATWrapper,
    assert_comm_mask_kind,
    num_agents_from,
)
from mat.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from mat.runner.shared.pickup_consensus_runner import PickupConsensusRunner as Runner


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "PickupConsensus":
                raise NotImplementedError(
                    "Can not support the " + all_args.env_name + " environment.")
            env = PickupConsensusMATWrapper(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "PickupConsensus":
                raise NotImplementedError(
                    "Can not support the " + all_args.env_name + " environment.")
            env = PickupConsensusMATWrapper(all_args)
            # Disjoint from the training seeds so eval episodes are unseen.
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='pickup_consensus',
                        help="Name used for the results directory")
    parser.add_argument('--run_dir', type=str, default='')

    # PickupConsensusEnv constructor arguments. Defaults are the paper
    # configuration (mac/presets.py).
    parser.add_argument('--n_agents', type=int, default=ENV_ARG_DEFAULTS["n_agents"],
                        help="Team size N. obs_dim and state_dim are always 2.")
    parser.add_argument('--k_min', type=int, default=ENV_ARG_DEFAULTS["k_min"],
                        help="Inclusive lower bound on the required count K. MUST be "
                             ">= 2 (k_min=1 lets a symmetric policy use the "
                             "zero-commit non-event as a free filter and destroys "
                             "the ALL-WAIT benchmark result).")
    parser.add_argument('--k_max', type=int, default=ENV_ARG_DEFAULTS["k_max"],
                        help="Inclusive upper bound on K; must be <= n_agents-1.")
    parser.add_argument('--n_max', type=int, default=ENV_ARG_DEFAULTS["n_max"],
                        help="Normaliser for the team_size obs field only.")
    parser.add_argument('--time_limit', type=int, default=ENV_ARG_DEFAULTS["time_limit"],
                        help="Steps before truncation; keep --episode_length equal "
                             "to this")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True

    # Refuses to launch a comm run whose state-tail format would be misread.
    assert_comm_mask_kind(all_args)

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
