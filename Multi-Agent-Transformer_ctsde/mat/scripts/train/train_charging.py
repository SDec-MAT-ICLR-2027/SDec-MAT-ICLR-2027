#!/usr/bin/env python
"""Train script for the Charging Rendezvous benchmark.

Derived from `train_aerial.py`. Two differences worth knowing:

  * Team size is `--n_agents` outright, not `1 + n_workers`, so `num_agents_from`
    is a direct read. It still has to happen before an env is built.
  * `--cohort_sizes` / `--cohort_distances` are list args and must satisfy
    sum(cohort_sizes) == n_agents; the env raises otherwise. They are validated
    here too so the failure arrives before 32 subprocesses are spawned.

The env-arg defaults are the paper configuration (mac/presets.py), so a bare
run reproduces the paper's task. Override any of them on the command line.

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
from mat.envs.charging.charging_rendezvous_mat_wrapper import (
    ChargingRendezvousMATWrapper,
    ENV_ARG_DEFAULTS,
    num_agents_from,
)
from mat.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from mat.runner.shared.charging_runner import ChargingRunner as Runner


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "Charging":
                raise NotImplementedError(
                    "Can not support the " + all_args.env_name + " environment.")
            env = ChargingRendezvousMATWrapper(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "Charging":
                raise NotImplementedError(
                    "Can not support the " + all_args.env_name + " environment.")
            env = ChargingRendezvousMATWrapper(all_args)
            # Disjoint from the training seeds so eval episodes are unseen.
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='charging_rendezvous',
                        help="Name used for the results directory")
    parser.add_argument('--run_dir', type=str, default='')

    # ChargingRendezvousEnv constructor arguments. Defaults are the paper
    # configuration (mac/presets.py).
    parser.add_argument('--n_agents', type=int, default=ENV_ARG_DEFAULTS["n_agents"],
                        help="Team size (number of drones)")
    parser.add_argument('--grid_size', type=int, default=ENV_ARG_DEFAULTS["grid_size"])
    parser.add_argument('--capacity', type=int, default=ENV_ARG_DEFAULTS["capacity"],
                        help="Charging ports at the single station (the C in exact-C)")
    parser.add_argument('--cohort_sizes', type=int, nargs='+',
                        default=list(ENV_ARG_DEFAULTS["cohort_sizes"]),
                        help="Drones per cohort; must sum to n_agents. A cohort "
                             "larger than capacity is what forces the contested "
                             "exact-C selection")
    parser.add_argument('--cohort_distances', type=int, nargs='+',
                        default=list(ENV_ARG_DEFAULTS["cohort_distances"]),
                        help="Spawn distance per cohort; staggers arrival waves")
    parser.add_argument('--init_battery', type=float, nargs='+',
                        default=list(ENV_ARG_DEFAULTS["init_battery"]),
                        help="BASE ARRIVAL battery range, sampled once per episode. "
                             "Pass one value to pin it. [0.06 0.12] is the hard regime")
    parser.add_argument('--battery_sigma', type=float,
                        default=ENV_ARG_DEFAULTS["battery_sigma"],
                        help="Per-drone jitter on arrival battery. 0 => exact "
                             "within-cohort ties (hardest); raise to soften")
    parser.add_argument('--target_level', type=float, default=ENV_ARG_DEFAULTS["target_level"])
    parser.add_argument('--drain_rate', type=float, default=ENV_ARG_DEFAULTS["drain_rate"])
    parser.add_argument('--charge_rate', type=float, default=ENV_ARG_DEFAULTS["charge_rate"])
    parser.add_argument('--comm_radius', type=int, default=ENV_ARG_DEFAULTS["comm_radius"],
                        help="Manhattan radius of the station's mesh; everyone "
                             "inside it forms ONE clique, everyone else is a singleton")
    parser.add_argument('--dock_radius', type=int, default=ENV_ARG_DEFAULTS["dock_radius"])
    parser.add_argument('--shaping_weight', type=float,
                        default=ENV_ARG_DEFAULTS["shaping_weight"],
                        help="Potential-based approach shaping; 0.0 for the pure reward")
    parser.add_argument('--wind_prob', type=float, default=ENV_ARG_DEFAULTS["wind_prob"])
    parser.add_argument('--time_limit', type=int, default=ENV_ARG_DEFAULTS["time_limit"],
                        help="Episode limit; keep --episode_length equal to this")

    all_args = parser.parse_known_args(args)[0]

    # Fail here rather than inside 32 freshly-spawned subprocesses.
    if len(all_args.cohort_sizes) != len(all_args.cohort_distances):
        raise ValueError(
            "--cohort_sizes and --cohort_distances must have equal length, got "
            f"{all_args.cohort_sizes} and {all_args.cohort_distances}.")
    if sum(all_args.cohort_sizes) != all_args.n_agents:
        raise ValueError(
            f"sum(--cohort_sizes)={sum(all_args.cohort_sizes)} must equal "
            f"--n_agents={all_args.n_agents}.")
    if len(all_args.init_battery) not in (1, 2):
        raise ValueError(
            "--init_battery takes one value (pinned) or two (low high), got "
            f"{all_args.init_battery}.")
    if len(all_args.init_battery) == 1:
        all_args.init_battery = all_args.init_battery * 2
    if all(s <= all_args.capacity for s in all_args.cohort_sizes):
        print(f"[warn] every cohort is <= capacity ({all_args.capacity}); the "
              f"contested exact-C selection never arises, so this run does not "
              f"exercise what the benchmark measures.")

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
        #     encoding it hands every agent the full state -- every battery,
        #     every position -- and voids the communication masking (enforced
        #     again in the model constructor).
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
