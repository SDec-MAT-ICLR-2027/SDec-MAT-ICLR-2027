#!/usr/bin/env python
"""Train InforMARL (graph rMAPPO) on an MAC benchmark env.

    cd InforMARL
    python -u mac_adapter/train_mac.py --env_config pickup_consensus --seed 42 \
        --name baseline_informarl [InforMARL flags ...] [--env_args key=value ...]

--env_config names epymarl/config/envs/<name>.yaml; its env_args build the env exactly
as the EPyMARL runs do, and its test_nepisode / test_interval / runner_log_interval /
learner_log_interval / t_max are used unless overridden here.

All of InforMARL's own flags (onpolicy/config.py) work unchanged. Careful: several
of them are store_false switches (--use_ReLU, --use_valuenorm, --use_centralized_V,
--share_policy, --use_recurrent_policy, --cuda): passing one turns it OFF.

Outputs:
    results/informarl/tb_logs/<name>_seed<seed>_<key>_<timestamp>/   tensorboard
    results/informarl/models/<name>_seed<seed>_<key>_<timestamp>/<t_env>/{actor,critic}.pt
    results/informarl/runs/<name>_seed<seed>_<key>_<timestamp>/config.json
"""

import datetime
import json
import os
import sys
from distutils.util import strtobool

_INFORMARL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _INFORMARL_ROOT not in sys.path:
    sys.path.insert(0, _INFORMARL_ROOT)

from mac_adapter.env_wrapper import (  # noqa: E402
    EDGE_VALUE, REPO_ROOT, MACGraphEnv, ensure_paths, load_env_config,
    parse_env_overrides,
)

ensure_paths()

import numpy as np  # noqa: E402
import setproctitle  # noqa: E402
import torch  # noqa: E402

from onpolicy.config import get_config, graph_config  # noqa: E402
from utils.utils import print_args, print_box  # noqa: E402


def add_mac_args(parser):
    parser.add_argument("--env_config", type=str, required=True,
                        help="name of epymarl/config/envs/<name>.yaml")
    parser.add_argument("--env_args", type=str, nargs="*", default=[],
                        help="env_args overrides, e.g. --env_args comm_radius=2 k_min=2")
    parser.add_argument("--name", type=str, default="informarl",
                        help="run name prefix (results/informarl/tb_logs/<name>_seed..)")
    parser.add_argument("--test_nepisode", type=int, default=None)
    parser.add_argument("--test_interval", type=int, default=None)
    parser.add_argument("--runner_log_interval", type=int, default=None)
    parser.add_argument("--learner_log_interval", type=int, default=None)
    parser.add_argument("--save_model_interval", type=int, default=100000)
    parser.add_argument("--eval_deterministic", type=lambda x: bool(strtobool(x)),
                        default=False,
                        help="False (default) samples test actions like the EPyMARL "
                             "baselines; True uses InforMARL's argmax eval")
    parser.add_argument("--comm_symmetric", type=lambda x: bool(strtobool(x)),
                        default=False,
                        help="symmetrise adjacency comm masks (CommNet/TarMAC comm_symmetric)")
    parser.add_argument("--obs_agent_id", type=lambda x: bool(strtobool(x)),
                        default=False,
                        help="append a one-hot agent index to each observation "
                             "(EPyMARL obs_agent_id)")
    # InforMARL's runner reads these; they are set from the env below.
    parser.add_argument("--num_agents", type=int, default=0)
    parser.add_argument("--num_obstacles", type=int, default=0)
    parser.set_defaults(
        env_name="GraphMPE",        # selects GR_MAPPO / GR_MAPPOPolicy / GraphReplayBuffer
        algorithm_name="rmappo",
        max_edge_dist=2.0,          # must exceed EDGE_VALUE so comm edges are kept
        use_wandb=False,
        use_eval=True,
        num_env_steps=-1,           # -1 -> t_max of the env config
        n_eval_rollout_threads=0,   # 0 -> same as n_rollout_threads
    )
    return parser


def main(argv):
    parser = add_mac_args(get_config())
    all_args, parser = graph_config(argv, parser)

    env_cfg = load_env_config(all_args.env_config)
    if env_cfg.get("env", "gymma") != "gymma":
        raise ValueError(f"{all_args.env_config}.yaml is not a gymma env")
    env_args = dict(env_cfg["env_args"])
    env_args.update(parse_env_overrides(all_args.env_args))
    common_reward = bool(env_cfg.get("common_reward", True))
    reward_scalarisation = env_cfg.get("reward_scalarisation", "sum")

    for key, fallback in (("test_nepisode", 10), ("test_interval", 50000),
                          ("runner_log_interval", 2000), ("learner_log_interval", 2000)):
        if getattr(all_args, key) is None:
            setattr(all_args, key, int(env_cfg.get(key, fallback)))
    if all_args.num_env_steps is None or all_args.num_env_steps < 0:
        all_args.num_env_steps = int(env_cfg.get("t_max", 1000000))
    if all_args.n_eval_rollout_threads <= 0:
        all_args.n_eval_rollout_threads = all_args.n_rollout_threads
    if all_args.max_edge_dist <= EDGE_VALUE:
        raise ValueError(f"--max_edge_dist must be > {EDGE_VALUE}, otherwise no comm edge survives")

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, \
            "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert not all_args.use_recurrent_policy and not all_args.use_naive_recurrent_policy, \
            "check recurrent policy!"

    probe = MACGraphEnv(env_args, common_reward, reward_scalarisation, seed=all_args.seed,
                          comm_symmetric=all_args.comm_symmetric,
                          obs_agent_id=all_args.obs_agent_id)
    env_info = probe.describe()
    probe.close()
    all_args.num_agents = env_info["n_agents"]

    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
    torch.set_num_threads(all_args.n_training_threads)

    key = env_args.get("key", all_args.env_config)
    token = (f"{all_args.name}_seed{all_args.seed}_{key}_"
             f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    results = os.path.join(REPO_ROOT, "results", "informarl")
    tb_dir = os.path.join(results, "tb_logs", token)
    models_dir = os.path.join(results, "models", token)
    run_dir = os.path.join(results, "runs", token)
    os.makedirs(run_dir, exist_ok=True)
    all_args.experiment_name = token

    if all_args.verbose:
        print_args(all_args)
    print_box(f"env: {key}  {env_info}\ntb_logs: {tb_dir}", 80)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(all_args), "env_args": env_args,
                   "common_reward": common_reward,
                   "reward_scalarisation": reward_scalarisation,
                   "env_info": env_info}, f, indent=2, default=str)

    setproctitle.setproctitle(f"informarl-{key}-seed{all_args.seed}")
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    from pathlib import Path
    from mac_adapter.runner import MACGraphRunner
    from mac_adapter.parallel_envs import make_parallel_envs

    # Same per-env seeding as epymarl/runners/parallel_runner.py (seed + i).
    T = all_args.n_rollout_threads
    envs = make_parallel_envs(env_args, common_reward, reward_scalarisation,
                              [all_args.seed + i for i in range(T)], all_args.comm_symmetric,
                              all_args.obs_agent_id)
    eval_envs = make_parallel_envs(env_args, common_reward, reward_scalarisation,
                                   [all_args.seed + 100000 + i
                                    for i in range(all_args.n_eval_rollout_threads)],
                                   all_args.comm_symmetric,
                                   all_args.obs_agent_id) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": Path(run_dir),
        "tb_dir": tb_dir,
        "models_dir": models_dir,
        "common_reward": common_reward,
    }
    runner = MACGraphRunner(config)
    if all_args.verbose:
        print_box("Actor Network", 80)
        print_box(runner.policy.actor, 80)
        print_box("Critic Network", 80)
        print_box(runner.policy.critic, 80)
    try:
        runner.run()
    finally:
        envs.close()
        if eval_envs is not None:
            eval_envs.close()
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
