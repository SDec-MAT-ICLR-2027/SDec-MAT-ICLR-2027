#!/usr/bin/env python
import argparse
from distutils.util import strtobool
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

import os, sys

sys.path.append(os.path.abspath(os.getcwd()))

from utils.utils import print_args, print_box, connected_to_internet
from onpolicy.config import get_config
from multiagent.MPE_env import MPEEnv, GraphMPEEnv
from onpolicy.envs.env_wrappers import (
    SubprocVecEnv,
    DummyVecEnv,
    GraphSubprocVecEnv,
    GraphDummyVecEnv,
)

"""Train script for MPEs."""


def make_train_env(all_args: argparse.Namespace):
    def get_env_fn(rank: int):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            elif all_args.env_name == "GraphMPE":
                env = GraphMPEEnv(all_args)
            else:
                print(f"Can not support the {all_args.env_name} environment")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        if all_args.env_name == "GraphMPE":
            return GraphDummyVecEnv([get_env_fn(0)])
        return DummyVecEnv([get_env_fn(0)])
    else:
        if all_args.env_name == "GraphMPE":
            return GraphSubprocVecEnv(
                [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
            )
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args: argparse.Namespace):
    def get_env_fn(rank: int):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            elif all_args.env_name == "GraphMPE":
                env = GraphMPEEnv(all_args)
            else:
                print(f"Can not support the {all_args.env_name} environment")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        if all_args.env_name == "GraphMPE":
            return GraphDummyVecEnv([get_env_fn(0)])
        return DummyVecEnv([get_env_fn(0)])
    else:
        if all_args.env_name == "GraphMPE":
            return GraphSubprocVecEnv(
                [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
            )
        return SubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
        )


def parse_args(args, parser):
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="simple_spread",
        help="Which scenario to run on",
    )
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=2, help="number of players")
    parser.add_argument(
        "--num_obstacles", type=int, default=3, help="Number of obstacles"
    )
    parser.add_argument(
        "--collaborative",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Number of agents in the env",
    )
    parser.add_argument(
        "--max_speed",
        type=float,
        default=2,
        help="Max speed for agents. NOTE that if this is None, "
        "then max_speed is 2 with discrete action space",
    )
    parser.add_argument(
        "--collision_rew",
        type=float,
        default=5,
        help="The reward to be negated for collisions with other "
        "agents and obstacles",
    )
    parser.add_argument(
        "--goal_rew",
        type=float,
        default=5,
        help="The reward to be added if agent reaches the goal",
    )
    parser.add_argument(
        "--min_dist_thresh",
        type=float,
        default=0.05,
        help="The minimum distance threshold to classify whether "
        "agent has reached the goal or not",
    )
    parser.add_argument(
        "--use_dones",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether we want to use the 'done=True' "
        "when agent has reached the goal or just return False like "
        "the `simple.py` or `simple_spread.py`",
    )

    # ------------------------------------------------------------------
    # joint-capture navigation (`navigation_joint_graph`) task parameters.
    # Names, types and defaults are copied verbatim from the MAT fork's
    # `mat/scripts/train/train_joint_nav.py` so the SAME command-line values
    # describe the SAME task on both sides. Do not let the two drift.
    # ------------------------------------------------------------------
    parser.add_argument("--num_targets", type=int, default=3)
    parser.add_argument(
        "--k_req",
        type=int,
        default=3,
        help="Simultaneous capturers needed to fell a target.",
    )
    parser.add_argument(
        "--k_max",
        type=int,
        default=None,
        help="Exact-k capture (E4): more than k_max in-range capturers on a "
        "target is an overkill (no capture, fail_rew). Default None = the "
        "original >= k_req rule.",
    )
    parser.add_argument("--capture_radius", type=float, default=0.3)
    parser.add_argument(
        "--group_radius",
        type=float,
        default=0.6,
        help="Radius for FORMING joint-decision groups (>= capture_radius). "
        "Unused by InforMARL itself, but the scenario validates it against "
        "capture_radius and min_target_sep, so keep it equal to the value "
        "used for the CT-SDE arms.",
    )
    parser.add_argument("--capture_rew", type=float, default=10.0)
    parser.add_argument("--fail_rew", type=float, default=-1.0)
    parser.add_argument("--assembly_rew", type=float, default=0.05)
    parser.add_argument("--dist_rew_scale", type=float, default=1.0)
    parser.add_argument(
        "--potential_shaping",
        type=int,
        default=1,
        help="1 (default): dense shaping as a potential difference. Must stay "
        "1 for any real run.",
    )
    parser.add_argument(
        "--shaping_gamma",
        type=float,
        default=0.99,
        help="Discount used in the shaping potential.",
    )
    parser.add_argument(
        "--obs_mode",
        type=str,
        default="shared",
        choices=["local", "shared", "radius"],
        help="Actor observation. `radius` is the partially observable setting "
        "used by E2/E3/E4.",
    )
    parser.add_argument(
        "--sense_radius",
        type=float,
        default=1.0,
        help="obs_mode=radius: agents perceive agents/targets only within "
        "this distance. Set --max_edge_dist to the same value so InforMARL's "
        "graph neighbourhood matches what the agent can actually sense.",
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
        "--comm_mode",
        type=str,
        default="target_clique",
        choices=["target_clique", "agent_radius"],
        help="How execution groups are formed. InforMARL does not use groups; "
        "kept so the same command line configures both sides.",
    )
    parser.add_argument(
        "--min_target_sep",
        type=float,
        default=None,
        help="Minimum distance between targets (default 2*group_radius).",
    )
    parser.add_argument("--shuffle_targets", type=int, default=0)
    parser.add_argument(
        "--target_order", type=str, default="id", choices=["id", "distance"]
    )
    parser.add_argument("--obs_ready_count", type=int, default=0)

    # ------------------------------------------------------------------
    # The three CT-SDE switches. Flag names, types and defaults mirror the
    # MAT fork (`mat/config.py`) EXACTLY so the same command line configures
    # both codebases. InforMARL has independent actors, so `behav_w_mode arm`
    # evaluates the marginal weight -- the same one the MAT fork uses for its
    # `mappo` and `mat_comm_dec` arms.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--commit_groups",
        type=str,
        default="comm",
        choices=["comm", "capture"],
        help="How exploration groups are formed. `comm` = connected "
        "components of the agent-radius graph (task-agnostic).",
    )
    parser.add_argument(
        "--group_commit_eps",
        type=float,
        default=0.0,
        help="Per group per step, probability of drawing ONE shared random "
        "action for the group. 0 = exploration switch off.",
    )
    parser.add_argument(
        "--group_commit_eps_final",
        type=float,
        default=-1.0,
        help="If >= 0, linearly decay group_commit_eps to this value.",
    )
    parser.add_argument(
        "--group_commit_decay_frac",
        type=float,
        default=1.0,
        help="Fraction of training over which the decay completes.",
    )
    parser.add_argument(
        "--commit_member_prob",
        type=float,
        default=1.0,
        help="q: probability each group member joins the shared action. "
        "Use 0.5 on exact-k tasks, 1.0 elsewhere.",
    )
    parser.add_argument(
        "--offpolicy_weight",
        type=int,
        default=0,
        help="1 = multiply advantages by w = min(1, pi/mu).",
    )
    parser.add_argument(
        "--behav_w_mode",
        type=str,
        default="marginal",
        choices=["marginal", "arm"],
        help="Which behaviour distribution mu to use.",
    )
    parser.add_argument(
        "--behav_w_floor",
        type=float,
        default=0.0,
        help="w = max(floor, min(1, pi/mu)). Stops the weight from erasing "
        "the exploration signal. 0 = off.",
    )
    parser.add_argument(
        "--optimistic_adv",
        type=int,
        default=0,
        help="1 = A <- max(opt_leaky_rate*A, A) before standardizing.",
    )
    parser.add_argument(
        "--opt_leaky_rate",
        type=float,
        default=0.0,
        help="Scale applied to negative advantages when optimistic_adv=1.",
    )

    all_args = parser.parse_known_args(args)[0]

    return all_args, parser


def main(args):
    parser = get_config()
    all_args, parser = parse_args(args, parser)
    if all_args.env_name == "GraphMPE":
        from onpolicy.config import graph_config

        all_args, parser = graph_config(args, parser)

    if all_args.algorithm_name in ["rmappo"]:
        assert (
            all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy
        ), "check recurrent policy!"
    elif all_args.algorithm_name in ["mappo"]:
        assert (
            all_args.use_recurrent_policy == False
            and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    else:
        raise NotImplementedError

    assert (
        all_args.share_policy == True
        and all_args.scenario_name == "simple_speaker_listener"
    ) == False, (
        "The simple_speaker_listener scenario can not use shared policy. "
        "Please check the config.py."
    )

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print_box("Choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print_box("Choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    if all_args.verbose:
        print_args(all_args)

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        # for supercloud when no internet_connection
        if not connected_to_internet():
            import json

            # save a json file with your wandb api key in your
            # home folder as {'my_wandb_api_key': 'INSERT API HERE'}
            # NOTE this is only for running on systems without internet access
            # have to run `wandb sync wandb/run_name` to sync logs to wandboard
            with open(os.path.expanduser("~") + "/keys.json") as json_file:
                key = json.load(json_file)
                my_wandb_api_key = key["my_wandb_api_key"]  # NOTE change here as well
            os.environ["WANDB_API_KEY"] = my_wandb_api_key
            os.environ["WANDB_MODE"] = "dryrun"
            os.environ["WANDB_SAVE_CODE"] = "true"

        print_box("Creating wandboard...")
        run = wandb.init(
            config=all_args,
            project=all_args.project_name,
            # project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name)
            + "_"
            + str(all_args.experiment_name)
            + "_seed"
            + str(all_args.seed),
            # group=all_args.scenario_name,
            dir=str(run_dir),
            # job_type="training",
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
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        if all_args.env_name == "GraphMPE":
            from onpolicy.runner.shared.graph_mpe_runner import GMPERunner as Runner
        else:
            from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    else:
        if all_args.env_name == "GraphMPE":
            raise NotImplementedError
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    if all_args.verbose:
        print_box("Actor Network", 80)
        if type(runner.policy) == list:
            print_box(runner.policy[0].actor, 80)
            print_box("Critic Network", 80)
            print_box(runner.policy[0].critic, 80)
        else:
            print_box(runner.policy.actor, 80)
            print_box("Critic Network", 80)
            print_box(runner.policy.critic, 80)
    runner.run()

    # post process
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
