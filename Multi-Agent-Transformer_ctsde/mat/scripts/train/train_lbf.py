#!/usr/bin/env python
"""Train MAPPO (mat_dec), MAT, or mat_comm on Level-Based Foraging.

B vs C: the only intended config diff is --algorithm_name.
    B: --algorithm_name mappo     (alias of mat_dec)
    C: --algorithm_name mat

mat_comm: group-masked AR decoder. This script forces --comm_mask_kind
pairwise and requires --comm_radius (Chebyshev). Soft critic is
--encoder_critic_attn soft_group_bias.
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
from mat.envs.lbf.lbf_mat_wrapper import ENV_ARG_DEFAULTS, LBFMATWrapper
from mat.runner.shared.lbf_runner import LBFRunner as Runner

MAPPO_ALGOS = ("mappo", "mat_dec")
COMM_ALGOS = ("mat_comm", "mat_comm_gru")


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name != "LBF":
                raise NotImplementedError("unsupported env: %s" % all_args.env_name)
            env = LBFMATWrapper(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = LBFMATWrapper(all_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv(
        [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
    )


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="lbf_8x8_4p_6f")
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="jointnav")
    parser.add_argument("--wandb_on", action="store_true", default=False)

    parser.add_argument("--grid_size", type=int, default=ENV_ARG_DEFAULTS["grid_size"])
    parser.add_argument("--n_agents", type=int, default=ENV_ARG_DEFAULTS["n_agents"])
    parser.add_argument("--n_foods", type=int, default=ENV_ARG_DEFAULTS["n_foods"])
    parser.add_argument(
        "--min_player_level", type=int, default=ENV_ARG_DEFAULTS["min_player_level"]
    )
    parser.add_argument(
        "--max_player_level", type=int, default=ENV_ARG_DEFAULTS["max_player_level"]
    )
    parser.add_argument(
        "--min_food_level", type=int, default=ENV_ARG_DEFAULTS["min_food_level"]
    )
    parser.add_argument(
        "--max_food_level", type=int, default=ENV_ARG_DEFAULTS["max_food_level"]
    )
    parser.add_argument("--sight", type=int, default=ENV_ARG_DEFAULTS["sight"])
    parser.add_argument(
        "--food_sight",
        type=int,
        default=ENV_ARG_DEFAULTS["food_sight"],
        help="Chebyshev radius for food slots. -1 means use --sight.",
    )
    parser.add_argument(
        "--agent_sight",
        type=int,
        default=ENV_ARG_DEFAULTS["agent_sight"],
        help="Chebyshev radius for agent slots. -1 means use --sight. "
        "Must equal grid_size for full teammate visibility.",
    )
    parser.add_argument(
        "--max_episode_steps", type=int, default=ENV_ARG_DEFAULTS["max_episode_steps"]
    )
    parser.add_argument("--force_coop", type=int, default=ENV_ARG_DEFAULTS["force_coop"])
    parser.add_argument(
        "--normalize_reward", type=int, default=ENV_ARG_DEFAULTS["normalize_reward"]
    )
    parser.add_argument("--penalty", type=float, default=ENV_ARG_DEFAULTS["penalty"])
    parser.add_argument(
        "--step_cost",
        type=float,
        default=ENV_ARG_DEFAULTS["step_cost"],
        help="Per-step subtract from the team reward AFTER LBF collection "
        "normalization. 0=off (default). Must stay small enough that "
        "horizon*step_cost < smallest food / sum(foods).",
    )
    parser.add_argument(
        "--player_levels",
        type=str,
        default=ENV_ARG_DEFAULTS["player_levels"],
        help="Comma-separated per-agent levels, e.g. 1,1,2,3. Pins roster; "
        "empty means sample from min/max_player_level.",
    )
    parser.add_argument(
        "--pin_player_levels",
        type=int,
        default=ENV_ARG_DEFAULTS["pin_player_levels"],
        help="1=do not permute levels across agent indices. "
        "-1=auto (pin when --player_levels is set).",
    )
    parser.add_argument(
        "--food_levels",
        type=str,
        default=ENV_ARG_DEFAULTS["food_levels"],
        help="Comma-separated per-food levels, e.g. 2,3,3,4. "
        "Empty means sample from min/max_food_level.",
    )
    parser.add_argument(
        "--comm_radius",
        type=int,
        default=-1,
        help="Chebyshev comm radius. Required (>=0) for mat_comm and "
        "for --group_obs 1.",
    )
    parser.add_argument(
        "--group_obs",
        type=int,
        default=0,
        help="0=primary for mat_comm: local gym triples, groups only "
        "mask AR/encoder (default). 1=SECONDARY merge arm: slot-wise "
        "union over the capped comm component plus a per-slot vis bit "
        "(y,x,lvl,vis). share_obs stays the raw gym concat. Do not "
        "use 1 as the CT-SDE claim arm — that confounds joint actions "
        "with observation sharing.",
    )
    parser.add_argument(
        "--max_group_size",
        type=int,
        default=0,
        help="Safety bound on 1-hop neighborhood size for the comm mask "
        "(self + at most K-1 neighbors). Also required (>=1) when "
        "--group_obs 1 (merge still chops connected components).",
    )
    parser.add_argument(
        "--group_cap_mode",
        type=str,
        default="id",
        help="How to apply --max_group_size. For the comm mask: keep the "
        "lowest-ID (id) or nearest Chebyshev (spatial) neighbors. "
        "For group_obs=1 merge: still chops connected components "
        "(id=ID chunks; spatial=tightest Chebyshev-k).",
    )
    parser.add_argument(
        "--group_action_memory",
        type=int,
        default=0,
        help="1=append the group's own joint action from t-1 to every "
        "member's obs (n_agents*n_actions one-hot, zeroed on episode start "
        "or membership change). Requires --group_obs 1. 0=off.",
    )
    parser.add_argument(
        "--food_indexed_load",
        type=int,
        default=0,
        help="1=action space is {NONE,N,S,W,E,LOAD[food_0],...,LOAD[food_k]}. "
        "LOAD[k] is legal iff 4-adjacent to living food slot k; gym LOAD "
        "fires only if LBF would hit that slot. Navigation is still learned. "
        "0=gym {NONE,N,S,W,E,LOAD} (default, old checkpoints).",
    )
    parser.add_argument(
        "--exact_match",
        type=int,
        default=0,
        help="1=LOAD succeeds iff sum(adjacent loaders)==food level. "
        "0=sum>=food (gym default). Bystanders who are not LOADing "
        "do not count.",
    )

    parser.set_defaults(episode_length=ENV_ARG_DEFAULTS["max_episode_steps"])
    return parser.parse_known_args(args)[0]


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args.env_name = "LBF"

    if all_args.algorithm_name in MAPPO_ALGOS:
        all_args.dec_actor = True
        all_args.share_actor = True

    if int(getattr(all_args, "group_obs", 0)) not in (0, 1):
        raise ValueError("group_obs must be 0 or 1, got %r" % all_args.group_obs)
    if int(all_args.group_obs) == 1:
        if int(all_args.comm_radius) < 0:
            raise ValueError(
                "group_obs=1 requires --comm_radius >= 0, got %r"
                % all_args.comm_radius
            )
        if int(all_args.max_group_size) < 1:
            raise ValueError(
                "group_obs=1 requires --max_group_size >= 1, got %r"
                % all_args.max_group_size
            )
        print(
            "[train_lbf] group_obs=1 SECONDARY merge arm "
            "comm_radius=%s max_group_size=%s cap_mode=%s "
            "(actor=union + vis bit, share_obs=raw gym concat%s). "
            "Primary CT-SDE is group_obs=0."
            % (
                all_args.comm_radius,
                all_args.max_group_size,
                getattr(all_args, "group_cap_mode", "id"),
                "+N×N mask" if all_args.algorithm_name in COMM_ALGOS else "",
            )
        )
    _cap_mode = str(getattr(all_args, "group_cap_mode", "id"))
    if _cap_mode not in ("id", "spatial"):
        raise ValueError("group_cap_mode must be id or spatial, got %r" % _cap_mode)
    if _cap_mode == "spatial" and int(all_args.max_group_size) < 1:
        raise ValueError("group_cap_mode=spatial requires --max_group_size >= 1")

    if int(getattr(all_args, "food_indexed_load", 0)) not in (0, 1):
        raise ValueError(
            "food_indexed_load must be 0 or 1, got %r"
            % all_args.food_indexed_load
        )
    if int(all_args.food_indexed_load) == 1:
        _n_act = 5 + int(all_args.n_foods)
        print(
            "[train_lbf] food_indexed_load=1 n_actions=%d "
            "(NONE,N,S,W,E + LOAD[food_0..food_%d]); no approach/move_toward"
            % (_n_act, int(all_args.n_foods) - 1)
        )

    if int(all_args.exact_match) not in (0, 1):
        raise ValueError(
            "exact_match must be 0 or 1, got %r" % all_args.exact_match
        )
    print(
        "[train_lbf] exact_match=%s (LOAD succeeds iff sum(loaders) %s food)"
        % (
            int(all_args.exact_match),
            "==" if int(all_args.exact_match) == 1 else ">=",
        )
    )

    if int(getattr(all_args, "group_action_memory", 0)) not in (0, 1):
        raise ValueError(
            "group_action_memory must be 0 or 1, got %r"
            % all_args.group_action_memory
        )
    if int(all_args.group_action_memory) == 1:
        if int(all_args.group_obs) != 1:
            raise ValueError("group_action_memory=1 requires --group_obs 1")
        print(
            "[train_lbf] group_action_memory=1 obs += n_agents*n_actions "
            "(one-hot group joint action from t-1, zeroed on reset/churn)"
        )

    if all_args.algorithm_name in COMM_ALGOS:
        all_args.comm_mask_kind = "pairwise"
        print(
            "[train_lbf] comm_mask_kind forced to pairwise for %s"
            % all_args.algorithm_name
        )
        if int(all_args.comm_radius) < 0:
            raise ValueError(
                "mat_comm on LBF requires --comm_radius >= 0, got %r"
                % all_args.comm_radius
            )
        if all_args.dec_actor:
            raise ValueError("mat_comm is incompatible with --dec_actor")
        if all_args.algorithm_name == "mat_comm_gru" and (
            getattr(all_args, "encoder_critic_attn", "hard") == "soft_group_bias"
        ):
            raise ValueError("encoder_critic_attn=soft_group_bias is not implemented for mat_comm_gru")
        if int(all_args.group_obs) == 0:
            print(
                "[train_lbf] mat_comm PRIMARY: group_obs=0 "
                "(local gym triples; groups only mask AR/encoder)"
            )
        if int(all_args.max_group_size) >= 1:
            print(
                "[train_lbf] mat_comm neighborhood cap max_group_size=%s "
                "(per-row safety bound, not a component clique)"
                % all_args.max_group_size
            )

    _cflag = int(all_args.commitment)
    if _cflag not in (0, 1):
        raise ValueError("commitment must be 0 or 1, got %r" % all_args.commitment)
    if _cflag == 1:
        if all_args.algorithm_name != "mat_comm":
            raise ValueError(
                "commitment=1 is mat_comm only, got %r" % all_args.algorithm_name
            )
        all_args.n_commit_targets = int(all_args.n_foods)
        if all_args.commit_entropy_coef is None:
            raise ValueError(
                "commitment=1 requires --commit_entropy_coef "
                "(H_z; 0 disables it. Action entropy stays --entropy_coef)"
            )
        if int(all_args.commit_action_blind) not in (0, 1):
            raise ValueError(
                "commit_action_blind must be 0 or 1, got %r"
                % all_args.commit_action_blind
            )
        if int(all_args.commit_horizon) < 1:
            raise ValueError(
                "commit_horizon must be >= 1, got %r" % all_args.commit_horizon
            )
        if int(all_args.commit_critic_z) not in (0, 1):
            raise ValueError(
                "commit_critic_z must be 0 or 1, got %r"
                % all_args.commit_critic_z
            )
        if int(all_args.commit_critic_pair) not in (0, 1):
            raise ValueError(
                "commit_critic_pair must be 0 or 1, got %r"
                % all_args.commit_critic_pair
            )
        if int(all_args.commit_critic_z) == 1 and int(all_args.commit_critic_pair) == 1:
            raise ValueError(
                "commit_critic_z and commit_critic_pair are mutually exclusive"
            )
        print(
            "[train_lbf] commitment=1 n_commit_targets=%d (foods) NULL=%d "
            "entropy_coef=%s commit_entropy_coef=%s action_blind=%s "
            "horizon=%s critic_z=%s critic_pair=%s"
            % (all_args.n_commit_targets, all_args.n_commit_targets,
               all_args.entropy_coef, all_args.commit_entropy_coef,
               int(all_args.commit_action_blind),
               int(all_args.commit_horizon),
               int(all_args.commit_critic_z),
               int(all_args.commit_critic_pair))
        )
    else:
        all_args.n_commit_targets = 0
        if int(all_args.commit_horizon) != 1:
            raise ValueError(
                "commit_horizon=%s requires commitment=1"
                % all_args.commit_horizon
            )
        if int(all_args.commit_critic_z) != 0:
            raise ValueError(
                "commit_critic_z=%s requires commitment=1"
                % all_args.commit_critic_z
            )
        if int(all_args.commit_critic_pair) != 0:
            raise ValueError(
                "commit_critic_pair=%s requires commitment=1"
                % all_args.commit_critic_pair
            )

    if float(all_args.penalty) < 0.0:
        raise ValueError("penalty must be >= 0, got %r" % all_args.penalty)
    if float(all_args.penalty) != 0.0:
        print(
            "WARNING: penalty=%s — failed LOAD is taxed; this is the RO lever, "
            "not the default B vs C contract." % all_args.penalty
        )
    if float(all_args.step_cost) < 0.0:
        raise ValueError("step_cost must be >= 0, got %r" % all_args.step_cost)
    if float(all_args.step_cost) > 0.0:
        _h = int(all_args.max_episode_steps)
        _max_tax = _h * float(all_args.step_cost)
        _fl = str(getattr(all_args, "food_levels", "") or "").strip()
        if not _fl:
            raise ValueError(
                "step_cost>0 requires --food_levels so we can refuse a cost "
                "that dominates the smallest normalized food"
            )
        _foods = [int(x) for x in _fl.split(",") if x.strip() != ""]
        if not _foods:
            raise ValueError("step_cost>0 but --food_levels parsed empty")
        _min_frac = min(_foods) / float(sum(_foods))
        print(
            "[train_lbf] step_cost=%s (wrapper subtract; collection norm "
            "unchanged). max tax %.4f over %d steps; min food frac %.4f"
            % (all_args.step_cost, _max_tax, _h, _min_frac)
        )
        if _max_tax >= _min_frac:
            raise ValueError(
                "step_cost * horizon = %.4f >= smallest food frac %.4f; "
                "skipping a food can beat collecting it. Lower --step_cost."
                % (_max_tax, _min_frac)
            )
    _grid = int(all_args.grid_size)
    _sight = int(all_args.sight)
    if _sight < 0 or _sight > _grid:
        raise ValueError("sight (%r) must be in [0, grid_size=%r]" % (_sight, _grid))
    _agent_s = int(all_args.agent_sight)
    if _agent_s < 0:
        _agent_s = _sight
    _food_s = int(all_args.food_sight)
    if _food_s < 0:
        _food_s = _sight
    if _agent_s > _grid:
        raise ValueError("agent_sight (%r) > grid_size (%r)" % (_agent_s, _grid))
    if _food_s > _grid:
        raise ValueError("food_sight (%r) > grid_size (%r)" % (_food_s, _grid))
    if _food_s < _grid or _agent_s < _grid:
        print(
            "WARNING: partial obs — food_sight=%s agent_sight=%s grid=%s "
            "(not the full-obs B vs C contract)." % (_food_s, _agent_s, _grid)
        )
    if int(all_args.force_coop) not in (0, 1):
        raise ValueError("force_coop must be 0 or 1, got %r" % all_args.force_coop)
    if int(all_args.normalize_reward) not in (0, 1):
        raise ValueError(
            "normalize_reward must be 0 or 1, got %r" % all_args.normalize_reward
        )
    if int(all_args.pin_player_levels) not in (-1, 0, 1):
        raise ValueError(
            "pin_player_levels must be -1, 0, or 1, got %r" % all_args.pin_player_levels
        )
    if int(all_args.episode_length) != int(all_args.max_episode_steps):
        raise ValueError(
            "episode_length (%r) must equal max_episode_steps (%r)"
            % (all_args.episode_length, all_args.max_episode_steps)
        )
    if all_args.encode_state:
        raise ValueError(
            "--encode_state is unsafe: vanilla MAT zeros the state, so the "
            "encoder would see only zeros."
        )

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
        / all_args.scenario_name
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
            group=all_args.scenario_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )
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
        "%s-LBF-%s@%s"
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
