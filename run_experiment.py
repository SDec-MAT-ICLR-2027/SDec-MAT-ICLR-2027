#!/usr/bin/env python3
"""Run one paper experiment, or print the command it would run.

Joint-Action Navigation (tasks e2, e2p, e4, e4m1, e4m1_floor005):

    python run_experiment.py --task e4 --arm sdec_mat --seed 1
    python run_experiment.py --task e4 --arm all --seed 1 2 3
    python run_experiment.py --task e2 --arm sdec_mat --seed 1 --dry-run
    python run_experiment.py --base --arm sdec_mat --seed 1        # no switches

MAC benchmark (tasks pickup_consensus, aerial_ordering, charging_rendezvous,
team_partition; arms sdec_mat, mat_dec, mappo, commnet, tarmac, qmix,
informarl):

    python run_experiment.py --task team_partition --arm sdec_mat
    python run_experiment.py --task pickup_consensus --arm all --dry-run
    python run_experiment.py --task pickup_consensus --arm qmix --agent-id off --seed 41

  Without --seed / --agent-id, a MAC run covers exactly the seeds and agent-id
  conditions the paper reports for that task and method (configs/mac.py).

The configuration lives in `configs/`. For Joint-Action Navigation `common.py`
is shared by everything, `tasks.py` holds the per-task rule and switch block,
`arms.py` the four methods; `mac.py` holds the whole MAC protocol. Composing the
command line here rather than duplicating it per task is what keeps every
method within a task on an identical footing.

Weights & Biases is off unless --wandb-entity is given (or WANDB_ENTITY is set);
it applies to Joint-Action Navigation. MAC runs log to TensorBoard under
results/ (see README).
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from configs import ARMS, ENV, NET, PPO, TASKS          # noqa: E402
from configs import mac as MAC                           # noqa: E402
from configs.common import (ENV_GRAPH_DROP, ENV_GRAPH_EXTRA,  # noqa: E402
                            ENV_GRAPH_RENAMES)
from configs.tasks import switches_for                   # noqa: E402

MAT_ROOT = os.path.join(HERE, "Multi-Agent-Transformer_ctsde")
GRAPH_ROOT = os.path.join(HERE, "InforMARL")
EPYMARL_ROOT = os.path.join(HERE, "epymarl")
ENTRY = {
    "mat": (os.path.join(MAT_ROOT, "mat", "scripts", "train"), "train_joint_nav.py"),
    "graph": (GRAPH_ROOT, os.path.join("onpolicy", "scripts", "train_mpe.py")),
}


def as_flags(d):
    """{name: value} -> ['--name', 'value'], with True meaning a bare flag.

    Several flags in both code bases are argparse `store_false`, so passing them
    DISABLES the feature. Those are spelled as bare `True` here and emitted
    without a value; never give them an explicit 'True'/'False' string.
    """
    out = []
    for k, v in d.items():
        if v is None:
            continue
        if v is True:
            out.append(f"--{k}")
        else:
            out += [f"--{k}", str(v)]
    return out


def build(task, arm_key, seed, base=False, steps=None, threads=None,
          entity=None, project=None, name=None):
    arm = ARMS[arm_key]
    graph = arm["entry"] == "graph"

    env = dict(ENV)
    if not base:
        env.update(TASKS[task]["env"])
    else:
        env.update({"fail_rew": TASKS[task]["env"].get("fail_rew", -1.0)})
    if graph:
        for src, dst in ENV_GRAPH_RENAMES.items():
            if src in env:
                env[dst] = env.pop(src)
        for k in ENV_GRAPH_DROP:
            env.pop(k, None)
        env.update(ENV_GRAPH_EXTRA)

    ppo = dict(PPO)
    ppo["num_env_steps"] = steps if steps is not None else TASKS[task]["num_env_steps"]
    if threads is not None:
        ppo["n_rollout_threads"] = threads

    argv = list(as_flags(arm["flags"]))
    argv += ["--experiment_name", name or f"{task}_{arm_key}_s{seed}"]
    argv += ["--seed", str(seed)]
    argv += as_flags(env)
    argv += as_flags(ppo)
    argv += as_flags(switches_for(task, base=base))
    if not graph:
        argv += as_flags(NET)

    if entity:
        if graph:
            argv += ["--user_name", entity, "--project_name", project or "jointnav"]
        else:
            argv += ["--wandb_on", "--user_name", entity,
                     "--wandb_project", project or "jointnav"]
    elif graph:
        # InforMARL's --use_wandb is store_false: passing it turns W&B OFF.
        argv += ["--use_wandb"]

    cwd, script = ENTRY[arm["entry"]]
    return cwd, [sys.executable, "-u", script] + argv


def build_mac(task, arm_key, seed, agent_id, steps=None, threads=None, name=None):
    """(cwd, argv, PYTHONPATH entries) for one MAC run.

    `agent_id` is "on" or "off": whether agents get a one-hot id appended to
    their observation.
    """
    t = MAC.TASKS[task]
    stack = MAC.ARMS[arm_key]["stack"]
    steps = steps if steps is not None else t["steps"]
    aid = agent_id == "on"
    suffix = "_aid" if aid else ""

    if stack == "mat":
        flags = dict(MAC.MAT_ARM_FLAGS[arm_key])
        flags.update(MAC.MAT_COMMON)
        flags.update(t["mat_flags"])
        flags.update({
            "env_name": t["mat_env_name"],
            "experiment_name": name or f"{arm_key}_s{seed}{suffix}",
            "seed": seed,
            "obs_agent_id": "true" if aid else "false",
            "episode_length": t["episode_length"],
            "num_env_steps": steps,
        })
        if threads is not None:
            flags["n_rollout_threads"] = threads
        cwd = os.path.join(MAT_ROOT, "mat", "scripts", "train")
        return cwd, [sys.executable, "-u", t["mat_script"]] + as_flags(flags), [MAT_ROOT]

    if stack == "epymarl":
        alg = MAC.ARMS[arm_key]["config"]
        overrides = {
            "seed": seed,
            "obs_agent_id": aid,
            "t_max": steps,
            "name": name or f"{alg}{suffix}",
        }
        if alg == "qmix":
            overrides["epsilon_anneal_time"] = t["qmix_eps_anneal"][agent_id]
        overrides.update(MAC.EPYMARL_COMMON)
        # EPyMARL's own parallel runner always uses batch_size_run=10 here:
        # the on-policy batch and buffer sizes are tied to it, so --threads
        # does not apply.
        argv = [sys.executable, "-u", os.path.join("epymarl", "main.py"),
                f"--config={alg}", f"--env-config={task}", "with"]
        argv += [f"{k}={v}" for k, v in overrides.items()]
        return HERE, argv, []

    # InforMARL
    flags = dict(MAC.INFORMARL_COMMON)
    flags.update({
        "env_config": task,
        "name": name or f"informarl{suffix}",
        "seed": seed,
        "num_env_steps": steps,
        "episode_length": t["episode_length"],
        "obs_agent_id": "True" if aid else "False",
    })
    if threads is not None:
        flags["n_rollout_threads"] = threads
    argv = [sys.executable, "-u", os.path.join("mac_adapter", "train_mac.py")]
    return GRAPH_ROOT, argv + as_flags(flags), [GRAPH_ROOT]


def run_mac(a, ap):
    arms = list(MAC.ARMS) if a.arm == "all" else [a.arm]
    for k in arms:
        if k not in MAC.ARMS:
            ap.error(f"unknown MAC arm {k!r}; choose from {list(MAC.ARMS)} or 'all'")
    rc = 0
    for arm_key in arms:
        if a.agent_id == "paper":
            conds = MAC.id_conditions_for(a.task, arm_key)
        elif a.agent_id == "both":
            conds = ("off", "on")
        else:
            conds = (a.agent_id,)
        seeds = a.seed if a.seed else MAC.seeds_for(a.task, arm_key)
        for cond in conds:
            for s in seeds:
                cwd, cmd, paths = build_mac(a.task, arm_key, s, cond,
                                            steps=a.steps, threads=a.threads)
                label = f"{a.task}/{arm_key}/id-{cond}/seed{s}"
                if a.dry_run:
                    print(f"# {label}\ncd {cwd} && \\\n  " + " ".join(cmd[1:]) + "\n")
                    continue
                env = dict(os.environ)
                env["PYTHONPATH"] = os.pathsep.join(
                    paths + [env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
                print(f"=== {label}", flush=True)
                r = subprocess.run(cmd, cwd=cwd, env=env)
                if r.returncode != 0:
                    print(f"!!! {label} exited {r.returncode}", flush=True)
                    rc = r.returncode
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=sorted(TASKS) + list(MAC.TASKS), default="e4")
    ap.add_argument("--arm", default="sdec_mat",
                    help="Joint-Action Navigation: one of %s; MAC: one of %s; "
                         "or 'all'" % (", ".join(sorted(ARMS)), ", ".join(MAC.ARMS)))
    ap.add_argument("--seed", type=int, nargs="+", default=None,
                    help="default: the task's reported seed set")
    ap.add_argument("--agent-id", choices=("paper", "on", "off", "both"),
                    default="paper",
                    help="MAC only: one-hot agent ids in the observation. "
                         "'paper' (default) runs the conditions the paper reports")
    ap.add_argument("--base", action="store_true",
                    help="all three switches off (the paper's base model)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the command instead of running it")
    a = ap.parse_args()

    if a.task in MAC.TASKS:
        if a.base:
            ap.error("--base applies to Joint-Action Navigation only")
        return run_mac(a, ap)

    arms = sorted(ARMS) if a.arm == "all" else [a.arm]
    for k in arms:
        if k not in ARMS:
            ap.error(f"unknown arm {k!r}; choose from {sorted(ARMS)} or 'all'")
    seeds = a.seed if a.seed else list(TASKS[a.task]["seeds"])

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [MAT_ROOT, GRAPH_ROOT, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.setdefault("INFORMARL_ROOT", GRAPH_ROOT)

    rc = 0
    for arm_key in arms:
        for s in seeds:
            cwd, cmd = build(a.task, arm_key, s, base=a.base, steps=a.steps,
                             threads=a.threads, entity=a.wandb_entity,
                             project=a.wandb_project)
            label = f"{a.task}/{arm_key}/seed{s}" + ("  [base]" if a.base else "")
            if a.dry_run:
                print(f"# {label}\ncd {cwd} && \\\n  " + " ".join(cmd[1:]) + "\n")
                continue
            print(f"=== {label}", flush=True)
            r = subprocess.run(cmd, cwd=cwd, env=env)
            if r.returncode != 0:
                print(f"!!! {label} exited {r.returncode}", flush=True)
                rc = r.returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
