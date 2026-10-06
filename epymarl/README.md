# epymarl/ — CTDE baselines for the MAC benchmark

A trimmed copy of **EPyMARL** (Apache-2.0, see `LICENSE`) that trains the four
CTDE baselines of the MAC experiments: **MAPPO**, **CommNet**, **TarMAC** and
**QMIX**.

> Papoudakis, Christianos, Schäfer, Albrecht. *Benchmarking Multi-Agent Deep
> Reinforcement Learning Algorithms in Cooperative Tasks.* NeurIPS Datasets and
> Benchmarks, 2021. https://github.com/uoe-agents/epymarl

Only what those four methods use is kept: the runners, the basic and
communication controllers, the RNN / CommNet / TarMAC agents, the centralised
critic, the PPO and Q learners, the QMIX/VDN mixers and the environment
wrappers (the SMAC, PettingZoo and VMAS wrappers are kept as upstream).

## What differs from upstream EPyMARL

| file | change |
|---|---|
| `modules/agents/commnet_agent.py`, `tarmac_agent.py`, `comm_groups.py` | CommNet and TarMAC agents; messages are exchanged only inside the communication groups the env reports |
| `controllers/comm_mac.py` | controller that passes each step's communication mask to those agents |
| `run.py`, `runners/*.py` | a `comm_mask` field in the episode buffer, filled from the env every step |
| `envs/gymma.py` | forwards the env's own action masks and communication mask |
| `envs/__init__.py` | registers the MAC environments (by importing the `mac` package) |
| `config/envs/*.yaml` | the four MAC tasks, generated from `mac/presets.py` |
| `config/algs/qmix.yaml` | the QMIX settings used in the paper (parallel runner, RNN agent, hidden 128, lr 3e-4) |
| `main.py`, `run.py`, `config/default.yaml` | results are written to `results/epymarl/` |

## Running

Through the repository's launcher, which supplies the paper's seeds, agent-id
conditions and per-task settings:

```bash
python run_experiment.py --task team_partition --arm mappo
python run_experiment.py --task pickup_consensus --arm qmix --agent-id off --seed 41
```

or directly, from the repository root:

```bash
python epymarl/main.py --config=tarmac --env-config=aerial_ordering \
    with seed=42 obs_agent_id=True name=tarmac_aid use_tensorboard=True
```

`--config` is `mappo`, `commnet`, `tarmac` or `qmix`; `--env-config` is
`pickup_consensus`, `aerial_ordering`, `charging_rendezvous` or `team_partition`.
TensorBoard logs go to `results/epymarl/tb_logs/`, sacred records to
`results/epymarl/sacred/`, checkpoints to `results/epymarl/models/`.
