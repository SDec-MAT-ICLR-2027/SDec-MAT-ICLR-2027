# CT-SDE: Centralized Training, Semi-Decentralized Execution

Code for **SDec-MAT**, a communication-constrained Multi-Agent Transformer
whose decoder conditions each agent on the actions of its **one-hop neighbours
only**, so execution needs no central controller and no global broadcast — and
for the two experiment families of the paper:

* **Joint-Action Navigation** (below), and
* the **Multi-Agent Counting benchmark (MAC)** and Pickup Consensus — see
  [MAC benchmark](#mac-benchmark).

On Joint-Action Navigation four methods are compared under an identical
training protocol:

| name in the paper | arm | actor |
|---|---|---|
| **SDec-MAT** | `mat_comm_soft` | group-masked encoder + autoregressive decoder over the 1-hop neighbourhood |
| MAT-Dec | `mat_comm_dec` | same encoder (observation sharing), **no** action conditioning |
| MAPPO | `mappo --dec_actor` | per-agent MLP on its own observation |
| InforMARL | InforMARL codebase | graph neural network over the entity graph |

---

## Built on prior work

This repository is a **fork of the Multi-Agent Transformer (MAT)** and uses it
as its base implementation. The encoder/decoder, the PPO trainer, the shared
replay buffer and the training scripts all derive from MAT; our contribution is
the communication-constrained variant
(`mat/algorithms/mat/algorithm/mat_comm_transformer.py`,
`mat/algorithms/utils/mat_comm_*.py`), the joint-capture navigation task, and
the training and evaluation tooling around them.

> Wen, Kuba, Lin, Zhang, Wen, Wang, Yang.
> *Multi-Agent Reinforcement Learning is a Sequence Modeling Problem.*
> arXiv:2205.14953, 2022. https://github.com/PKU-MARL/Multi-Agent-Transformer

```bibtex
@article{wen2022multi,
  title={Multi-Agent Reinforcement Learning is a Sequence Modeling Problem},
  author={Wen, Muning and Kuba, Jakub Grudzien and Lin, Runji and Zhang, Weinan
          and Wen, Ying and Wang, Jun and Yang, Yaodong},
  journal={arXiv preprint arXiv:2205.14953},
  year={2022}
}
```

The environment base and the graph baseline come from **InforMARL** (MIT
licence, see `InforMARL/LICENSE`). The copy in `InforMARL/` is **modified**, not
upstream: the joint-capture scenario is new, and the training loop was extended
so that the same exploration / off-policy-weight / optimism flags configure both
code bases and both log the same metric names, which is what makes the
comparison like-for-like. Changed and added files:

| file | change |
|---|---|
| `multiagent/custom_scenarios/navigation_joint_graph.py` | the joint-capture scenario (exact-$k$ rule, potential shaping, radius observations) |
| `onpolicy/scripts/train_mpe.py` | the task flags and the three CT-SDE switches |
| `onpolicy/runner/shared/graph_mpe_runner.py` | group-commit exploration and behaviour-weight bookkeeping |
| `onpolicy/algorithms/graph_mappo.py` | optimistic advantages and the off-policy weight |
| `onpolicy/runner/shared/base_runner.py` | metric names matching the MAT fork |
| `onpolicy/utils/ctsde_switches.py` | **new** — the switch implementations, ported verbatim from the MAT runner |
| `mac_adapter/` | **new** — runs InforMARL on the MAC envs; the graph is each step's communication mask, built exactly as CommNet/TarMAC build theirs |

Everything else under `InforMARL/` is upstream.

> Nayak, Choi, Ding, Dolan, Gopalakrishnan, Balakrishnan.
> *Scalable Multi-Agent Reinforcement Learning through Intelligent Information
> Aggregation.* arXiv:2211.02127, 2022. https://github.com/nsidn98/InforMARL

```bibtex
@article{nayak2022scalable,
  title={Scalable Multi-Agent Reinforcement Learning through Intelligent
         Information Aggregation},
  author={Nayak, Siddharth and Choi, Kenneth and Ding, Wenqi and Dolan, Sydney
          and Gopalakrishnan, Karthik and Balakrishnan, Hamsa},
  journal={arXiv preprint arXiv:2211.02127},
  year={2022}
}
```

---

## Layout

```
run_experiment.py                compose and run one configuration (both families)
configs/                          task, method and shared settings
  mac.py                          the whole MAC protocol
mac/                              the MAC environments (+ Pickup Consensus)
  envs/                           pickup_consensus, aerial_ordering,
                                  charging_rendezvous, team_partition
  presets.py                      the paper configuration of each env
  reference_policies.py           scripted reference levels
Multi-Agent-Transformer_ctsde/   CT-SDE fork of MAT (SDec-MAT, MAT-Dec, MAPPO)
  mat/algorithms/                 the communication-constrained model
  mat/envs/                       MAT-side env wrappers (joint_nav + the MAC envs)
  mat/scripts/train/              train_joint_nav.py, train_<mac env>.py
epymarl/                          EPyMARL fork: MAPPO, CommNet, TarMAC, QMIX (MAC)
InforMARL/                        graph MAPPO baseline
  mac_adapter/                    InforMARL on the MAC envs
scripts/                          figures, evaluation, fairness audit
  generalization/                 MAC zero-shot generalisation tests
docs/                             architecture, environment spec, run manifest
results/                          figures, tables and rollouts, per task
```

## Install

Python 3.8. The exact package set every reported run used is pinned in
`requirements.txt`:

```bash
conda create -n ctsde python=3.8.10 && conda activate ctsde
pip install --no-deps -r requirements.txt
export PYTHONPATH="$PWD/Multi-Agent-Transformer_ctsde:$PWD/InforMARL"
```

## Run

The three learning switches — group-commit exploration, the off-policy weight
and optimistic advantages — are **off by default**, so the base model needs no
switch flags at all:

```bash
python run_experiment.py --task e4 --arm sdec_mat --seed 1 --base
```

A reported configuration is one command; the task supplies the environment rule
and the switch block, the arm supplies only the architecture:

```bash
python run_experiment.py --task e4   --arm all --seed 1 2 3     # 4 methods x 3 seeds
python run_experiment.py --task e2   --arm sdec_mat --seed 1
python run_experiment.py --task e2p  --arm mappo    --seed 2
python run_experiment.py --task e4m1 --arm informarl --seed 3
```

| `--task` | rule | miss penalty | steps |
|---|---|---|---|
| `e2` | at least $k$ | $-1$ | 2M |
| `e2p` | at least $k$ | $-5$ | 2M |
| `e4` | exactly $k$ | $-5$ | 4M |
| `e4m1` | exactly $k$ | $-1$ | 4M |
| `e4m1_floor005` | the behaviour-weight-floor ablation of `e4m1` | | 4M |

`--arm` is `sdec_mat`, `mat_dec`, `mappo`, `informarl`, or `all`.
`--dry-run` prints the command instead of running it, `--steps` and `--threads`
shorten a run for a smoke test.

The configuration lives in `configs/`: `common.py` is shared by every task and
method, `tasks.py` holds each task's rule and switch block, `arms.py` the four
methods. Everything outside `arms.py` is identical across methods within a task,
which is what makes the comparison controlled.

## Weights & Biases

Logging is **off** unless an entity is given:

```bash
export WANDB_ENTITY=<your entity>
export WANDB_PROJECT=<your project>
```

`run_experiment.py` passes them through, and the plotting, evaluation and audit
tools read from the same two variables.

## Reproduce the figures and tables

```bash
python scripts/plot_jointnav_paper.py --task {e2,e2p,e4,e4m1f0} --png
python scripts/eval_jointnav_all_arms.py --task e4 --prefix e4fair --episodes 60
python scripts/audit_run_fairness.py <run> <run> ...
```

Figures land in `results/<task>/figures/`. Note that `--task e4m1f0` is the
reported E4-m1 (behaviour-weight floor 0) and writes to `results/e4m1/`, while
`--task e4m1` is the floor-0.05 ablation and writes to
`results/e4m1_floor005_ablation/`.

`scripts/audit_run_fairness.py` diffs the logged configurations of a set of runs
and reports any hyperparameter that is not identical across arms; it is how the
fairness claims in the appendix were checked. Every run behind every reported
number is listed in `docs/jointnav_runs.md`.

## Tests

```bash
cd Multi-Agent-Transformer_ctsde/mat
for t in test_joint_nav.py test_mat_comm.py test_mat_comm_pairwise.py \
         test_mat_comm_soft_critic.py test_radius_commit_groups.py; do python $t; done
cd ../../InforMARL && python test_navigation_joint_graph.py
```

These cover the parts that are easy to get subtly wrong: that an agent never
attends to a non-neighbour's observation or action, that a two-hop chain is not
treated as a clique, and that rollout and teacher-forced log-probabilities agree
so the PPO importance ratio starts at 1.

---

## MAC benchmark

The **Multi-Agent Counting benchmark (MAC)** asks a team of interchangeable
agents to get **exactly K** of its members to act, with K revealed only at run
time. Agents that see identical observations and share a policy cannot do that
without some way to break the tie between them, which is the gap SDec-MAT's
autoregressive decoder closes.

| env | task | agents | steps |
|---|---|---|---|
| **Pickup Consensus** | the bare counting problem: K of N identical drones must PICKUP in the same step | 5, K ∈ {2,3,4} | 1M |
| **Aerial Ordering** | a sensor finds a target and reads its hidden K; exactly K workers must lift it together | 1 sensor + 5 workers | 1M |
| **Charging Rendezvous** | drones arriving in waves must ration a station with fewer ports than drones | 6 drones, 2 ports | 2M |
| **Team Partition** | a team regroups, then splits across zones to match a public quota exactly | 9 agents, 3 zones | 1M |

Pickup Consensus is the controlled special case of Aerial Ordering (no grid,
sensor, comm range or wind) and isolates the counting problem; the other three
form MAC. All four live in `mac/`:

```python
import mac
env = mac.make("team_partition", seed=0)       # the paper configuration
env = mac.make("aerial_ordering", wind_prob=0)  # with an override
```

Each env is a gymnasium env (`pickup-consensus-v0`, `aerial-ordering-v0`,
`charging-rendezvous-v0`, `team-partition-v0`) that also speaks the SMAC-style
API (`get_obs`, `get_state`, `get_avail_actions`, `get_comm_mask`,
`get_env_info`) the training stacks read. `mac/presets.py` holds the exact
configuration every method was trained on; each env's module docstring
describes its observations, actions, rewards and logged metrics.

### Methods

| name in the paper | `--arm` | trained by |
|---|---|---|
| **SDec-MAT** | `sdec_mat` | MAT fork: `mat_comm`, communication-masked encoder + autoregressive decoder |
| MAT-Dec | `mat_dec` | MAT fork: `mat_dec`, shared decentralised MLP actor |
| MAPPO | `mappo` | `epymarl/` |
| CommNet | `commnet` | `epymarl/` |
| TarMAC | `tarmac` | `epymarl/` |
| QMIX | `qmix` | `epymarl/` |
| InforMARL | `informarl` | `InforMARL/mac_adapter/` |

All three code bases build the env from `mac/presets.py`, so within a task every
method plays the same game. Evaluation **samples** from the policy for every
method except QMIX (greedy): with argmax, agents that see the same observation
all choose the same action.

### Protocol

| task | SDec-MAT | MAT-Dec | MAPPO / CommNet / TarMAC / QMIX | InforMARL |
|---|---|---|---|---|
| Pickup Consensus | seeds 42–44 | w/ and w/o ID, 42–44 | w/ and w/o ID, 41–45 | w/ and w/o ID, 42–44 |
| Aerial Ordering | 42–44 | w/ ID, 42–44 | w/ ID, 41–45 | w/ ID, 42–44 |
| Charging Rendezvous | 42–44 | w/ ID, 42–44 | w/ ID, 42–44 | w/ ID, 42–44 |
| Team Partition | 42–44 | w/ ID, 42–44 | w/ ID, 42–44 | w/ ID, 42–44 |

"ID" is a one-hot agent index appended to every observation. On Pickup
Consensus the baselines are reported both ways: without an id a shared policy
cannot break the symmetry and stays at the `p = K/N` ceiling. The MAC tasks then
give every baseline ids. SDec-MAT never receives one. Every per-method setting
is in `configs/mac.py`.

### Hyperparameters

The values every MAC run uses. They live in `configs/mac.py` (SDec-MAT,
MAT-Dec, InforMARL), `epymarl/config/` (MAPPO, CommNet, TarMAC, QMIX) and
`mac/presets.py` (environments); `python run_experiment.py --task <task> --arm
<arm> --dry-run` prints the exact command for any run. All methods use Adam.

**SDec-MAT and MAT-Dec** (MAT fork)

| | |
|---|---|
| learning rate | 3e-4 (Adam, eps 1e-5) |
| PPO epochs / mini-batches / clip | 4 / 1 / 0.05 |
| entropy coefficient | 0.1 |
| value loss | coefficient 1, Huber (δ = 10), value normalisation, clipped |
| γ / GAE λ | 0.99 / 0.95 |
| gradient-norm clip | 10 |
| parallel envs / rollout length | 32 / one episode (`time_limit`) |
| transformer | 1 block, embedding 64, 1 head |
| evaluation | 100 episodes, actions sampled from the policy |
| SDec-MAT only | communication mask from the env's per-agent bits, soft group bias in the critic's attention, decoder attention gain 0.01, no observation skip |
| MAT-Dec only | one shared MLP actor over each agent's own observation |
| Pickup Consensus only | no LayerNorm on the raw observation (both methods) |

**MAPPO, CommNet, TarMAC** (EPyMARL)

| | |
|---|---|
| learning rate | 3e-4 |
| PPO epochs / clip | 4 / 0.2 |
| entropy coefficient | 0.1 |
| agent | GRU, hidden 128; the last action is not part of the input |
| critic | centralised V, n-step returns (n = 5), target update τ = 0.01 |
| γ | 0.99 |
| gradient-norm clip | 0.5 |
| parallel envs / batch | 10 / 10 episodes |
| rewards | standardised |
| evaluation | actions sampled from the policy |

**QMIX** (EPyMARL)

| | |
|---|---|
| learning rate | 3e-4 |
| agent | GRU, hidden 128 |
| exploration | ε-greedy, 1.0 → 0.05 over the anneal steps below |
| replay buffer / batch | 5000 / 32 episodes |
| target network | updated every 200 episodes, double Q |
| mixer | embedding 32, hypernetwork 2 layers of 64 |
| γ | 0.99 |
| gradient-norm clip | 10 |
| parallel envs | 10 |
| rewards | standardised |
| evaluation | greedy (ε = 0) |

**InforMARL** (rMAPPO with a graph encoder)

| | |
|---|---|
| actor / critic learning rate | 3e-4 / 3e-4 |
| PPO epochs / mini-batches / clip | 4 / 1 / 0.2 |
| entropy coefficient | 0.1 |
| value loss | coefficient 1, Huber (δ = 10), value normalisation |
| γ / GAE λ | 0.99 / 0.95 |
| gradient-norm clip | 0.5 |
| networks | hidden 128, recurrent (GRU), chunk length 10 |
| graph encoder | 2 layers, hidden 16, 3 heads; the graph is the env's communication mask |
| parallel envs | 10 |
| evaluation | actions sampled from the policy |

**Per task**

| | Pickup Consensus | Aerial Ordering | Charging Rendezvous | Team Partition |
|---|---|---|---|---|
| environment steps | 1M | 1M | 2M | 1M |
| episode length | 10 | 100 | 100 | 20 |
| QMIX ε anneal (w/o ID / w/ ID) | 50k / 50k | 50k / 500k | 1M / 1M | 1M / 1M |
| EPyMARL eval episodes / interval | 100 / 20k | 20 / 50k | 20 / 50k | 400 / 20k |

### Run

```bash
python run_experiment.py --task team_partition --arm all            # every method, every reported seed
python run_experiment.py --task pickup_consensus --arm sdec_mat --seed 42
python run_experiment.py --task pickup_consensus --arm qmix --agent-id off
python run_experiment.py --task charging_rendezvous --arm all --dry-run
```

Without `--seed` / `--agent-id` a run covers exactly the seeds and agent-id
conditions in the table above; `--agent-id on|off|both` overrides the latter.
`--steps` and `--threads` shorten a run for a smoke test (`--threads` does not
apply to the EPyMARL baselines, whose batch size is tied to their 10 parallel
envs).

Outputs (TensorBoard):

| methods | location |
|---|---|
| SDec-MAT, MAT-Dec | `Multi-Agent-Transformer_ctsde/mat/scripts/results/<Env>/<scenario>/<algorithm>/<arm>_s<seed>[_aid]/` |
| EPyMARL baselines | `results/epymarl/tb_logs/<alg>[_aid]_seed<seed>_<gym id>_<time>/` |
| InforMARL | `results/informarl/tb_logs/informarl[_aid]_seed<seed>_<gym id>_<time>/` |

### Figures

```bash
python scripts/plot_mac.py                  # every MAC figure
python scripts/plot_mac.py --task team_partition
python -m mac.reference_policies            # the scripted reference levels
```

`plot_mac.py` finds the runs above by name and writes the paper's return and
success-rate figures (one per task and agent-id condition), the Pickup
Consensus 1×2 figure and the three-panel MAC figure to `results/figures/mac/`.
It needs **matplotlib ≥ 3.7** (`pip install "matplotlib>=3.7,<3.8"`), newer
than the version pinned for training. Each curve is the mean over seeds with a
min/max band; the dotted lines are scripted policies rolled out on the real env
(`mac/reference_policies.py`):

| task | upper level | lower level |
|---|---|---|
| Pickup Consensus | exactly K PICKUP at t=0, +9.70 | each agent PICKUPs w.p. K/N, −2.80 |
| Aerial Ordering | Perfect Solution, +18.78 | Navigate-and-Wait, −10.36 |
| Charging Rendezvous | Perfect Solution, +159.96 | Navigate-and-Wait, −106.06 |
| Team Partition | Perfect Solution, +55.96 | Navigate-and-Wait, −6.75 |

### Zero-shot generalisation

Trained policies are frozen and evaluated greedily on teams they never saw.
Point the scripts at checkpoints with `--checkpoint` (a run directory, step
directory or weights file), or at runs trained with `run_experiment.py` with
`--seeds`.

**Pickup Consensus** (`scripts/generalization/pickup_consensus.py`), for any method:

```bash
python scripts/generalization/pickup_consensus.py --method sdec_mat --seeds 42 43 44
python scripts/generalization/pickup_consensus.py --method mappo --seeds 41 42 43 44 45
```

| condition | what changes |
|---|---|
| `base` | nothing: the trained 5-agent team |
| `T1`, `T1b` | 1 or 2 of the 5 agents drop out; the survivors play a 4- or 3-agent episode |
| `T2` | 10 agents split at random into two teams of 5 that cannot hear each other, each with its own K; success means both teams are right |
| sizes | SDec-MAT only: the same weights on teams of 3–10 agents |

SDec-MAT has no agent ids: a lost agent is simply absent, and a split is one
forward pass whose communication graph has two components. The baselines,
which only solve the task with ids, keep their original ids when agents drop
out; for the split each id is issued twice. A checkpoint whose unperturbed
success is below `--min-base-success` (0.99) never solved the task and is left
out. The table is the mean ± std across checkpoints, with success broken down
by K.

**Aerial Ordering, several targets at once**
(`scripts/generalization/aerial_multitarget.py`, SDec-MAT):

```bash
python scripts/generalization/aerial_multitarget.py --seed 42
```

A policy trained on one sensor, five workers and one target runs T sensors, T
targets and N workers in one arena (default 2×10 and 3×15), every worker
assigned to one target (`mac/envs/aerial_ordering_multitarget.py`).
Communication is physical, so a team can end up in another team's decode
chain. Three conditions isolate why success drops: uneven teams with shared
chains, then chains separated by team, then every team exactly the trained
1 sensor + 5 workers. Reported: simultaneous success (all targets lifted in one
episode), per-target success, return, and the success of episodes in which a
team committed while its chain held another team.

---

## Documentation

| file | contents |
|---|---|
| `docs/appendix_jointnav_short.tex` | protocol, hyperparameters and results tables |
| `docs/jointnav_runs.md` | every run behind every reported figure, and which code path those runs exercise |
| `results/README.md` | what each figure is, and how the bands differ from the tables |

The model itself is described in the paper; `mat/algorithms/mat/algorithm/mat_comm_transformer.py`
and `mat/algorithms/utils/mat_comm_groups.py` carry the implementation notes.

## Licence

InforMARL is MIT (`InforMARL/LICENSE`). EPyMARL is Apache-2.0
(`epymarl/LICENSE`; see `epymarl/README.md` for what was changed). MAT is
distributed under the licence of its upstream repository; see
https://github.com/PKU-MARL/Multi-Agent-Transformer.
