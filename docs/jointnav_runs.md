# Joint-Action Navigation: runs behind every reported figure and table

W&B project (set `WANDB_ENTITY` / `WANDB_PROJECT` to point the tools at it). Five sets of 12 runs (4 methods x 3 seeds),
all finished. Within each set every hyperparameter is identical across the four
methods — verified by diffing the full logged configs, not the launch scripts
(`scripts/audit_run_fairness.py`).

Method names map to W&B run names as:

| paper | W&B arm | `algorithm_name` |
|---|---|---|
| SDec-MAT | `ctsde` | `mat_comm` (`arm_name=mat_comm_soft`) |
| MAT-Dec | `dec` | `mat_comm` (`arm_name=mat_comm_dec`) |
| MAPPO | `mappo` | `mappo` + `--dec_actor` |
| InforMARL | `informarl` | `mappo` (InforMARL codebase, graph MAPPO) |

Reproduce figures and tables with:

```
python scripts/plot_jointnav_paper.py --task {e2,e2p,e4,e4m1,e4m1f0} --png
```

---

## E2 — at-least-$k$, miss $-1$, 2M steps  (prefix `e2fair`, seeds 1/2/3)

```
mat_comm_soft_e2fair_ctsde_2m_s1_k3_cr10_fr-1_radius_ar0.05_seed1
mat_comm_soft_e2fair_ctsde_2m_s2_k3_cr10_fr-1_radius_ar0.05_seed2
mat_comm_soft_e2fair_ctsde_2m_s3_k3_cr10_fr-1_radius_ar0.05_seed3
mat_comm_dec_e2fair_dec_2m_s1_k3_cr10_fr-1_radius_ar0.05_seed1
mat_comm_dec_e2fair_dec_2m_s2_k3_cr10_fr-1_radius_ar0.05_seed2
mat_comm_dec_e2fair_dec_2m_s3_k3_cr10_fr-1_radius_ar0.05_seed3
mappo_e2fair_mappo_2m_s1_k3_cr10_fr-1_radius_ar0.05_seed1
mappo_e2fair_mappo_2m_s2_k3_cr10_fr-1_radius_ar0.05_seed2
mappo_e2fair_mappo_2m_s3_k3_cr10_fr-1_radius_ar0.05_seed3
mappo_e2fair_informarl_2m_s1_seed1
mappo_e2fair_informarl_2m_s2_seed2
mappo_e2fair_informarl_2m_s3_seed3
```

Figures: `results/viz/e2/paper/`. Overkills and false captures are undefined
here (no `k_max`), so only success rate, joint captures and undershoots are
plotted.

## E2-p — at-least-$k$, miss $-5$, 2M steps  (prefix `e2pfair`, seeds 1/2/3)

```
mat_comm_soft_e2pfair_ctsde_2m_s{1,2,3}_k3_cr10_fr-5_radius_ar0.05_seed{1,2,3}
mat_comm_dec_e2pfair_dec_2m_s{1,2,3}_k3_cr10_fr-5_radius_ar0.05_seed{1,2,3}
mappo_e2pfair_mappo_2m_s{1,2,3}_k3_cr10_fr-5_radius_ar0.05_seed{1,2,3}
mappo_e2pfair_informarl_2m_s{1,2,3}_seed{1,2,3}
```

Figures: `results/viz/e2p/paper/`.

## E4 — exact-$k$, miss $-5$, 4M steps  (prefix `e4fair`, seeds 1/2/3)

```
mat_comm_soft_e4fair_ctsde_4m_s{1,2,3}_k3_cr10_fr-5_radius_ar0.05_seed{1,2,3}
mat_comm_dec_e4fair_dec_4m_s{1,2,3}_k3_cr10_fr-5_radius_ar0.05_seed{1,2,3}
mappo_e4fair_mappo_4m_s{1,2,3}_k3_cr10_fr-5_radius_ar0.05_seed{1,2,3}
mappo_e4fair_informarl_4m_s{1,2,3}_seed{1,2,3}
```

Figures: `results/viz/e4/paper/`. All five metrics are defined.

## E4-m1 — exact-$k$, miss $-1$, 4M steps  (prefix `e4m1f0`, seeds 2/3/4)

```
mat_comm_soft_e4m1f0_ctsde_4m_s{2,3,4}_k3_cr10_fr-1_radius_ar0.05_seed{2,3,4}
mat_comm_dec_e4m1f0_dec_4m_s{2,3,4}_k3_cr10_fr-1_radius_ar0.05_seed{2,3,4}
mappo_e4m1f0_mappo_4m_s{2,3,4}_k3_cr10_fr-1_radius_ar0.05_seed{2,3,4}
mappo_e4m1f0_informarl_4m_s{2,3,4}_seed{2,3,4}
```

Figures: `results/viz/e4m1f0/paper/`. Seeds are 2/3/4 rather than 1/2/3; the
set shares that seed set across all four methods.

---

## Ablation: behaviour-weight floor on E4-m1

`e4m1v2` is identical to `e4m1f0` in every respect except
`--behav_w_floor 0.05` versus `0`. The pair isolates the floor's effect.

```
mat_comm_soft_e4m1v2_ctsde_4m_s{2,3,4}_k3_cr10_fr-1_radius_ar0.05_seed{2,3,4}
mat_comm_dec_e4m1v2_dec_4m_s{2,3,4}_k3_cr10_fr-1_radius_ar0.05_seed{2,3,4}
mappo_e4m1v2_mappo_4m_s{2,3,4}_k3_cr10_fr-1_radius_ar0.05_seed{2,3,4}
mappo_e4m1v2_informarl_4m_s{2,3,4}_seed{2,3,4}
```

Result: the floor **hurts** SDec-MAT on this task (success 0.845 at floor 0
versus 0.000 at floor 0.05), which is why the reported E4-m1 set uses floor 0,
matching every other task. Figures: `results/viz/e4m1/paper/`.

---

## Not used in the paper

An alternative E4 set ("sub") replaced seed 3 of SDec-MAT and MAT-Dec with two
earlier runs of the same configuration:

```
mat_comm_soft_e4_ctsde_sub05_armw_opt_4m_k3_cr10_fr-5_radius_ar0.05_seed1
mat_comm_dec_e4_dec_sub05_armw_opt_4m_k3_cr10_fr-5_radius_ar0.05_seed1
```

Both are **seed-1** runs, so that set is seeds {1, 2, 1} for two methods and
{1, 2, 3} for the other two — the four rows would not share a seed set, and the
reported E4 numbers use `e4fair` seeds 1/2/3 instead. Figures from that variant,
if regenerated, land in `results/viz/e4_sub/`.

---

## Which code these runs actually exercise

Verified against the policy dispatch in
`mat/algorithms/mat/algorithm/transformer_policy.py:59-72` and the arm
normalisation in `mat/scripts/train/train_joint_nav.py`.

### Active path

| stage | SDec-MAT / MAT-Dec | MAPPO | InforMARL |
|---|---|---|---|
| entry point | `mat/scripts/train/train_joint_nav.py` | same | `onpolicy/scripts/train_mpe.py` |
| network | `mat_comm_transformer.py` (`CommMultiAgentTransformer`) | `ma_transformer.py` (`MultiAgentTransformer`, `dec_actor=True`) | `graph_MAPPOPolicy.py` |
| masks / tokens | `algorithms/utils/mat_comm_groups.py` | -- | -- |
| sampling | `algorithms/utils/mat_comm_act.py` | `algorithms/utils/transformer_act.py` | its own |
| environment | `mat/envs/joint_nav/` over `InforMARL/multiagent/custom_scenarios/navigation_joint_graph.py` | same | the scenario directly |
| rollout + switches | `mat/runner/shared/joint_nav_runner.py` | same | `onpolicy/runner/shared/graph_mpe_runner.py` |
| PPO update | `mat/algorithms/mat/mat_trainer.py` | same | `onpolicy/algorithms/graph_mappo.py` |
| buffer / GAE | `mat/utils/shared_buffer.py` | same | `onpolicy/utils/shared_buffer.py` |

`train_joint_nav.py` pins the comm arms to `algorithm_name=mat_comm`,
`comm_mask_kind=pairwise`, `encoder_critic_attn=soft_group_bias` and
`critic_global_state=1`; the only difference between SDec-MAT and MAT-Dec is
`decoder_self_only` (0 versus 1).

### Present but NOT used by any reported run

The fork carries several alternative coordination mechanisms explored during
development. None is engaged by the runs above:

| module | status |
|---|---|
| `algorithms/utils/mat_comm_group_token.py` | `train_joint_nav.py` **raises** if `--group_token != 0` |
| `algorithms/utils/mat_comm_commit.py` | **raises**: "commitment is not wired for joint_nav" |
| `algorithms/utils/mat_comm_group_latent.py` | reachable by flag; `--group_latent 0` in all 60 runs |
| `algorithms/utils/mat_comm_act_latent.py` | used only by the group-latent path |
| `algorithms/mat/algorithm/mat_comm_gru_transformer.py` | the `mat_comm_gru` arm, never selected here |

They remain because `mat_comm_transformer.py` imports them at module load and
its constructor carries the corresponding branches. Two of the three are
hard-refused for this task, so they cannot be engaged by accident.

Note that the decoder's **action tokens** are a different thing and are central:
`build_action_tokens` in `mat_comm_groups.py` feeds each decode position the
action of its nearest earlier neighbour *inside the querier's neighbourhood*,
which is what a global right-shift cannot express.

### Modified upstream MAT files

Everything else under `Multi-Agent-Transformer_ctsde/mat/` is upstream MAT. Only
six files carry real changes (the rest differ by line endings only):

| file | changed lines |
|---|---|
| `algorithms/mat/algorithm/transformer_policy.py` | 341 |
| `config.py` | 295 |
| `utils/shared_buffer.py` | 207 |
| `algorithms/mat/mat_trainer.py` | 193 |
| `algorithms/mat/algorithm/ma_transformer.py` | 8 |
| `algorithms/utils/transformer_act.py` | 5 |
