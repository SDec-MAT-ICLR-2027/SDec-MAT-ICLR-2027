"""Paired evaluation of joint_nav checkpoints on held-out eval seeds.

Every arm is rolled out on the SAME eval initial states (seed 50000+i, the same
offset make_eval_env uses), so per-episode results can be compared pairwise.
No exploration commitment is applied: this is the deployable policy. Both
greedy (argmax) and sampled rollouts are reported, because for CT-SDE the
sampled policy was measured to capture more than the greedy one.

  python -u eval_joint_nav.py [n_episodes]
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Multi-Agent-Transformer_ctsde"))

from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.envs.joint_nav.joint_nav_env import JointNavMATEnv

R = os.path.join(ROOT, "Multi-Agent-Transformer_ctsde", "mat", "scripts",
                 "results", "joint_nav", "joint_nav_6a3t_k3")
CKPTS = [
    ("mappo_rand_s1", "mappo",
     "mappo/var_mappo_rand_s1/wandb/run-20260914_084227-9v8j9hgr/files/transformer_311.pt"),
    ("mappo_rand_s2", "mappo",
     "mappo/var_mappo_rand_s2/wandb/run-20260914_092251-iqkgs3pu/files/transformer_311.pt"),
    ("plain_rand_s1", "plain",
     "mat_comm_soft/var_plain_rand_s1/wandb/run-20260914_084905-zaxxpdkr/files/transformer_311.pt"),
    ("plain_rand_s2", "plain",
     "mat_comm_soft/var_plain_rand_s2/wandb/run-20260914_092942-9hj4cm3x/files/transformer_311.pt"),
    ("latent_commit_s1", "latent",
     "mat_comm_soft/var_commit_0.2_nohold/wandb/run-20260912_113926-b2f0pswi/files/transformer_311.pt"),
    ("latent_commit_s2", "latent",
     "mat_comm_soft/var_latent_commit_s2/wandb/run-20260914_091108-9a5r2sie/files/transformer_311.pt"),
]


def build_args(arm):
    a = SimpleNamespace(
        experiment_name="eval_joint_nav", seed=1, cuda=True,
        n_training_threads=1, n_rollout_threads=1, n_eval_rollout_threads=1,
        n_agents=6, num_targets=3, k_req=3, episode_length=50,
        capture_radius=0.3, group_radius=0.3, capture_rew=10.0, fail_rew=-1.0,
        assembly_rew=0.05, dist_rew_scale=1.0, potential_shaping=1,
        shaping_gamma=1.0, collision_rew=0.0, world_size=2.0,
        num_obstacles=0, max_speed=2.0, obs_mode="shared",
        obs_ready_count=False,
        n_block=1, n_embd=64, n_head=1, encode_state=False,
        comm_mask_kind="pairwise", encoder_critic_attn="soft_group_bias",
        decoder_attn_gain=0.01, commitment=0, group_token=0, use_obs_skip=0,
        group_token_entropy_coef=None, n_commit_targets=0, fixed_groups=None,
        fixed_groups_assign="id", z_loss_ready_balance=0,
        group_latent=0, n_latent=6, latent_mi_coef=0.5,
        latent_entropy_coef=0.1, latent_cond_std=1.0, latent_bias_std=1.0,
        latent_span_init=1, latent_bias_norm=1, latent_bias_scale=0.3,
        latent_bias_floor=1.0, latent_hold=0,
        lr=5e-4, opti_eps=1e-5, weight_decay=0.0,
        use_policy_active_masks=True, hidden_size=64, recurrent_N=1, gain=0.01,
        use_orthogonal=True, use_feature_normalization=True, use_ReLU=True,
        stacked_frames=1, use_naive_recurrent_policy=False,
        use_recurrent_policy=False, data_chunk_length=10, use_popart=False,
        use_valuenorm=True, use_value_active_masks=True, value_loss_coef=1.0,
        entropy_coef=0.01, clip_param=0.2, ppo_epoch=5, num_mini_batch=1,
        huber_delta=10.0, use_max_grad_norm=True, max_grad_norm=10.0,
        use_clipped_value_loss=True, use_huber_loss=True, use_gae=True,
        gamma=0.99, gae_lambda=0.95, use_proper_time_limits=False,
        env_name="joint_nav",
    )
    if arm == "mappo":
        a.algorithm_name = "mappo"
        a.dec_actor = True
        a.share_actor = True
    else:
        a.algorithm_name = "mat_comm"
        a.dec_actor = False
        a.share_actor = False
        a.group_latent = 1 if arm == "latent" else 0
    return a


def load(arm, path, device):
    a = build_args(arm)
    env = JointNavMATEnv(a)
    pol = TransformerPolicy(a, env.observation_space[0],
                            env.share_observation_space[0],
                            env.action_space[0], env.n_agents, device=device)
    pol.restore(path)
    pol.eval()
    return a, env, pol


def run_episode(env, pol, seed, greedy):
    env.seed(seed)
    torch.manual_seed(seed)
    n = env.n_agents
    rnn = np.zeros((n, 1, 64), dtype=np.float32)
    mask = np.ones((n, 1), dtype=np.float32)
    obs, share, avail = env.reset()
    ret, clear_t = 0.0, None
    for t in range(env.episode_limit):
        with torch.no_grad():
            out = pol.get_actions(np.asarray(share), np.asarray(obs), rnn, rnn,
                                  mask, np.asarray(avail), deterministic=greedy)
        acts = out[1].detach().cpu().numpy().reshape(-1)
        obs, share, rew, dones, info, avail = env.step(acts)
        ret += float(rew[0][0])
        if info[0]["won"] and clear_t is None:
            clear_t = t + 1
        if bool(np.all(dones)):
            break
    i0 = info[0]
    return dict(caps=i0["joint_captures"], unders=i0["undershoots"],
                acts=i0["capture_actions"], won=float(i0["won"]), ret=ret,
                clear_t=clear_t)


def main():
    n_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seeds = [50000 + i for i in range(n_eps)]
    per = {}
    for name, arm, rel in CKPTS:
        _, env, pol = load(arm, os.path.join(R, rel), device)
        for mode in ("greedy", "sampled"):
            rows = [run_episode(env, pol, s, mode == "greedy") for s in seeds]
            per[(name, mode)] = rows
            print("done %-17s %s" % (name, mode), flush=True)

    def stat(rows, k):
        v = np.array([r[k] for r in rows], dtype=float)
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

    for mode in ("greedy", "sampled"):
        print("\n" + "=" * 96)
        print("EVAL (%s), %d paired eval episodes, no exploration commitment"
              % (mode, n_eps))
        print("%-17s %12s %12s %12s %12s %12s %9s" % (
            "run", "captures", "win_rate", "undershoots", "capture_acts",
            "return", "t_clear"))
        print("-" * 96)
        for name, _, _ in CKPTS:
            rows = per[(name, mode)]
            ts = [r["clear_t"] for r in rows if r["clear_t"] is not None]
            cells = []
            for k in ("caps", "won", "unders", "acts", "ret"):
                m, se = stat(rows, k)
                cells.append("%6.2f+-%-4.2f" % (m, se))
            print("%-17s %s %9s" % (name, " ".join(cells),
                                   "%.1f" % np.mean(ts) if ts else "-"))

        print("\n  paired, same %d eval states: CT-SDE vs MAPPO (captures)" % n_eps)
        for ct in ("plain_rand_s1", "plain_rand_s2",
                   "latent_commit_s1", "latent_commit_s2"):
            for mp in ("mappo_rand_s1", "mappo_rand_s2"):
                a = np.array([r["caps"] for r in per[(ct, mode)]])
                b = np.array([r["caps"] for r in per[(mp, mode)]])
                print("    %-17s vs %-14s  win %3d  tie %3d  loss %3d   "
                      "mean diff %+.2f" % (ct, mp, (a > b).sum(), (a == b).sum(),
                                           (a < b).sum(), (a - b).mean()))
    print("=" * 96)


if __name__ == "__main__":
    main()
