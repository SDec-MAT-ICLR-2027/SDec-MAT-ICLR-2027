"""
The observation-sharing DE control must differ from CT-SDE in exactly one way.

In the radius env (comm_mode=agent_radius), for a pair of neighbours i < j
(i decodes earlier in j's chain):

  CT-SDE (decoder_self_only=0)
    1. j's action distribution DEPENDS on i's action        (action conditioning)
    2. j's distribution depends on i's observation           (obs sharing)
    3. j's distribution does NOT depend on a non-neighbour's observation

  DE control (decoder_self_only=1)
    1. j's action distribution does NOT depend on i's action (no conditioning)
    2. j's distribution STILL depends on i's observation     (obs sharing kept)
    3. j's distribution does NOT depend on a non-neighbour's observation

Checks use teacher-forced evaluate_actions on O(1) random weights: allowed
paths give clearly nonzero effects, masked paths must be exactly zero.

Run:  python -m mat.test_decoder_self_only
"""
import sys
from types import SimpleNamespace

import numpy as np
import torch

from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.envs.joint_nav import JointNavMATEnv


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(cond)


def args(self_only):
    return SimpleNamespace(
        algorithm_name="mat_comm", experiment_name="t", seed=1, cuda=False,
        n_agents=6, num_targets=3, k_req=3, episode_length=50,
        capture_radius=0.3, group_radius=0.3, capture_rew=10.0, fail_rew=-1.0,
        assembly_rew=0.05, dist_rew_scale=1.0, potential_shaping=1,
        shaping_gamma=1.0, collision_rew=0.0, world_size=2.0,
        num_obstacles=0, max_speed=2.0, obs_mode="radius", sense_radius=0.8,
        comm_mode="agent_radius", obs_ready_count=False,
        n_block=1, n_embd=64, n_head=1, encode_state=False, dec_actor=False,
        share_actor=False, comm_mask_kind="pairwise",
        encoder_critic_attn="soft_group_bias", decoder_attn_gain=0.01,
        commitment=0, group_token=0, use_obs_skip=0, group_latent=0,
        decoder_self_only=self_only,
        lr=5e-4, opti_eps=1e-5, weight_decay=0.0,
        use_policy_active_masks=True, hidden_size=64, recurrent_N=1,
        gain=0.01, use_orthogonal=True, use_feature_normalization=True,
        use_ReLU=True, stacked_frames=1, use_naive_recurrent_policy=False,
        use_recurrent_policy=False, data_chunk_length=10, use_popart=False,
        use_valuenorm=True, use_value_active_masks=True, value_loss_coef=1.0,
        entropy_coef=0.01, clip_param=0.2, ppo_epoch=5, num_mini_batch=1,
        huber_delta=10.0, use_max_grad_norm=True, max_grad_norm=10.0,
        use_clipped_value_loss=True, use_huber_loss=True, use_gae=True,
        gamma=0.99, gae_lambda=0.95, use_proper_time_limits=False,
        env_name="joint_nav",
    )


def logp_j(pol, share, obs, acts, j):
    n = len(acts)
    z = np.zeros((n, 1, 64), dtype=np.float32)
    m = np.ones((n, 1), dtype=np.float32)
    av = np.ones((n, 6), dtype=np.float32)
    with torch.no_grad():
        out = pol.evaluate_actions(share, obs, z, z, acts.reshape(-1, 1), m, av)
    return float(out[1].reshape(-1)[j])


def find_state(env):
    """A state with neighbours i<j and some agent k not adjacent to j."""
    for s in range(500):
        env.seed(s)
        obs, share, _ = env.reset()
        adj = env.get_comm_mask()
        for j in range(1, 6):
            for i in range(j):
                if adj[i, j]:
                    far = [k for k in range(6) if k != j and not adj[k, j]]
                    if far:
                        return obs, share, i, j, far[0]
    raise RuntimeError("no suitable state")


def run(self_only):
    torch.manual_seed(0)
    a = args(self_only)
    env = JointNavMATEnv(a)
    pol = TransformerPolicy(a, env.observation_space[0],
                            env.share_observation_space[0],
                            env.action_space[0], 6)
    # Init gains (0.01) make cross-agent effects fall below float32 resolution,
    # which would make "depends" checks vacuous. Mask semantics do not depend
    # on the weights, so use O(1) random weights: allowed paths become large,
    # masked paths must stay exactly zero.
    net = getattr(pol, "transformer", pol)
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.3)
    obs, share, i, j, k = find_state(env)
    acts = np.zeros(6, dtype=np.int64)
    base = logp_j(pol, share, obs, acts, j)

    a2 = acts.copy()
    a2[i] = 5
    d_act = abs(logp_j(pol, share, obs, a2, j) - base)

    # perturb ONE feature (own position x): a constant shift of the whole
    # vector would be erased by the encoder's input LayerNorm
    o2 = obs.copy()
    o2[i, 2] += 0.5
    d_nb = abs(logp_j(pol, share, o2, acts, j) - base)

    o3 = obs.copy()
    o3[k, 2] += 0.5
    d_far = abs(logp_j(pol, share, o3, acts, j) - base)
    return i, j, k, d_act, d_nb, d_far


def main():
    ok = True
    for self_only, name in ((0, "CT-SDE"), (1, "DE control")):
        i, j, k, d_act, d_nb, d_far = run(self_only)
        print("== %s  (neighbours %d->%d, non-neighbour %d) ==" % (name, i, j, k))
        if self_only:
            ok &= check("j ignores i's ACTION", d_act < 1e-9, "%.2e" % d_act)
        else:
            ok &= check("j depends on i's ACTION", d_act > 1e-9, "%.2e" % d_act)
        ok &= check("j depends on neighbour i's OBSERVATION", d_nb > 1e-9,
                    "%.2e" % d_nb)
        ok &= check("j ignores non-neighbour k's observation", d_far < 1e-9,
                    "%.2e" % d_far)
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
