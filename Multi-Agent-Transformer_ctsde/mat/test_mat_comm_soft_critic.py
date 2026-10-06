"""Dual encoder attention: actor hard, critic soft group_bias.

Default hard must keep existing isolation. soft_group_bias must let V see
out-of-group obs while logits stay isolated. From repo root:

    python -u Multi-Agent-Transformer_ctsde/mat/test_mat_comm_soft_critic.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath("Multi-Agent-Transformer_ctsde"))

import numpy as np
import torch

from mat.algorithms.mat.algorithm.mat_comm_transformer import CommMultiAgentTransformer
from mat.algorithms.utils.mat_comm_act import decode_logits_per_agent

torch.manual_seed(0)
np.random.seed(0)

N, OBS, ACT, SD = 6, 5, 3, 24
BITS = [1., 1., 1., 0., 0., 0.]  # {0,1,2} {3} {4} {5}

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
errs = []


def check(c, m):
    print("  %s: %s" % (PASS if c else FAIL, m))
    if not c:
        errs.append(m)


def make_state(B=1):
    s = np.random.rand(B, N, SD).astype(np.float32)
    s[:, :, -N:] = np.array(BITS, dtype=np.float32)
    return s


def energize(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear):
            torch.nn.init.orthogonal_(mod.weight, gain=1.0)
            if mod.bias is not None:
                torch.nn.init.normal_(mod.bias, std=0.1)
    return m


def logits_and_values(m, state, obs):
    with torch.no_grad():
        st = torch.tensor(state)
        ob = torch.tensor(obs)
        gmask, cmask = m._masks(st)
        v, rep = m.encoder(st, ob, gmask)
        a = torch.zeros((1, N), dtype=torch.long)
        lg = decode_logits_per_agent(m.decoder, rep, ob, a, gmask, ACT, m.tpdv)
    return lg[0].numpy(), v[0].numpy(), rep


def copy_shared(src, dst):
    sd = src.state_dict()
    dst.load_state_dict({k: v for k, v in sd.items() if k in dst.state_dict()},
                        strict=False)


print("\n=== 0. refuse unknown mode ===")
try:
    CommMultiAgentTransformer(
        SD, OBS, ACT, N, n_block=1, n_embd=32, n_head=1,
        encoder_critic_attn="full",
    )
    check(False, "unknown encoder_critic_attn must raise")
except ValueError as e:
    check("encoder_critic_attn" in str(e), "unknown mode raises: %s" % e)


print("\n=== 1. default hard: value isolation still holds ===")
hard = energize(CommMultiAgentTransformer(
    SD, OBS, ACT, N, n_block=2, n_embd=32, n_head=2,
))
hard.eval()
check(hard.encoder_critic_attn == "hard", "default encoder_critic_attn=hard")
check(not hasattr(hard.encoder, "group_bias"), "hard path has no group_bias param")

state = make_state()
obs = np.random.rand(1, N, OBS).astype(np.float32)
base_lg, base_v, _ = logits_and_values(hard, state, obs)
obs_pert = obs.copy()
obs_pert[0, 4, :] = np.random.rand(OBS)
pert_lg, pert_v, _ = logits_and_values(hard, state, obs_pert)
d_lg = np.abs(base_lg - pert_lg).max(axis=1)
d_v = np.abs(base_v - pert_v).ravel()
for i in range(N):
    if i == 4:
        check(d_lg[i] > 1e-6, "hard: agent %d (perturbed) DOES change" % i)
    else:
        check(d_lg[i] < 1e-6, "hard: agent %d logits isolated" % i)
        check(d_v[i] < 1e-6, "hard: agent %d value isolated" % i)


print("\n=== 2. soft_group_bias: logits isolated, values are not ===")
soft = energize(CommMultiAgentTransformer(
    SD, OBS, ACT, N, n_block=2, n_embd=32, n_head=2,
    encoder_critic_attn="soft_group_bias",
))
soft.eval()
check(hasattr(soft.encoder, "group_bias"), "soft path has group_bias")
check(float(soft.encoder.group_bias.item()) == 0.0, "group_bias init 0")

base_lg, base_v, _ = logits_and_values(soft, state, obs)
pert_lg, pert_v, _ = logits_and_values(soft, state, obs_pert)
d_lg = np.abs(base_lg - pert_lg).max(axis=1)
d_v = np.abs(base_v - pert_v).ravel()
print("  per-agent max |dlogit| = %s" % np.round(d_lg, 8))
print("  per-agent max |dvalue| = %s" % np.round(d_v, 8))
for i in range(N):
    if i == 4:
        check(d_lg[i] > 1e-6, "soft: agent %d (perturbed) logits change" % i)
        check(d_v[i] > 1e-6, "soft: agent %d (perturbed) value changes" % i)
    else:
        check(d_lg[i] < 1e-6, "soft: agent %d logits still isolated" % i)
        check(d_v[i] > 1e-6, "soft: agent %d value SEES agent 4" % i)


print("\n=== 3. same weights: obs_rep matches hard, V does not ===")
soft2 = CommMultiAgentTransformer(
    SD, OBS, ACT, N, n_block=2, n_embd=32, n_head=2,
    encoder_critic_attn="soft_group_bias",
)
copy_shared(hard, soft2)
soft2.eval()
st = torch.tensor(state)
ob = torch.tensor(obs)
gmask, _ = hard._masks(st)
with torch.no_grad():
    v_h, r_h = hard.encoder(st, ob, gmask)
    v_s, r_s = soft2.encoder(st, ob, gmask)
check(torch.allclose(r_h, r_s, atol=1e-6), "actor obs_rep identical (shared hard pass)")
check(not torch.allclose(v_h, v_s, atol=1e-5), "V differs (soft critic vs hard)")


print("\n=== 4. action isolation still holds on soft model ===")
def logits_given_actions(acts):
    with torch.no_grad():
        st_ = torch.tensor(state)
        ob_ = torch.tensor(obs)
        gmask, cmask = soft._masks(st_)
        _, rep = soft.encoder(st_, ob_, gmask)
        a = torch.tensor([acts], dtype=torch.long)
        return decode_logits_per_agent(
            soft.decoder, rep, ob_, a, gmask, ACT, soft.tpdv
        )[0].numpy()


l0 = logits_given_actions([0, 0, 0, 0, 0, 0])
l_out = logits_given_actions([0, 0, 0, 2, 0, 0])
check(np.abs(l0[4] - l_out[4]).max() < 1e-6, "agent 4 ignores agent 3's action")


print("\n=== 5. rollout vs teacher-forced values agree ===")
B = 4
stn = make_state(B)
obn = np.random.rand(B, N, OBS).astype(np.float32)
av = np.ones((B, N, ACT), dtype=np.float32)
a, alog, v = soft.get_actions(stn, obn, av, deterministic=False)
alog2, v2, ent = soft(stn, obn, a.detach().numpy(), av)
check(np.abs(alog.detach().numpy() - alog2.detach().numpy()).max() < 1e-5,
      "log-probs agree")
check(np.abs(v.detach().numpy() - v2.detach().numpy()).max() < 1e-6,
      "soft V agrees between get_actions and forward")
v3 = soft.get_values(stn, obn)
check(np.abs(v.detach().numpy() - v3.detach().numpy()).max() < 1e-6,
      "get_values matches get_actions V")

print("\n" + ("ALL PASS" if not errs else "%d FAILURE(S): %s" % (len(errs), errs)))
sys.exit(1 if errs else 0)
