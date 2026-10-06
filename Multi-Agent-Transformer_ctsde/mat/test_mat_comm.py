"""Isolation + consistency checks for mat_comm."""
import sys, os
sys.path.append(os.path.abspath("Multi-Agent-Transformer_ctsde"))

import numpy as np
import torch

from mat.algorithms.mat.algorithm.mat_comm_transformer import CommMultiAgentTransformer
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from mat.algorithms.utils.mat_comm_groups import build_group_mask, component_ids

torch.manual_seed(0); np.random.seed(0)

N, OBS, ACT, SD = 6, 5, 3, 24          # n_workers=5 -> 6 agents, state_dim 24
BITS = [1., 1., 1., 0., 0., 0.]        # groups: {0,1,2} {3} {4} {5}

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
errs = []
def check(c, m):
    print(f"  {PASS}: {m}" if c else f"  {FAIL}: {m}")
    if not c: errs.append(m)

def make_state(B=1):
    s = np.random.rand(B, N, SD).astype(np.float32)
    s[:, :, -N:] = np.array(BITS, dtype=np.float32)   # same bits on every row
    return s

def energize(m):
    """Re-init every Linear with gain 1.0.

    Stock MAT inits attention projections with gain=0.01, so at construction the
    action-token pathway contributes ~1e-4 of the encoder residual and decays
    per block -- real, but below any sane tolerance. Isolation is a STRUCTURAL
    property that must hold for every weight setting, so the tests below run on
    non-degenerate weights: cross-group influence must be exactly 0, and
    within-group influence must be clearly non-zero.
    """
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear):
            torch.nn.init.orthogonal_(mod.weight, gain=1.0)
            if mod.bias is not None:
                torch.nn.init.normal_(mod.bias, std=0.1)
    return m

model = energize(CommMultiAgentTransformer(SD, OBS, ACT, N, n_block=2, n_embd=32, n_head=2))
model.eval()

# ---------------------------------------------------------------- grouping
print("\n=== 1. group construction ===")
gm = build_group_mask(torch.tensor([BITS]))
cid = component_ids(gm)[0].tolist()
check(cid == [0, 0, 0, 3, 4, 5], f"components = {cid} (expect [0,0,0,3,4,5])")
check(bool(gm[0, 0, 1]) and not bool(gm[0, 0, 3]), "0~1 same group, 0~3 not")
check(not bool(gm[0, 3, 4]), "two singletons are NOT grouped together")

# ------------------------------------------------------- observation leak
print("\n=== 2. observation isolation (logits) ===")
state = make_state()
obs = np.random.rand(1, N, OBS).astype(np.float32)

def logits_and_values(m, state, obs):
    """Decoder logits for every agent under a fixed zero action set, + values."""
    with torch.no_grad():
        st = torch.tensor(state); ob = torch.tensor(obs)
        if isinstance(m, CommMultiAgentTransformer):
            gmask, cmask = m._masks(st)
            v, rep = m.encoder(st, ob, gmask)
            from mat.algorithms.utils.mat_comm_act import decode_logits_per_agent
            a = torch.zeros((1, N), dtype=torch.long)
            lg = decode_logits_per_agent(m.decoder, rep, ob, a, gmask, ACT, m.tpdv)
        else:
            zs = np.zeros((1, N, 37), dtype=np.float32)
            v, rep = m.encoder(torch.tensor(zs), ob)
            sh = torch.zeros((1, N, ACT + 1)); sh[:, 0, 0] = 1
            lg = m.decoder(sh, rep, ob)
    return lg[0].numpy(), v[0].numpy()

base_lg, base_v = logits_and_values(model, state, obs)

obs_pert = obs.copy(); obs_pert[0, 4, :] = np.random.rand(OBS)   # perturb agent 4 (singleton)
pert_lg, pert_v = logits_and_values(model, state, obs_pert)

d_lg = np.abs(base_lg - pert_lg).max(axis=1)
d_v = np.abs(base_v - pert_v).ravel()
print(f"  per-agent max |dlogit| = {np.round(d_lg, 8)}")
print(f"  per-agent max |dvalue| = {np.round(d_v, 8)}")
for i in range(N):
    if i == 4:
        check(d_lg[i] > 1e-6, f"agent {i} (perturbed) DOES change")
    else:
        check(d_lg[i] < 1e-6, f"agent {i} logits unchanged by disconnected agent 4")
        check(d_v[i] < 1e-6, f"agent {i} value unchanged by disconnected agent 4")

# ------------------------------------------------------------ action leak
print("\n=== 3. action isolation ===")
from mat.algorithms.utils.mat_comm_act import decode_logits_per_agent
def logits_given_actions(acts):
    with torch.no_grad():
        st = torch.tensor(state); ob = torch.tensor(obs)
        gmask, cmask = model._masks(st)
        _, rep = model.encoder(st, ob, gmask)
        a = torch.tensor([acts], dtype=torch.long)
        return decode_logits_per_agent(
            model.decoder, rep, ob, a, gmask, ACT, model.tpdv
        )[0].numpy()

l0 = logits_given_actions([0, 0, 0, 0, 0, 0])
l_in = logits_given_actions([0, 2, 0, 0, 0, 0])   # agent 1's action changes (same group as 2)
l_out = logits_given_actions([0, 0, 0, 2, 0, 0])  # agent 3's action changes (singleton)

d_in = float(np.abs(l0[2] - l_in[2]).max())
check(d_in > 1e-6, "agent 2 REACTS to agent 1's action (same group)")
check(d_in > 0.1, "agent 2 Δlogit magnitude %.4f > 0.1 (energized; 1e-6 only tests wiring)" % d_in)
check(np.abs(l0[4] - l_out[4]).max() < 1e-6, "agent 4 ignores agent 3's action (different group)")
check(np.abs(l0[5] - l_out[5]).max() < 1e-6, "agent 5 ignores agent 3's action (different group)")
check(np.abs(l0[0] - l_in[0]).max() < 1e-6, "agent 0 ignores agent 1's action (0 decodes first)")

# --------------------------------------------- rollout / training agreement
print("\n=== 4. rollout vs teacher-forced log-probs (PPO ratio) ===")
B = 8
st = make_state(B)
# vary the bits per batch element to exercise several partitions
st[1, :, -N:] = np.array([1., 1., 1., 1., 1., 1.], np.float32)   # one big group
st[2, :, -N:] = np.array([0., 0., 0., 0., 0., 0.], np.float32)   # all isolated
ob = np.random.rand(B, N, OBS).astype(np.float32)
av = np.ones((B, N, ACT), dtype=np.float32); av[:, 0, 1] = 0     # mask an action

a, alog, v = model.get_actions(st, ob, av, deterministic=False)
alog2, v2, ent = model(st, ob, a.detach().numpy(), av)
alog, alog2 = alog.detach().numpy(), alog2.detach().numpy()

check(np.abs(alog - alog2).max() < 1e-5,
      f"log-probs agree (max diff {np.abs(alog - alog2).max():.2e}) -> PPO ratio == 1")
check(np.abs(v.detach().numpy() - v2.detach().numpy()).max() < 1e-6, "values agree between paths")
check(not np.isnan(alog2).any(), "no NaNs with an all-isolated batch element")
check((a.detach().numpy()[:, 0, 0] != 1).all(), "masked action never chosen for agent 0")

# ------------------------------------------------------------- vs baseline
print("\n=== 5. contrast: stock MAT does NOT isolate ===")
base = energize(MultiAgentTransformer(SD, OBS, ACT, N, n_block=2, n_embd=32, n_head=2))
base.eval()
b_lg, _ = logits_and_values(base, state, obs)
p_lg, _ = logits_and_values(base, state, obs_pert)
bd = np.abs(b_lg - p_lg).max(axis=1)
print(f"  stock MAT per-agent max |dlogit| = {np.round(bd, 6)}")
check((bd[[0, 1, 2, 3, 5]] > 1e-6).all(),
      "stock MAT leaks agent 4's obs into every other agent (expected)")

print("\n=== 6. raw obs skip: same token + same obs_rep, only obs changes ===")
from mat.algorithms.utils.mat_comm_act import _tokens_to_input
from mat.algorithms.utils.mat_comm_groups import build_group_causal_mask

st_t = torch.tensor(state)
ob_t = torch.tensor(obs)
gmask, _ = model._masks(st_t)
with torch.no_grad():
    _, rep = model.encoder(st_t, ob_t, gmask)
    acts = torch.zeros((1, N), dtype=torch.long)
    dec_in = _tokens_to_input(gmask, acts, ACT, model.tpdv)
    cm = build_group_causal_mask(gmask)
    lg_same_rep = model.decoder(dec_in, rep, ob_t, cm)
    ob_only = ob_t.clone()
    ob_only[0, 0] = torch.randn(OBS)
    lg_obs_flip = model.decoder(dec_in, rep, ob_only, cm)
d0 = float((lg_same_rep[0, 0] - lg_obs_flip[0, 0]).abs().max())
d1 = float((lg_same_rep[0, 1] - lg_obs_flip[0, 1]).abs().max())
check(d0 > 1e-6, "agent 0 logits change when only raw obs changes (skip live), Δ=%.4f" % d0)
check(d1 < 1e-6, "agent 1 logits unchanged (own obs fixed, skip is per-agent), Δ=%.4e" % d1)

print("\n" + ("ALL PASS" if not errs else f"{len(errs)} FAILURE(S): {errs}"))
sys.exit(1 if errs else 0)
