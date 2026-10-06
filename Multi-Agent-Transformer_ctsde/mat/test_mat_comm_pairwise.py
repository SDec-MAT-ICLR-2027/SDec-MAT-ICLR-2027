"""Isolation checks for mat_comm pairwise (DICG-style N×N adjacency).

Two disjoint groups {0,1,2} and {3,4,5}. First agent of group 2 must not
see group 1's last action. From repo root:

    python -u Multi-Agent-Transformer_ctsde/mat/test_mat_comm_pairwise.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath("Multi-Agent-Transformer_ctsde"))

import numpy as np
import torch

from mat.algorithms.mat.algorithm.mat_comm_transformer import CommMultiAgentTransformer
from mat.algorithms.utils.mat_comm_act import decode_logits_per_agent
from mat.algorithms.utils.mat_comm_groups import build_group_mask, component_ids

torch.manual_seed(0)
np.random.seed(0)

N, OBS, ACT = 6, 5, 3
# Two cliques, no edge between them. Agent 2 is last of group 1; agent 3 is
# first of group 2 — the stock MAT right-shift leak.
ADJ = np.zeros((N, N), dtype=np.float32)
ADJ[0:3, 0:3] = 1.0
ADJ[3:6, 3:6] = 1.0
np.fill_diagonal(ADJ, 1.0)
SD = 10 + N * N  # dummy prefix + flattened adjacency

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
errs = []


def check(c, m):
    print("  %s: %s" % (PASS if c else FAIL, m))
    if not c:
        errs.append(m)


def make_state(B=1):
    s = np.random.rand(B, N, SD).astype(np.float32)
    s[:, :, -N * N:] = ADJ.reshape(-1)
    return s


def energize(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear):
            torch.nn.init.orthogonal_(mod.weight, gain=1.0)
            if mod.bias is not None:
                torch.nn.init.normal_(mod.bias, std=0.1)
    return m


model = energize(CommMultiAgentTransformer(
    SD, OBS, ACT, N, n_block=2, n_embd=32, n_head=2, comm_mask_kind="pairwise",
))
model.eval()

print("\n=== 1. pairwise groups ===")
gm = build_group_mask(torch.tensor(ADJ).unsqueeze(0))
cid = component_ids(gm)[0].tolist()
check(cid == [0, 0, 0, 3, 3, 3], "components = %s (expect [0,0,0,3,3,3])" % cid)
check(bool(gm[0, 0, 2]) and not bool(gm[0, 2, 3]), "0~2 same group, 2~3 not")

print("\n=== 2. bits layout must NOT silently accept this state ===")
try:
    bad = CommMultiAgentTransformer(
        SD, OBS, ACT, N, n_block=1, n_embd=32, n_head=1, comm_mask_kind="bits",
    )
    st = torch.tensor(make_state())
    # bits reader would take last N of the flattened adj (last row), not fail.
    # Constructor succeeds (state_dim is large). Isolation would be wrong.
    # Guard: model groups under bits must differ from pairwise.
    gm_bits = build_group_mask(st[:, 0, -N:])
    gm_pair = build_group_mask(st[:, 0, -N * N:].reshape(-1, N, N))
    check(
        not bool(torch.equal(gm_bits, gm_pair)),
        "bits vs pairwise groups differ on this state (would be a silent bug "
        "if DICG used bits)",
    )
    del bad
except Exception as e:
    check(False, "unexpected constructor error: %s" % e)

print("\n=== 3. observation isolation ===")
state = make_state()
obs = np.random.rand(1, N, OBS).astype(np.float32)


def logits_and_values(state, obs):
    with torch.no_grad():
        st = torch.tensor(state)
        ob = torch.tensor(obs)
        gmask, cmask = model._masks(st)
        v, rep = model.encoder(st, ob, gmask)
        a = torch.zeros((1, N), dtype=torch.long)
        lg = decode_logits_per_agent(model.decoder, rep, ob, a, gmask, ACT, model.tpdv)
    return lg[0].numpy(), v[0].numpy()


base_lg, base_v = logits_and_values(state, obs)
obs_pert = obs.copy()
obs_pert[0, 5, :] = np.random.rand(OBS)
pert_lg, pert_v = logits_and_values(state, obs_pert)
d_lg = np.abs(base_lg - pert_lg).max(axis=1)
d_v = np.abs(base_v - pert_v).ravel()
print("  per-agent max |dlogit| = %s" % np.round(d_lg, 8))
for i in range(N):
    if i < 3:
        check(d_lg[i] < 1e-6, "agent %d logits unchanged by group-2 obs" % i)
        check(d_v[i] < 1e-6, "agent %d value unchanged by group-2 obs" % i)
    else:
        check(d_lg[i] > 1e-6, "agent %d in group 2 DOES change" % i)

print("\n=== 4. first agent of group 2 ignores previous group's last action ===")


def logits_given_actions(acts):
    with torch.no_grad():
        st = torch.tensor(state)
        ob = torch.tensor(obs)
        gmask, cmask = model._masks(st)
        _, rep = model.encoder(st, ob, gmask)
        a = torch.tensor([acts], dtype=torch.long)
        return decode_logits_per_agent(
            model.decoder, rep, ob, a, gmask, ACT, model.tpdv
        )[0].numpy()


l0 = logits_given_actions([0, 0, 0, 0, 0, 0])
l_prev_group = logits_given_actions([0, 0, 2, 0, 0, 0])  # agent 2, last of group 1
l_same = logits_given_actions([0, 0, 0, 2, 0, 0])  # agent 3 action; 4 should react
check(
    np.abs(l0[3] - l_prev_group[3]).max() < 1e-6,
    "agent 3 (first of group 2) ignores agent 2's action",
)
check(
    np.abs(l0[4] - l_same[4]).max() > 1e-6,
    "agent 4 REACTS to agent 3's action (same group)",
)

print("\n=== 5. line graph A—B—C is NOT a clique ===")
from mat.algorithms.utils.mat_comm_groups import transitive_closure

CHAIN = np.eye(3, dtype=np.float32)
CHAIN[0, 1] = CHAIN[1, 0] = 1.0
CHAIN[1, 2] = CHAIN[2, 1] = 1.0
gm_chain = build_group_mask(torch.tensor(CHAIN).unsqueeze(0))
closed = transitive_closure(gm_chain)
check(bool(gm_chain[0, 0, 1]) and bool(gm_chain[0, 1, 2]), "1-hop edges A—B and B—C")
check(not bool(gm_chain[0, 0, 2]), "A does not attend to C (no closure)")
check(bool(closed[0, 0, 2]), "closure would have connected A—C (diagnostic)")
check(bool(gm_chain[0, 0, 0]), "diagonal stays on")
from mat.algorithms.utils.mat_comm_groups import one_hop_shift_leak

leak = one_hop_shift_leak(gm_chain)[0].tolist()
check(leak == [False, False, False], "line-graph leak is 0 after per-agent decode: %s" % leak)

print("\n=== 5b. A—B—C: C ignores A, reacts to B ===")
N3, SD3 = 3, 10 + 9
m3 = energize(CommMultiAgentTransformer(
    SD3, OBS, ACT, N3, n_block=2, n_embd=32, n_head=2, comm_mask_kind="pairwise",
))
m3.eval()
st3 = np.random.rand(1, N3, SD3).astype(np.float32)
st3[:, :, -9:] = CHAIN.reshape(-1)
ob3 = np.random.rand(1, N3, OBS).astype(np.float32)

def lg3(acts):
    with torch.no_grad():
        gmask, _ = m3._masks(torch.tensor(st3))
        _, rep = m3.encoder(torch.tensor(st3), torch.tensor(ob3), gmask)
        a = torch.tensor([acts], dtype=torch.long)
        return decode_logits_per_agent(
            m3.decoder, rep, torch.tensor(ob3), a, gmask, ACT, m3.tpdv
        )[0].numpy()

z = lg3([0, 0, 0])
check(np.abs(z[2] - lg3([2, 0, 0])[2]).max() < 1e-6, "C ignores A's action")
check(np.abs(z[2] - lg3([0, 2, 0])[2]).max() > 1e-6, "C REACTS to B's action")

print("\n=== 6. LBF neighborhood cap does not clique-ify a chain ===")
from mat.envs.lbf.lbf_mat_wrapper import _cap_neighborhoods

capped = _cap_neighborhoods(CHAIN, 3)
check(capped[0, 2] < 0.5 and capped[2, 0] < 0.5, "cap=3 keeps A—C off")
check(capped[0, 1] > 0.5 and capped[1, 2] > 0.5, "cap=3 keeps 1-hop edges")
capped2 = _cap_neighborhoods(CHAIN, 2)
check(capped2[0].sum() <= 2.0 + 1e-6, "cap=2 bounds row 0 to self+1")

print("\n=== 7. rollout vs teacher-forced log-probs ===")
B = 8
st = make_state(B)
ob = np.random.rand(B, N, OBS).astype(np.float32)
av = np.ones((B, N, ACT), dtype=np.float32)
a, alog, v = model.get_actions(st, ob, av, deterministic=False)
alog2, v2, ent = model(st, ob, a.detach().numpy(), av)
check(
    np.abs(alog.detach().numpy() - alog2.detach().numpy()).max() < 1e-5,
    "log-probs agree -> PPO ratio == 1",
)

print("\n" + ("ALL PASS" if not errs else "%d FAILURE(S): %s" % (len(errs), errs)))
sys.exit(1 if errs else 0)
