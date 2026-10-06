"""Load a trained policy, frozen, and act with it greedily on any number of agents.

One loader per training stack, all rebuilding the network from the checkpoint
itself (layer shapes, or the run's saved config for InforMARL), so no training
flag has to be remembered:

  sdec_mat               MAT fork, `mat_comm`   transformer_<step>.pt
  mat_dec                MAT fork, `mat_dec`    transformer_<step>.pt
  mappo, commnet,        epymarl/               <run>/<step>/agent.th
  tarmac, qmix
  informarl              InforMARL/mac_adapter  <run>/<step>/actor.pt (+ the run's config.json)

`--checkpoint` may point at the weights file, a step directory or a whole run
directory; the newest checkpoint inside is used unless a step is given.
`default_checkpoint` finds the run `run_experiment.py` trains for a given seed.

Every policy acts greedily (argmax): the generalisation tests ask what the
learned policy itself does, not a sample of it.
"""
import contextlib
import glob
import io
import json
import os
import sys
import types

import numpy as np
import torch as th

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAT_ROOT = os.path.join(REPO, "Multi-Agent-Transformer_ctsde")
EPYMARL_ROOT = os.path.join(REPO, "epymarl")
INFORMARL_ROOT = os.path.join(REPO, "InforMARL")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

METHODS = ("sdec_mat", "mat_dec", "mappo", "commnet", "tarmac", "qmix", "informarl")
EPYMARL = ("mappo", "commnet", "tarmac", "qmix")


def _add_path(p):
    if p not in sys.path:
        sys.path.insert(0, p)


# ----------------------------------------------------------------------------
# Locating checkpoints
# ----------------------------------------------------------------------------
def _newest(paths, key):
    if not paths:
        return None
    return max(paths, key=key)


def _mat_file(path, step=None):
    """transformer_<step>.pt from a file, a models/ dir or a MAT run dir."""
    if os.path.isfile(path):
        return path
    for models in (path, os.path.join(path, "models"),
                   *sorted(glob.glob(os.path.join(path, "run*", "models")))[::-1]):
        files = glob.glob(os.path.join(models, "transformer_*.pt"))
        if files:
            if step is not None:
                f = os.path.join(models, f"transformer_{step}.pt")
                if not os.path.isfile(f):
                    raise FileNotFoundError(f)
                return f
            return _newest(files, lambda p: int(p.rsplit("_", 1)[1][:-3]))
    raise FileNotFoundError(f"no transformer_*.pt under {path}")


def _step_dir(path, weights, step=None):
    """<run>/<step>/ holding `weights`, from a file, a step dir or a run dir."""
    if os.path.isfile(path):
        return os.path.dirname(path)
    if os.path.isfile(os.path.join(path, weights)):
        return path
    steps = [d for d in os.listdir(path)
             if d.isdigit() and os.path.isfile(os.path.join(path, d, weights))]
    if not steps:
        raise FileNotFoundError(f"no <step>/{weights} under {path}")
    if step is not None:
        if str(step) not in steps:
            raise FileNotFoundError(os.path.join(path, str(step), weights))
        return os.path.join(path, str(step))
    return os.path.join(path, max(steps, key=int))


def default_checkpoint(method, task, seed, agent_id):
    """The run `run_experiment.py` trains for (method, task, seed, agent ids)."""
    from configs import mac as MAC
    import mac

    suffix = "_aid" if agent_id else ""
    gym_id = mac.GYM_IDS[task]
    if method in ("sdec_mat", "mat_dec"):
        t = MAC.TASKS[task]
        algo = MAC.MAT_ARM_FLAGS[method]["algorithm_name"]
        return os.path.join(MAT_ROOT, "mat", "scripts", "results", t["mat_env_name"],
                            task, algo, f"{method}_s{seed}{suffix}")
    if method in EPYMARL:
        root, name = os.path.join(REPO, "results", "epymarl", "models"), f"{method}{suffix}"
    else:
        root, name = os.path.join(REPO, "results", "informarl", "models"), f"informarl{suffix}"
    runs = sorted(glob.glob(os.path.join(root, f"{name}_seed{seed}_{gym_id}_*")))
    if not runs:
        raise FileNotFoundError(f"no {method} run for seed {seed} under {root}")
    return runs[-1]


# ----------------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------------
class SDecMATPolicy:
    """SDec-MAT: communication-masked encoder + autoregressive decoder.

    No agent ids and no per-agent parameter, so it runs on any number of
    agents; the group mask decides who conditions on whom.
    """
    uses_ids = False

    def __init__(self, path, step=None, n_head=1):
        _add_path(MAT_ROOT)
        from mat.algorithms.mat.algorithm.mat_comm_transformer import CommMultiAgentTransformer
        from mat.algorithms.utils.mat_comm_act import discrete_autoregressive_act
        from mat.algorithms.utils.mat_comm_groups import (build_group_causal_mask,
                                                          build_group_mask)
        self._act_fn, self._causal = discrete_autoregressive_act, build_group_causal_mask
        self._group = build_group_mask
        self.path = _mat_file(path, step)
        sd = th.load(self.path, map_location="cpu")
        if "encoder.group_bias" not in sd and not any(k.startswith("decoder.blocks") for k in sd):
            raise ValueError(f"{self.path} is not a mat_comm (SDec-MAT) checkpoint")
        # Older checkpoints carry a decoder.obs_encoder that the decoder never
        # used in its forward pass; it is dropped.
        sd = {k: v for k, v in sd.items() if not k.startswith("decoder.obs_encoder.")}
        enc0 = sd["encoder.obs_encoder.0.weight"]
        norm_raw_obs = enc0.dim() == 1          # LayerNorm(obs_dim) first, else Linear
        self.obs_dim = int(enc0.shape[0] if norm_raw_obs else enc0.shape[1])
        self.action_dim = int(sd["decoder.head.3.weight"].shape[0])
        self.state_dim = int(sd["encoder.state_encoder.0.weight"].shape[0])
        blocks = [int(k.split(".")[2]) for k in sd if k.startswith("encoder.blocks.")]
        with contextlib.redirect_stdout(io.StringIO()):
            m = CommMultiAgentTransformer(
                self.state_dim, self.obs_dim, self.action_dim, 1,
                n_block=1 + max(blocks), n_embd=int(sd["encoder.ln.weight"].shape[0]),
                n_head=n_head, encode_state=False, device=th.device("cpu"),
                action_type="Discrete", comm_mask_kind="bits",
                encoder_critic_attn="soft_group_bias" if "encoder.group_bias" in sd else "hard",
                norm_raw_obs=norm_raw_obs,
                use_obs_skip="decoder.obs_skip.0.weight" in sd)
        m.load_state_dict(sd)
        self.model = _freeze(m)

    def group_mask(self, adjacency):
        """[1, N, N] groups from an [N, N] 0/1 adjacency (closed transitively)."""
        return self._group(th.as_tensor(np.asarray(adjacency, np.float32)).unsqueeze(0))

    @th.no_grad()
    def act(self, obs, adjacency=None, avail=None, group_mask=None):
        """Greedy joint action for N agents. Pass the [N, N] `adjacency`, or a
        `group_mask` already built from it."""
        n = obs.shape[0]
        gm = self.group_mask(adjacency) if group_mask is None else group_mask
        o = th.as_tensor(np.asarray(obs, np.float32)).unsqueeze(0)
        av = (th.ones(1, n, self.action_dim) if avail is None
              else th.as_tensor(np.asarray(avail, np.float32)).unsqueeze(0))
        _, rep = self.model.encoder(th.zeros(1, n, self.state_dim), o, gm)
        a, _ = self._act_fn(self.model.decoder, rep, o, gm, self._causal(gm), 1, n,
                            self.action_dim, self.model.tpdv, av, deterministic=True)
        return a[0, :, 0].numpy()


class _IDPolicy:
    """Shared-parameter policy whose agents are told apart by a one-hot id.

    `act(obs, ids, h)` takes the env observation and the [N, width] one-hot rows
    separately; the id is appended last, as in training. `id_width` is the
    one-hot width the network was trained with (0 = trained without ids).
    """
    uses_ids = True
    id_width = 0

    def init_hidden(self, n):
        return None


class MATDecPolicy(_IDPolicy):
    """MAT-Dec: one shared MLP actor over each agent's own observation."""

    def __init__(self, path, step=None, n_head=1, env_obs_dim=None):
        _add_path(MAT_ROOT)
        from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
        self.path = _mat_file(path, step)
        sd = th.load(self.path, map_location="cpu")
        mlp = sorted((int(k.split(".")[2]), v) for k, v in sd.items()
                     if k.startswith("decoder.mlp.") and k.endswith(".weight"))
        if not mlp:
            raise ValueError(f"{self.path} has no decoder.mlp: not a mat_dec checkpoint")
        linears = [v for _, v in mlp if v.dim() == 2]
        obs_in, self.action_dim = int(linears[0].shape[1]), int(linears[-1].shape[0])
        self.id_width = obs_in - env_obs_dim
        blocks = [int(k.split(".")[2]) for k in sd if k.startswith("encoder.blocks.")]
        with contextlib.redirect_stdout(io.StringIO()):
            m = MultiAgentTransformer(
                2, obs_in, self.action_dim, max(self.id_width, 1), n_block=1 + max(blocks),
                n_embd=int(linears[0].shape[0]), n_head=n_head, encode_state=False,
                device=th.device("cpu"), action_type="Discrete", dec_actor=True,
                share_actor=True, norm_raw_obs=mlp[0][1].dim() == 1)
        m.load_state_dict(sd)
        self.model = _freeze(m)

    @th.no_grad()
    def act(self, obs, ids, h=None):
        n = obs.shape[0]
        x = th.as_tensor(np.concatenate([obs, ids], axis=1), dtype=th.float32).unsqueeze(0)
        logits = self.model.decoder(th.zeros(1, n, self.action_dim + 1), None, x)
        return logits[0].argmax(-1).numpy(), None


class EPyMARLPolicy(_IDPolicy):
    """MAPPO / QMIX (RNN agent) and CommNet / TarMAC (communicating agents)."""

    def __init__(self, method, path, step=None, env_obs_dim=None):
        _add_path(EPYMARL_ROOT)
        from modules.agents import REGISTRY as agents
        self.dir = _step_dir(path, "agent.th", step)
        self.path = os.path.join(self.dir, "agent.th")
        sd = th.load(self.path, map_location="cpu")
        self.is_comm = method in ("commnet", "tarmac")
        first, last = (("encoder", "head") if self.is_comm else ("fc1", "fc2"))
        self.hidden = int(sd[f"{first}.weight"].shape[0])
        in_dim, n_actions = int(sd[f"{first}.weight"].shape[1]), int(sd[f"{last}.weight"].shape[0])
        self.id_width = in_dim - env_obs_dim
        args = types.SimpleNamespace(
            n_agents=max(self.id_width, 1), n_actions=n_actions, hidden_dim=self.hidden,
            use_rnn="rnn.weight_ih" in sd, comm_mask_type="bits", comm_symmetric=False)
        net = agents["rnn" if method in ("mappo", "qmix") else method](in_dim, args)
        net.load_state_dict(sd)
        self.net = _freeze(net)

    def init_hidden(self, n):
        return th.zeros(1, n, self.hidden)

    @th.no_grad()
    def act(self, obs, ids, h):
        n = obs.shape[0]
        x = th.as_tensor(np.concatenate([obs, ids], axis=1), dtype=th.float32).unsqueeze(0)
        if self.is_comm:
            logits, h = self.net(x, h, th.ones(1, n, 1))     # fully connected team
        else:
            q, hf = self.net(x.view(n, -1), h.view(n, -1))
            logits, h = q.unsqueeze(0), hf.view(1, n, -1)
        return logits[0].argmax(-1).numpy(), h


class InforMARLPolicy(_IDPolicy):
    """InforMARL: graph-attention actor with a GRU, over a fully connected team."""

    def __init__(self, path, step=None, env_obs_dim=None):
        _add_path(INFORMARL_ROOT)
        from gym import spaces
        from mac_adapter.env_wrapper import pairwise_to_adj
        from onpolicy.algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy
        self._adj = pairwise_to_adj
        self.dir = _step_dir(path, "actor.pt", step)
        self.path = os.path.join(self.dir, "actor.pt")
        run, token = os.path.dirname(self.dir), os.path.basename(os.path.dirname(self.dir))
        base = os.path.dirname(os.path.dirname(run))      # the dir holding models/
        # run_experiment.py layout (results/informarl/{models,runs}/<token>),
        # then the layout of earlier runs (results/{models,informarl}/<token>).
        cands = [os.path.join(base, "runs", token, "config.json"),
                 os.path.join(base, "informarl", token, "config.json")]
        cfg_path = next((c for c in cands if os.path.isfile(c)), None)
        if cfg_path is None:
            raise FileNotFoundError(f"no config.json for {run}; looked at {cands}")
        cfg = json.load(open(cfg_path))
        args = types.SimpleNamespace(**cfg["args"])
        info = cfg["env_info"]
        n, d, a = int(info["n_agents"]), int(info["obs_dim"]), int(info["n_actions"])
        self.id_width = d - env_obs_dim
        self.hidden, self.recurrent_n = int(args.hidden_size), int(args.recurrent_N)

        def box(shape):
            return spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

        policy = GR_MAPPOPolicy(args, box((d,)), box((n * d,)), box((n, d + 1)),
                                box((1,)), spaces.Discrete(a))
        policy.actor.load_state_dict(th.load(self.path, map_location="cpu"))
        self.actor = _freeze(policy.actor)

    def init_hidden(self, n):
        return np.zeros((n, self.recurrent_n, self.hidden), np.float32)

    @th.no_grad()
    def act(self, obs, ids, h):
        n = obs.shape[0]
        x = np.concatenate([obs, ids], axis=1).astype(np.float32)
        pair = np.ones((n, n), dtype=bool)
        np.fill_diagonal(pair, False)                         # fully connected team
        node = np.concatenate([x, np.zeros((n, 1), np.float32)], axis=1)
        adj = self._adj(pair)
        actions, _, h = self.actor(x, np.repeat(node[None], n, 0), np.repeat(adj[None], n, 0),
                                   np.arange(n).reshape(n, 1), h,
                                   np.ones((n, 1), np.float32), None, deterministic=True)
        return actions.view(-1).cpu().numpy(), h.cpu().numpy()


def _freeze(net):
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def load_policy(method, path, env_obs_dim, step=None, n_head=1):
    """A frozen policy for `method` from `path` (file, step dir or run dir)."""
    if method == "sdec_mat":
        return SDecMATPolicy(path, step, n_head)
    if method == "mat_dec":
        return MATDecPolicy(path, step, n_head, env_obs_dim)
    if method in EPYMARL:
        return EPyMARLPolicy(method, path, step, env_obs_dim)
    if method == "informarl":
        return InforMARLPolicy(path, step, env_obs_dim)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
