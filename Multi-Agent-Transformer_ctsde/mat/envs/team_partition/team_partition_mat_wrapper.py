"""MAT-side wrapper for the Team Partition benchmark (`team-partition-v0`).

`TeamPartitionEnv` (mac/envs/team_partition.py) speaks the
Gymnasium API; MAT's runners and vector envs expect the SMAC-style API used
throughout `mat/envs/starcraft2`. This module is the only translation layer;
nothing about the benchmark itself is changed here.

    Gymnasium                             MAT
    -------------------------------       ----------------------------------
    reset() -> obs, info                  reset() -> obs, share_obs, avail
    step()  -> obs, r, term, trunc, info  step()  -> obs, share_obs, rewards,
                                                     dones, infos, avail
    scalar reward                         (n_agents, 1) -- common reward, tiled
    single global state vector            (n_agents, state_dim + N) -- tiled,
                                          comm bits appended (see below)
    one info dict                         list of n_agents dicts

THE ENV IS BUILT FROM THE PAPER CONFIGURATION
---------------------------------------------
Defaults are mac.PAPER["team_partition"] -- the same constructor arguments the
EPyMARL and InforMARL baselines use -- and every one of them is a CLI override
of the same name (see train_team_partition.py).

THE COMM GRAPH IS APPENDED HERE, AS PER-AGENT BITS
--------------------------------------------------
The env's comm graph is a star on the staging cell: `get_comm_mask()` is
(N, 1), bit_i = 1 iff agent i is inside the radio disk (Manhattan comm_radius)
AND at least one other agent is too (`comm_mask_shape == 1`). That is the
aerial / charging layout, so `--comm_mask_kind bits` is the right reading and
`outer(bits)` gives ONE clique of assembled agents plus singletons.

But unlike aerial / charging, this env's `get_state()` is
[quota one-hot, N distances to the staging cell, t / time_limit] and does NOT
end with the bits, so nothing would reach `mat_comm` through the buffer on its
own. The wrapper appends the N bits, the same fix pickup_consensus's wrapper
applies. Do not append anything after them -- the tail slice is positional.

Fairness of the groups. EPyMARL's CommNet and TarMAC receive the same (N, 1)
bits through CommMAC and form M = outer(bits) (commnet_agent.py /
tarmac_agent.py); SDec-MAT's `build_group_mask(bits)` forms outer(bits) | I and
takes the transitive closure, which on an outer product of bits is a no-op. The
groups are identical.

OBSERVATIONS AND IDS
--------------------
obs is the env's per-agent vector untouched. `--obs_agent_id true` appends an
n_agents-wide one-hot AFTER the env returns it, so the env's Condition 3
tripwire (get_obs raises if the deciding agents' observations differ) still
runs on the id-free vector and is unaffected. share_obs never carries ids.
"""

import os
import sys

import numpy as np
from gymnasium.spaces import Box, Discrete

# <repo>/Multi-Agent-Transformer_ctsde/mat/envs/team_partition/this_file.py
# -> <repo>, which holds the `mac` benchmark package.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mac import PAPER, TeamPartitionEnv  # noqa: E402


def env_arg_defaults():
    """Constructor kwargs of the paper configuration (mac/presets.py)."""
    return dict(PAPER["team_partition"])


def env_args_from(args, defaults=None):
    """Constructor kwargs: paper defaults, overridden by any matching CLI arg."""
    defaults = env_arg_defaults() if defaults is None else defaults
    return {k: getattr(args, k, v) for k, v in defaults.items()}


def num_agents_from(args):
    """Team size, needed by the train script before an env is built."""
    return int(getattr(args, "n_agents", env_arg_defaults()["n_agents"]))


def assert_comm_mask_kind(args):
    """Refuse to launch a comm run whose state-tail layout is wrong.

    With `--comm_mask_kind pairwise` the model would reshape the last N^2
    entries of share_obs -- the bits plus most of the distance and quota
    fields -- into an adjacency. Nothing would crash; the groups would just be
    wrong.

    `use_centralized_V=False` makes the runner substitute obs for share_obs,
    which drops the comm tail altogether.
    """
    if getattr(args, "algorithm_name", "") not in ("mat_comm", "mat_comm_gru"):
        return
    if getattr(args, "comm_mask_kind", "bits") != "bits":
        raise ValueError(
            "Team Partition needs --comm_mask_kind bits for "
            f"{args.algorithm_name}. get_comm_mask() is (N, 1) per-agent bits "
            "(a star on the staging cell) and the wrapper appends those N bits "
            "to share_obs; 'pairwise' would read N^2 entries of state as a graph.")
    if not getattr(args, "use_centralized_V", True):
        raise ValueError(
            f"{args.algorithm_name} reads the comm graph from share_obs; with "
            "--use_centralized_V off the runner replaces share_obs with obs and "
            "the bits never reach the model.")


class TeamPartitionMATWrapper:
    """Team Partition exposed through MAT's SMAC-style env API."""

    def __init__(self, args):
        env_cls = TeamPartitionEnv
        self.env_kwargs = env_args_from(args)
        self._env = env_cls(**self.env_kwargs)

        info = self._env.get_env_info()
        self.n_agents = info["n_agents"]
        self.n_actions = info["n_actions"]
        # Optional one-hot agent index appended to obs (--obs_agent_id).
        # obs ONLY -- never share_obs, whose tail mat_comm slices for the comm
        # bits. Default off: the headline protocol is id-free.
        self.obs_agent_id = bool(getattr(args, "obs_agent_id", False))
        self.obs_dim = info["obs_shape"] + (
            self.n_agents if self.obs_agent_id else 0)
        # The env's own horizon (= time_limit).
        self.episode_limit = info["episode_limit"]

        if info.get("comm_mask_shape", 1) != 1:
            raise ValueError(
                "This wrapper appends per-agent comm BITS to the state; the env "
                f"reports comm_mask_shape={info.get('comm_mask_shape')}. A pairwise "
                "[N, N] mask needs the adjacency carried instead.")
        self.env_state_dim = info["state_shape"]
        self.state_dim = self.env_state_dim + self.n_agents

        # MAT indexes these with [0] and reads .shape / .n off the result. The
        # quota slots of a non-reader hold -1 (n_quota_readers), hence low=-1.
        self.observation_space = [
            Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.share_observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.action_space = [Discrete(self.n_actions) for _ in range(self.n_agents)]

    # ------------------------------------------------------------------
    # Views the runner consumes
    # ------------------------------------------------------------------

    def _obs(self):
        obs = np.stack(self._env.get_obs()).astype(np.float32)
        if self.obs_agent_id:
            obs = np.hstack([obs, np.eye(self.n_agents, dtype=np.float32)])
        return obs

    def _comm_bits(self):
        bits = np.asarray(self._env.get_comm_mask(), dtype=np.float32)
        if bits.shape != (self.n_agents, 1):
            raise ValueError(
                f"get_comm_mask() is {bits.shape}, expected ({self.n_agents}, 1).")
        return bits.reshape(-1)

    def _share_obs(self):
        # One global state, repeated per agent: MAT slices share_obs per agent
        # position but every slice is meant to be the same privileged state.
        # Its last n_agents entries are the comm bits (see module docstring).
        state = np.concatenate([
            np.asarray(self._env.get_state(), dtype=np.float32),
            self._comm_bits(),
        ])
        return np.tile(state, (self.n_agents, 1))

    def _avail(self):
        # The env masks nothing: an early GO is absorbed (commit_gate) rather
        # than masked, because a per-agent mask is itself a symmetry break.
        return np.asarray(self._env.get_avail_actions(), dtype=np.int64)

    def _infos(self, diag):
        """One dict per agent -- MAT reads infos[thread][agent_id][key].

        Every env info value is a summable scalar by the env's own contract
        (EPyMARL's parallel runner sums them), so all of them are forwarded
        rather than a hand-kept subset that would drift as the env grows keys.

        `bad_transition` is always False. Every truncation here is a real
        outcome, not a cut-off: the gate deadline or commit window expired, or
        nobody ever committed, and the env charges miss_penalty for it.
        """
        info = {k: float(v) for k, v in diag.items() if np.ndim(v) == 0}
        info["bad_transition"] = False
        info["won"] = bool(diag.get("exact_match", 0.0) >= 1.0)
        return [dict(info) for _ in range(self.n_agents)]

    # ------------------------------------------------------------------
    # MAT env API
    # ------------------------------------------------------------------

    def reset(self):
        self._env.reset()
        return self._obs(), self._share_obs(), self._avail()

    def step(self, actions):
        # MAT hands each env an (n_agents, 1) action block.
        actions = np.asarray(actions).reshape(-1).astype(int)

        _, reward, terminated, truncated, diag = self._env.step(actions)
        done = bool(terminated or truncated)

        rewards = np.full((self.n_agents, 1), float(reward), dtype=np.float32)
        dones = np.full(self.n_agents, done, dtype=bool)
        infos = self._infos(diag)

        return self._obs(), self._share_obs(), rewards, dones, infos, self._avail()

    def seed(self, seed=None):
        # Seeds the generator once; reset() then advances it, so successive
        # episodes differ while the whole run stays reproducible.
        self._env.seed(seed)

    def close(self):
        self._env.close()
