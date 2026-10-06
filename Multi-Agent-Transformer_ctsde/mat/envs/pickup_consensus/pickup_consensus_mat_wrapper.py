"""MAT-side wrapper for the (single-group) Pickup Consensus benchmark.

`PickupConsensusEnv` (mac/envs/pickup_consensus.py) speaks the
Gymnasium API; this is the same translation layer pattern as
`mat/envs/aerial/aerial_ordering_mat_wrapper.py`. Nothing about the benchmark
itself is changed here, and the env is constructed with EXACTLY the same
kwargs (n_agents, k_min, k_max, n_max, time_limit) and reward constants
(REWARD_CORRECT_PICKUP=10.0, REWARD_WRONG_COUNT=-10.0, REWARD_STEP=-0.3) that
the EPyMARL and InforMARL baselines train against (mac/presets.py) — this
wrapper does not touch the env,
so any SDec-MAT (mat_comm) run here is dynamics- and reward-equivalent to
those baselines. (Do NOT reuse train_multiple_pickup.py for this benchmark:
`MultiplePickupConsensusEnv` is a different class with different reward
constants (`success_reward=20.0` / `failure_penalty=20.0`) and extra machinery
(`coupling`, `arming_delay_max`, `commit_noise`) that don't exist here.)

    Gymnasium                             MAT
    -------------------------------       ----------------------------------
    reset() -> obs, info                  reset() -> obs, share_obs, avail
    step()  -> obs, r, term, trunc, info  step()  -> obs, share_obs, rewards,
                                                     dones, infos, avail
    scalar reward                         (n_agents, 1) — common reward, tiled
    single global state vector            (n_agents, state_dim) — tiled
    one info dict                         list of n_agents dicts

Communication. The env has no topology: `get_comm_mask()` always returns all
ones (every agent is mutually reachable — "fully connected: there is no range
or topology in this env"). Unlike Aerial Ordering, the env's own `get_state()`
does NOT carry per-agent connectivity bits in its tail (state_dim is fixed at
2: `[K/N, N/n_max]`), so this wrapper appends the `n_agents` comm bits itself
(`get_comm_mask()`, always 1s) after the base state before tiling. That gives
`mat_comm` / `mat_comm_gru` (`--comm_mask_kind bits`) the tail format
`comm_from_share_obs` expects, and it partitions into exactly ONE group
covering every agent — the single sequential decode chain across the whole
identical team is the mechanism this benchmark is designed to exercise (the
same symmetry-breaking argument SDec-MAT's autoregressive decode order relies
on), not a real communication constraint.
"""

import os
import sys

import numpy as np
from gymnasium.spaces import Box, Discrete

# <repo>/Multi-Agent-Transformer_ctsde/mat/envs/<pkg>/this_file.py -> <repo>,
# which holds the `mac` benchmark package.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mac import PAPER, PickupConsensusEnv  # noqa: E402


# Constructor parameters forwarded from the argparse namespace. Defaults are
# the paper configuration (mac/presets.py), so a bare run reproduces the
# paper's task. The env's __init__ ends in **kwargs and silently swallows
# unknown names, so this list is the only guard against a silent typo.
ENV_ARG_KEYS = (
    "n_agents",
    "k_min",
    "k_max",
    "n_max",
    "time_limit",
)

ENV_ARG_DEFAULTS = {k: PAPER["pickup_consensus"][k] for k in ENV_ARG_KEYS}

# Per-step info keys the benchmark returns (see PickupConsensusEnv.step). Read
# on the terminal step to get the episode's outcome.
DIAG_KEYS = (
    "n_pickup",
    "K",
    "success",
    "overshoot",
    "undershoot",
    "no_attempt",
)


def env_args_from(args):
    """Pull the env constructor kwargs out of an argparse namespace."""
    return {k: getattr(args, k, ENV_ARG_DEFAULTS[k]) for k in ENV_ARG_KEYS}


def num_agents_from(args):
    """Team size, needed by the train script before an env is built."""
    return int(getattr(args, "n_agents", ENV_ARG_DEFAULTS["n_agents"]))


def assert_comm_mask_kind(args):
    """Refuse to launch a comm run whose state-tail layout is wrong.

    This wrapper appends `n_agents` per-agent comm bits (see module
    docstring), which is the 'bits' layout. 'pairwise' would read the last
    `n_agents**2` state entries instead and silently pick up the wrong
    numbers.
    """
    if getattr(args, "algorithm_name", "") not in ("mat_comm", "mat_comm_gru"):
        return
    if getattr(args, "comm_mask_kind", "bits") != "bits":
        raise ValueError(
            "Pickup Consensus needs --comm_mask_kind bits for "
            f"{args.algorithm_name}. This wrapper appends n_agents per-agent "
            "comm bits (always 1: the env is fully connected) to the state "
            "tail; 'pairwise' would read n_agents**2 entries instead and "
            "silently build the wrong graph.")


class PickupConsensusMATWrapper:
    """Pickup Consensus exposed through MAT's SMAC-style env API."""

    def __init__(self, args):
        env_cls = PickupConsensusEnv
        self._env = env_cls(**env_args_from(args))

        info = self._env.get_env_info()
        self.n_agents = info["n_agents"]
        self.n_actions = info["n_actions"]
        # Optional one-hot agent index appended to obs (--obs_agent_id).
        # obs ONLY -- never share_obs, whose tail mat_comm slices for the comm
        # bits. Default off, so the observation is unchanged unless asked for.
        self.obs_agent_id = bool(getattr(args, "obs_agent_id", False))
        self.obs_dim = info["obs_shape"] + (
            self.n_agents if self.obs_agent_id else 0)
        # Base state ([K/N, N/n_max]) plus the n_agents comm bits this wrapper
        # appends (see module docstring) — the tail 'bits' format mat_comm
        # reads off share_obs.
        self.state_dim = info["state_shape"] + self.n_agents
        self.episode_limit = info["episode_limit"]

        # MAT indexes these with [0] and reads .shape / .n off the result.
        self.observation_space = [
            Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
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

    def _share_obs(self):
        # One global state, repeated per agent: MAT slices share_obs per agent
        # position but every slice is meant to be the same privileged state.
        # Its last n_agents entries are the comm bits (see module docstring).
        base = np.asarray(self._env.get_state(), dtype=np.float32)
        bits = np.asarray(self._env.get_comm_mask(), dtype=np.float32).reshape(-1)
        state = np.concatenate([base, bits])
        return np.tile(state, (self.n_agents, 1))

    def _avail(self):
        return np.asarray(self._env.get_avail_actions(), dtype=np.int64)

    def _infos(self, diag, bad_transition):
        """One dict per agent — MAT reads infos[thread][agent_id][key]."""
        info = {k: float(diag.get(k, 0.0)) for k in DIAG_KEYS}
        info["bad_transition"] = bool(bad_transition)
        # Solved == exactly K committed. Read by the eval loop.
        info["won"] = bool(diag.get("success", 0.0) >= 1.0)
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
        # A stall (no one ever commits) is a truncation, not a real terminal:
        # flag it so returns are not treated as if the episode genuinely
        # resolved there. Mirrors the aerial wrapper's convention.
        infos = self._infos(diag, bad_transition=truncated and not terminated)

        return self._obs(), self._share_obs(), rewards, dones, infos, self._avail()

    def seed(self, seed=None):
        # Seeds the generator once; reset() then advances it, so successive
        # episodes differ while the whole run stays reproducible.
        self._env.seed(seed)

    def close(self):
        self._env.close()
