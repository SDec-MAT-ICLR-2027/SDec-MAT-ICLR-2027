"""MAT-side wrapper for the Charging Rendezvous benchmark.

`ChargingRendezvousEnv` (mac/envs/charging_rendezvous.py)
speaks the Gymnasium API; MAT's runners and vector envs expect the SMAC-style API
used throughout `mat/envs/starcraft2`. This module is the only translation layer,
and it is a near-copy of `mat/envs/aerial/aerial_ordering_mat_wrapper.py` — the
two benchmarks now expose the SAME communication contract, so the two wrappers
deliberately look alike.

What the translation does:

    Gymnasium                             MAT
    -------------------------------       ----------------------------------
    reset() -> obs, info                  reset() -> obs, share_obs, avail
    step()  -> obs, r, term, trunc, info  step()  -> obs, share_obs, rewards,
                                                     dones, infos, avail
    scalar reward                         (n_agents, 1) — the benchmark is
                                          common-reward, so it is replicated
    single global state vector            (n_agents, state_dim) — tiled, MAT
                                          indexes share_obs per agent
    one info dict                         list of n_agents dicts

MAT additionally requires two info keys the benchmark does not produce:

    bad_transition : True when the episode ended on the time limit rather than a
                     real terminal. MAT's runner turns this into `bad_masks`.
    won            : True when every drone reached its target level.

Communication. The env's `get_comm_mask()` is deliberately NOT forwarded, and it
does not need to be: `ChargingRendezvousEnv.get_state()` ends with
`comm_bits`, so the last `n_agents` entries of the tiled `share_obs` ARE the
per-agent connectivity bits. That is exactly what `mat_comm`/`mat_comm_gru` read
(`_comm_bits_from_state` -> `state[:, 0, -n_agent:]`), so the comm graph reaches
the model through the buffer with no extra plumbing. Plain `mat` ignores the
tail, as it ignores the rest of the state.

Do not append anything to the state after the comm bits, in either the env or
here — the tail slice is positional.
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

from mac import PAPER, ChargingRendezvousEnv  # noqa: E402


# Constructor parameters forwarded from the argparse namespace. Defaults are
# the paper configuration (mac/presets.py), so a bare run reproduces the
# paper's task. The env's __init__ ends in **kwargs and silently swallows
# unknown names, so this list is the only guard against a silent typo.
ENV_ARG_KEYS = (
    "n_agents",
    "grid_size",
    "capacity",
    "cohort_sizes",
    "cohort_distances",
    "init_battery",
    "battery_sigma",
    "target_level",
    "drain_rate",
    "charge_rate",
    "comm_radius",
    "dock_radius",
    "shaping_weight",
    "wind_prob",
    "time_limit",
)

ENV_ARG_DEFAULTS = {k: PAPER["charging_rendezvous"][k] for k in ENV_ARG_KEYS}

# Per-episode counters the benchmark returns in `info`. Cumulative within an
# episode, so reading them on the terminal step gives the episode totals.
DIAG_KEYS = (
    "brownouts",
    "brownout_excess",
    "wasted_port_steps",
    "wasted_ports",
    "connects",
    "charge_steps",
    "docked_wait_steps",
    "mesh_size_sum",
    "n_satisfied",
    "dryout",
    "all_satisfied",
)


def env_args_from(args):
    """Pull the env constructor kwargs out of an argparse namespace."""
    return {k: getattr(args, k, ENV_ARG_DEFAULTS[k]) for k in ENV_ARG_KEYS}


def num_agents_from(args):
    """Team size, needed by the train script before an env is built."""
    return int(getattr(args, "n_agents", ENV_ARG_DEFAULTS["n_agents"]))


class ChargingRendezvousMATWrapper:
    """Charging Rendezvous exposed through MAT's SMAC-style env API."""

    def __init__(self, args):
        env_cls = ChargingRendezvousEnv
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
        self.state_dim = info["state_shape"]
        self.episode_limit = info["episode_limit"]

        # The comm bits live in the last n_agents entries of the state. mat_comm
        # slices them positionally, so a state too short to hold them would fail
        # silently (it would read real state fields as connectivity bits).
        if self.state_dim < self.n_agents:
            raise ValueError(
                f"state_shape ({self.state_dim}) < n_agents ({self.n_agents}); "
                f"the last n_agents state entries must be the comm bits.")
        if info.get("comm_mask_shape", 1) != 1:
            raise ValueError(
                "This wrapper assumes per-agent comm BITS (comm_mask_shape == 1), "
                f"got {info.get('comm_mask_shape')}. An env emitting a pairwise "
                "[N, N] mask needs the adjacency to be carried explicitly; the "
                "state tail cannot hold it.")

        # MAT indexes these with [0] and reads .shape / .n off the result.
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

    def _share_obs(self):
        # One global state, repeated per agent: MAT slices share_obs per agent
        # position but every slice is meant to be the same privileged state.
        # Its last n_agents entries are the comm bits (see module docstring).
        return np.tile(self._env.get_state().astype(np.float32), (self.n_agents, 1))

    def _avail(self):
        return np.asarray(self._env.get_avail_actions(), dtype=np.int64)

    def _infos(self, diag, bad_transition):
        """One dict per agent — MAT reads infos[thread][agent_id][key]."""
        info = {k: float(diag.get(k, 0.0)) for k in DIAG_KEYS}
        info["bad_transition"] = bool(bad_transition)
        info["won"] = bool(diag.get("all_satisfied", 0))
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
        # Truncation is not a real terminal: flag it so returns are not treated
        # as if the episode genuinely ended there.
        infos = self._infos(diag, bad_transition=truncated and not terminated)

        return self._obs(), self._share_obs(), rewards, dones, infos, self._avail()

    def seed(self, seed=None):
        # Seeds the generator once; reset() then advances it, so successive
        # episodes differ while the whole run stays reproducible.
        self._env.seed(seed)

    def close(self):
        self._env.close()
