"""MAT-side wrapper for the Aerial Ordering benchmark.

`AerialOrderingEnv` (mac/envs/aerial_ordering.py) speaks the
Gymnasium API. MAT's runners and vector envs expect the SMAC-style API used
throughout `mat/envs/starcraft2`. This module is the only translation layer;
nothing about the benchmark itself is changed here.

What the translation actually does:

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

MAT additionally requires two info keys that the benchmark does not produce:

    bad_transition : True when the episode ended on the time limit rather than
                     on a real terminal. MAT's runner turns this into
                     `bad_masks`. (Note: this fork's `compute_returns` does not
                     currently consume bad_masks, so the key is carried for
                     interface compatibility and for the runner's logging.)
    won            : True when every target was lifted, i.e. the episode was
                     solved. Read by the eval loop.

Communication note. MAT has no notion of a communication graph: its encoder
attention is dense over all agents (`masked=False`), and the decoder's mask is
the autoregressive causal mask, not a comm mask. `get_comm_mask()` is therefore
deliberately NOT forwarded — it would be ignored. The benchmark's comm radius
still binds through the observations (K is -1.0 for a disconnected worker) and
through the PICKUP action mask, both of which pass through unchanged.
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

from mac import PAPER, AerialOrderingEnv  # noqa: E402


# Constructor parameters forwarded from the argparse namespace. Defaults are
# the paper configuration (mac/presets.py), so a bare run reproduces the
# paper's task. The env's __init__ ends in **kwargs and silently swallows
# unknown names, so this list is the only guard against a silent typo.
ENV_ARG_KEYS = (
    "obs_mode",
    "grid_size",
    "n_workers",
    "n_targets",
    "k_min",
    "k_max",
    "comm_radius",
    "sense_radius",
    "wind_prob",
    "time_limit",
)

ENV_ARG_DEFAULTS = {k: PAPER["aerial_ordering"][k] for k in ENV_ARG_KEYS}

# Per-episode counters the benchmark returns in `info`. Cumulative within an
# episode, so reading them on the terminal step gives the episode totals.
DIAG_KEYS = (
    "assembled_steps",
    "pickup_steps",
    "correct_pickups",
    "overshoot_pickups",
    "undershoot_pickups",
    "offtarget_pickups",
    "premature_pickups",
    "targets_lifted",
)


def env_args_from(args):
    """Pull the env constructor kwargs out of an argparse namespace."""
    return {k: getattr(args, k, ENV_ARG_DEFAULTS[k]) for k in ENV_ARG_KEYS}


def num_agents_from(args):
    """Team size, needed by the train script before an env is built."""
    return 1 + int(getattr(args, "n_workers", ENV_ARG_DEFAULTS["n_workers"]))


class AerialOrderingMATWrapper:
    """Aerial Ordering exposed through MAT's SMAC-style env API."""

    def __init__(self, args):
        env_cls = AerialOrderingEnv
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
        return np.tile(self._env.get_state().astype(np.float32), (self.n_agents, 1))

    def _avail(self):
        return np.asarray(self._env.get_avail_actions(), dtype=np.int64)

    def _infos(self, diag, bad_transition):
        """One dict per agent — MAT reads infos[thread][agent_id][key]."""
        info = dict(diag)
        info["bad_transition"] = bool(bad_transition)
        info["won"] = bool(diag.get("targets_lifted", 0) == self._env.n_targets)
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
