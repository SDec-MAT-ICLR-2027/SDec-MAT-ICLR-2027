"""MAT wrapper around CT-SDE `StagHunt` (explicit catch action).

YAML defaults from `src/config/envs/stag_hunt.yaml`. `comm_radius` is the
only intended experimental knob from the train CLI. share_obs is concat of
per-agent obs (not the full grid state) plus the flattened N×N comm mask
for mat_comm, matching the dicg_pp mat_comm plumbing.
"""

import os
import random
import sys

import numpy as np
from gym.spaces import Discrete


COMM_ALGOS = ("mat_comm", "mat_comm_gru")


def _add_ctsde_src_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(os.path.join(here, "..", "..", "..", "..", "src"))
    if not os.path.isdir(src):
        raise RuntimeError("CT-SDE src not found at %s" % src)
    if src not in sys.path:
        sys.path.insert(0, src)


_add_ctsde_src_to_path()
from envs.stag_hunt import StagHunt  # noqa: E402


def _stag_kwargs(args):
    return dict(
        map_name=getattr(args, "map_name", "stag_hunt"),
        world_shape=[int(args.grid_size), int(args.grid_size)],
        n_agents=int(args.n_agents),
        n_stags=int(args.n_stags),
        n_hare=int(getattr(args, "n_hare", 0)),
        episode_limit=int(args.episode_limit),
        comm_radius=int(args.comm_radius),
        miscapture_punishment=float(args.miscapture_punishment),
        capture_action_conditions=[2, 1],
        capture_action=True,
        capture_conditions=[0, 1],
        capture_freezes=True,
        capture_terminal=False,
        remove_frozen=True,
        prevent_cannibalism=True,
        agent_obs=[2, 2],
        agent_move_block=[0, 1, 2],
        observe_state=False,
        observe_walls=False,
        observe_ids=False,
        observe_one_hot=False,
        directed_observations=False,
        directed_cone_narrow=True,
        directed_exta_actions=True,
        mountain_slope=0.0,
        mountain_spawn=False,
        mountain_agent_row=-1,
        p_stags_rest=0.0,
        p_hare_rest=0.0,
        random_ghosts=False,
        reward_stag=float(args.capture_reward),
        reward_hare=1.0,
        reward_collision=0.0,
        reward_time=float(args.step_cost),
        toroidal=False,
    )


class StagHuntMATEnv:
    def __init__(self, args):
        self._env = StagHunt(**_stag_kwargs(args))
        self.n_agents = self._env.n_agents
        self.episode_limit = self._env.episode_limit
        self.n_actions = self._env.get_total_actions()
        self.algorithm_name = args.algorithm_name
        self._append_pairwise_comm = self.algorithm_name in COMM_ALGOS
        self.catch_id = int(self._env.action_labels["catch"])
        self._dicg_style_obs = bool(int(getattr(args, "dicg_style_obs", 0)))

        if self._append_pairwise_comm:
            kind = getattr(args, "comm_mask_kind", None)
            if kind != "pairwise":
                raise ValueError(
                    "stag_hunt mat_comm requires --comm_mask_kind pairwise"
                )

        obs_dim = 53 if self._dicg_style_obs else int(self._env.get_obs_size())
        state_dim = obs_dim * self.n_agents
        if self._append_pairwise_comm:
            state_dim = state_dim + self.n_agents * self.n_agents

        self.observation_space = [[obs_dim] for _ in range(self.n_agents)]
        self.share_observation_space = [[state_dim] for _ in range(self.n_agents)]
        self.action_space = [Discrete(self.n_actions) for _ in range(self.n_agents)]

        print(
            "[StagHuntMATEnv] algo=%s obs_dim=%d share_obs_dim=%d "
            "n_actions=%d catch_id=%d comm_radius=%d append_mask=%s "
            "dicg_style_obs=%s"
            % (self.algorithm_name, obs_dim, state_dim, self.n_actions,
               self.catch_id, self._env.comm_radius, self._append_pairwise_comm,
               self._dicg_style_obs),
            flush=True,
        )

        self._ep_step = 0
        self._ep_return = 0.0

    def _obs(self):
        if self._dicg_style_obs:
            return self._dicg_layout_obs()
        return np.asarray(self._env.get_obs(), dtype=np.float32)

    def _dicg_layout_obs(self):
        """Match visible DICG PP: own pos (2) + 5x5 stags + time + 5x5 agents.

        Crop is Chebyshev r=2, same indexing as PredatorPrey.get_agent_obs /
        _build_obs. Frozen agents are omitted from the occupancy map (they
        are removed from the Stag Hunt grid). Hare plane dropped (n_hare=0).
        """
        env = self._env
        n = self.n_agents
        gx, gy = int(env.grid_shape[0]), int(env.grid_shape[1])
        out = np.zeros((n, 53), dtype=np.float32)
        t = float(env.steps) / float(max(1, self.episode_limit))
        for i in range(n):
            r = int(env.agents[i, 0, 0])
            c = int(env.agents[i, 0, 1])
            out[i, 0] = r / float(gx)
            out[i, 1] = c / float(gy - 1)
            prey_map = out[i, 2:27].reshape(5, 5)
            agent_map = out[i, 28:53].reshape(5, 5)
            r0, r1 = max(0, r - 2), min(r + 3, gx)
            c0, c1 = max(0, c - 2), min(c + 3, gy)
            for p in range(env.n_stags):
                if env.prey_alive[p, 0] <= 0:
                    continue
                pr, pc = int(env.prey[p, 0, 0]), int(env.prey[p, 0, 1])
                if r0 <= pr < r1 and c0 <= pc < c1:
                    prey_map[pr - (r - 2), pc - (c - 2)] = 1.0
            for a in range(n):
                if env.agents_not_frozen[a, 0] <= 0:
                    continue
                ar, ac = int(env.agents[a, 0, 0]), int(env.agents[a, 0, 1])
                if r0 <= ar < r1 and c0 <= ac < c1:
                    agent_map[ar - (r - 2), ac - (c - 2)] = 1.0
            out[i, 27] = t
        return out

    def _share_obs(self):
        obs = self._obs()
        state = np.concatenate(obs, axis=0)
        if self._append_pairwise_comm:
            mask = np.asarray(self._env.get_comm_mask(), dtype=np.float32).reshape(-1)
            n2 = self.n_agents * self.n_agents
            if mask.size != n2:
                raise ValueError("comm mask size %d != %d" % (mask.size, n2))
            state = np.concatenate([state, mask], axis=0)
        return np.tile(state, (self.n_agents, 1))

    def _avail(self):
        return np.asarray(self._env.get_avail_actions(), dtype=np.float32)

    def reset(self):
        self._env.reset()
        self._ep_step = 0
        self._ep_return = 0.0
        return self._obs(), self._share_obs(), self._avail()

    def step(self, actions):
        acts = [int(a) for a in np.asarray(actions).reshape(-1)]
        _, reward, terminated, truncated, env_info = self._env.step(acts)

        self._ep_step += 1
        self._ep_return += float(reward)
        done = bool(terminated or truncated)

        rewards = np.full((self.n_agents, 1), float(reward), dtype=np.float32)
        dones = np.full(self.n_agents, done, dtype=bool)
        bad_transition = bool(env_info.get("episode_limit", False)) and done

        info = [
            {
                "bad_transition": bad_transition,
                "episode_steps": self._ep_step,
                "episode_return": self._ep_return,
                "joint_captures": int(env_info.get("caught_stags", 0)),
                "miscaptures": int(env_info.get("miscaptures", 0)),
                "solo_catch_attempts": int(env_info.get("solo_catch_attempts", 0)),
                "joint_catch_attempts": int(env_info.get("joint_catch_attempts", 0)),
                "episode_done": done,
            }
            for _ in range(self.n_agents)
        ]
        return self._obs(), self._share_obs(), rewards, dones, info, self._avail()

    def seed(self, seed=None):
        if seed is not None:
            seed = int(seed)
            random.seed(seed)
            np.random.seed(seed)
            self._env.seed(seed)
            self._env.reset()
            self._ep_step = 0
            self._ep_return = 0.0
        return seed

    def close(self):
        self._env.close()

    def save_replay(self):
        pass
