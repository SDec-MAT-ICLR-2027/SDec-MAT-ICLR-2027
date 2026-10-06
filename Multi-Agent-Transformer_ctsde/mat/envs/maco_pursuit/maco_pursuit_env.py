"""MAT facade over CASEC MACO Pursuit.

MAPPO/mat_dec: tiled concat-obs share_obs, no comm tail.

mat_comm: append flattened N×N adjacency. Default groups are same-prey
catcher cliques (Manhattan ≤ 1, size ≥ 2). `--fixed_groups 3,3,4` replaces
that with a static partition (no radius, no prey assignment).
Group token: ready/GO count Manhattan ≤ 1; z also copied to agents
who can step onto that prey (Manhattan ≤ 2). No comm_radius.
`--comm_mask_kind pairwise` required.
"""

from __future__ import annotations

import os
import random
import sys
import types

import numpy as np
from gymnasium.spaces import Discrete


REQUIRED_ARGS = (
    "n_agents",
    "n_preys",
    "episode_limit",
    "map_size",
    "catch_reward",
    "catch_fail_reward",
    "sight_range",
    "algorithm_name",
    "seed",
    "n_rollout_threads",
    "fail_penalty_switch_steps",
    "obs_ready_count",
)

MAPPO_ALGOS = ("mappo", "mat_dec")
COMM_ALGOS = ("mat_comm",)
ALLOWED_ALGOS = MAPPO_ALGOS + COMM_ALGOS
# Must match PursuitEnv.is_catch_neighbor (Manhattan ≤ 1).
CATCH_RADIUS = 1
# Hear the pair's z if one step from a catch cell (not in the comm clique).
RECRUIT_RADIUS = 2


def assign_same_prey_catchers(
    agent_positions,
    prey_positions,
    alive_agents,
    alive_preys,
    radius=CATCH_RADIUS,
):
    """Nearest live prey within radius; ties → lowest prey id. -1 if none."""
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be >= 0, got %r" % radius)
    agent_positions = np.asarray(agent_positions)
    prey_positions = np.asarray(prey_positions)
    alive_agents = np.asarray(alive_agents, dtype=np.float32).reshape(-1)
    alive_preys = np.asarray(alive_preys, dtype=np.float32).reshape(-1)
    n_agents = int(alive_agents.shape[0])
    n_preys = int(alive_preys.shape[0])
    if n_agents < 1:
        raise ValueError("n_agents must be >= 1, got %r" % n_agents)
    if n_preys < 1:
        raise ValueError("n_preys must be >= 1, got %r" % n_preys)
    if agent_positions.shape != (n_agents, 2):
        raise ValueError(
            "agent_positions must be [%d, 2], got %s"
            % (n_agents, tuple(agent_positions.shape))
        )
    if prey_positions.shape != (n_preys, 2):
        raise ValueError(
            "prey_positions must be [%d, 2], got %s"
            % (n_preys, tuple(prey_positions.shape))
        )
    assigned = np.full(n_agents, -1, dtype=np.int64)
    assigned_dist = np.full(n_agents, np.iinfo(np.int32).max, dtype=np.int64)
    for i in range(n_agents):
        if alive_agents[i] < 0.5:
            continue
        ax = int(agent_positions[i, 0])
        ay = int(agent_positions[i, 1])
        for p in range(n_preys):
            if alive_preys[p] < 0.5:
                continue
            d = abs(ax - int(prey_positions[p, 0])) + abs(
                ay - int(prey_positions[p, 1])
            )
            if d > radius:
                continue
            if d < assigned_dist[i] or (d == assigned_dist[i] and p < assigned[i]):
                assigned_dist[i] = d
                assigned[i] = p
    return assigned


def ready_count_from_same_prey_catchers(
    agent_positions,
    prey_positions,
    alive_agents,
    alive_preys,
):
    """Per-agent clique size: hunters assigned Manhattan ≤ 1 to the same prey.

    Unassigned / dead rows are 1 (same convention as the group-token pack).
    """
    assigned = assign_same_prey_catchers(
        agent_positions, prey_positions, alive_agents, alive_preys
    )
    alive_agents = np.asarray(alive_agents, dtype=np.float32).reshape(-1)
    n_agents = int(assigned.shape[0])
    ready = np.ones(n_agents, dtype=np.float32)
    members = {}
    for i in range(n_agents):
        p = int(assigned[i])
        if p < 0 or alive_agents[i] < 0.5:
            continue
        members.setdefault(p, []).append(i)
    for ids in members.values():
        n = float(len(ids))
        for i in ids:
            ready[i] = n
    return ready


def group_adj_from_same_prey_catchers(
    agent_positions,
    prey_positions,
    alive_agents,
    alive_preys,
):
    """Pairwise adj: cliques of co-catchers of the same live prey.

    An agent is a catcher of prey p if it is alive and Manhattan(agent, p)
    ≤ CATCH_RADIUS. Each agent is assigned to at most one prey (nearest,
    then lowest prey id) so two prey cliques never chain through a shared
    hunter. A prey with fewer than two assigned catchers adds no edges —
    those hunters stay singletons (decoder START).

    Returns float32 [N, N] with diagonal 1.
    """
    assigned = assign_same_prey_catchers(
        agent_positions, prey_positions, alive_agents, alive_preys
    )
    n_agents = int(assigned.shape[0])
    adj = np.eye(n_agents, dtype=np.float32)
    members = {}
    for i in range(n_agents):
        p = int(assigned[i])
        if p < 0:
            continue
        members.setdefault(p, []).append(i)
    for ids in members.values():
        if len(ids) < 2:
            continue
        for i in ids:
            for j in ids:
                adj[i, j] = 1.0
    return adj


def _load_pursuit_env():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    casec_envs = os.path.join(repo, "CASEC-MACO-benchmark", "src", "envs")
    local_envs = os.path.join(repo, "src", "envs")
    if not os.path.isdir(casec_envs):
        raise FileNotFoundError(
            "CASEC-MACO-benchmark/src/envs not found at %s" % casec_envs
        )
    if local_envs not in sys.path:
        sys.path.insert(0, local_envs)
    from multiagentenv import MultiAgentEnv  # noqa: WPS433

    smac_mae = types.ModuleType("smac.env.multiagentenv")
    smac_mae.MultiAgentEnv = MultiAgentEnv
    sys.modules.setdefault("smac", types.ModuleType("smac"))
    sys.modules.setdefault("smac.env", types.ModuleType("smac.env"))
    sys.modules["smac.env.multiagentenv"] = smac_mae

    if casec_envs not in sys.path:
        sys.path.insert(0, casec_envs)
    from pursuit import PursuitEnv  # noqa: WPS433

    return PursuitEnv


PursuitEnv = _load_pursuit_env()


class PursuitEnvCatchMin(PursuitEnv):
    """CASEC PursuitEnv with success threshold catch_min (paper is 2).

    CASEC is not edited. Undershoot (0 < n < catch_min) gets catch_fail_reward;
    n >= catch_min gets catch_reward, prey dies, catch_min hunters removed.
    """

    def __init__(self, catch_min=2, **kwargs):
        super(PursuitEnvCatchMin, self).__init__(**kwargs)
        cm = int(catch_min)
        if cm < 2 or cm > 4:
            raise ValueError(
                "catch_min must be in 2..4 (prey has 4 adjacent seats), got %r"
                % catch_min
            )
        self.catch_min = cm

    def step(self, actions):
        self._total_steps += 1
        self._episode_steps += 1
        info = {}
        reward = 0
        terminated = False
        info["battle_won"] = False
        neighbors = self.neighbors
        catch_min = int(self.catch_min)

        def is_catch_neighbor(_agent_x, _agent_y, _prey_x, _prey_y):
            return abs(_agent_x - _prey_x) + abs(_agent_y - _prey_y) <= 1

        for prey_i in range(self.n_preys):
            if self.alive_preys[prey_i]:
                catch_number = 0
                prey_x, prey_y = (
                    self.prey_positions_idx[prey_i, 0],
                    self.prey_positions_idx[prey_i, 1],
                )
                for agent_i, action in enumerate(actions):
                    agent_x, agent_y = (
                        self.agent_positions_idx[agent_i, 0],
                        self.agent_positions_idx[agent_i, 1],
                    )
                    if (
                        is_catch_neighbor(agent_x, agent_y, prey_x, prey_y)
                        and action < 4
                        and agent_x + neighbors[action][0] == prey_x
                        and agent_y + neighbors[action][1] == prey_y
                    ):
                        catch_number += 1
                if 0 < catch_number < catch_min:
                    reward += self.catch_fail_reward
                if catch_number >= catch_min:
                    reward += self.catch_reward
                    self.alive_preys[prey_i] = 0
                    self.rest_prey -= 1
                    self.prey_positions[prey_x, prey_y] = 0
                    self.occ[prey_x, prey_y] = 0
                    self.prey_positions_idx[prey_i, 0] = self.map_size + 1
                    self.prey_positions_idx[prey_i, 1] = self.map_size + 1
                    predator_count = catch_min
                    for agent_i, action in enumerate(actions):
                        agent_x, agent_y = (
                            self.agent_positions_idx[agent_i, 0],
                            self.agent_positions_idx[agent_i, 1],
                        )
                        if (
                            is_catch_neighbor(agent_x, agent_y, prey_x, prey_y)
                            and action < 4
                            and agent_x + neighbors[action][0] == prey_x
                            and agent_y + neighbors[action][1] == prey_y
                        ):
                            self.alive_agents[agent_i] = 0
                            self.agent_positions[agent_x, agent_y] = 0
                            self.agent_positions_idx[agent_i, 0] = self.map_size + 1
                            self.agent_positions_idx[agent_i, 1] = self.map_size + 1
                            self.occ[agent_x, agent_y] = 0
                            predator_count -= 1
                            if predator_count == 0:
                                break
                if self.rest_prey == 0:
                    break

        info["prey_left"] = self.rest_prey

        for agent_i, action in enumerate(actions):
            if 4 <= action <= 7:
                x, y = self.agent_positions_idx[agent_i, 0], self.agent_positions_idx[agent_i, 1]
                if action == 4:
                    target_x, target_y = x, min(self.map_size - 1, y + 1)
                elif action == 5:
                    target_x, target_y = min(x + 1, self.map_size - 1), y
                elif action == 6:
                    target_x, target_y = x, max(0, y - 1)
                elif action == 7:
                    target_x, target_y = max(0, x - 1), y
                else:
                    continue
                if not self.occ[target_x, target_y]:
                    self.agent_positions[x, y] = 0
                    self.agent_positions[target_x, target_y] = agent_i + 1
                    self.occ[x, y] = 0
                    self.occ[target_x, target_y] = 1
                    self.agent_positions_idx[agent_i, 0] = target_x
                    self.agent_positions_idx[agent_i, 1] = target_y

        for prey_i in range(self.n_preys):
            if self.alive_preys[prey_i]:
                x, y = self.prey_positions_idx[prey_i, 0], self.prey_positions_idx[prey_i, 1]
                action = np.random.randint(low=0, high=5)
                if action <= 3:
                    if action == 0:
                        target_x, target_y = x, min(self.map_size - 1, y + 1)
                    elif action == 1:
                        target_x, target_y = min(x + 1, self.map_size - 1), y
                    elif action == 2:
                        target_x, target_y = x, max(0, y - 1)
                    else:
                        target_x, target_y = max(0, x - 1), y
                    if not self.occ[target_x, target_y]:
                        self.prey_positions[x, y] = 0
                        self.prey_positions[target_x, target_y] = prey_i + 1
                        self.occ[x, y] = 0
                        self.occ[target_x, target_y] = 1
                        self.prey_positions_idx[prey_i, 0] = target_x
                        self.prey_positions_idx[prey_i, 1] = target_y

        self._obs = -np.ones(
            [self.map_size + 2 * self.sight_range, self.map_size + 2 * self.sight_range]
        )
        self._obs[
            self.sight_range : -self.sight_range,
            self.sight_range : -self.sight_range,
        ] = self.agent_positions + 10 * self.prey_positions

        if self.rest_prey == 0:
            terminated = True
            info["battle_won"] = True
            self.battles_won += 1
        elif self._episode_steps >= self.episode_limit:
            terminated = True

        if terminated:
            self._episode_count += 1
            self.battles_game += 1

        return reward, terminated, info


def _expected_obs_size(map_size, sight_range):
    crop = (int(sight_range) * 2 + 1) ** 2
    return crop * 2 + int(map_size) * 2


class MACOPursuitMATEnv:
    """Vec-env API over CASEC PursuitEnv. Team reward tiled to all agents."""

    def __init__(self, args):
        missing = [k for k in REQUIRED_ARGS if not hasattr(args, k)]
        if missing:
            raise ValueError("missing required env args: %s" % missing)
        algo = str(args.algorithm_name)
        if algo not in ALLOWED_ALGOS:
            raise ValueError(
                "maco_pursuit wrapper allows mappo/mat_dec/mat_comm, got %r"
                % algo
            )
        if getattr(args, "comm_radius", None) is not None:
            raise ValueError(
                "do not pass comm_radius (got %r); use same-prey cliques "
                "or --fixed_groups"
                % args.comm_radius
            )
        from mat.algorithms.utils.mat_comm_groups import parse_fixed_group_sizes

        self._fixed_sizes = parse_fixed_group_sizes(
            getattr(args, "fixed_groups", None), int(args.n_agents)
        )
        raw_assign = getattr(args, "fixed_groups_assign", "id")
        if raw_assign is None:
            raw_assign = "id"
        raw_assign = str(raw_assign)
        if raw_assign not in ("id", "nearest"):
            raise ValueError(
                "fixed_groups_assign must be id or nearest, got %r" % raw_assign
            )
        self._fixed_assign = raw_assign
        if self._fixed_sizes is None and self._fixed_assign != "id":
            raise ValueError(
                "fixed_groups_assign=%r requires --fixed_groups" % self._fixed_assign
            )
        self._fixed_members = None
        self._append_pairwise_comm = algo in COMM_ALGOS
        raw_gt = getattr(args, "group_token", 0)
        if raw_gt is None:
            raise ValueError("group_token cannot be None")
        if int(raw_gt) not in (0, 1):
            raise ValueError("group_token must be 0 or 1, got %r" % raw_gt)
        self.use_group_token = int(raw_gt) == 1
        if self.use_group_token and algo not in COMM_ALGOS:
            raise ValueError("group_token=1 is mat_comm only, got %r" % algo)
        if self._fixed_sizes is not None:
            if algo not in COMM_ALGOS:
                raise ValueError("fixed_groups is mat_comm only, got %r" % algo)
            if self.use_group_token:
                raise ValueError("fixed_groups and group_token are mutually exclusive")
        raw_ready = getattr(args, "obs_ready_count", None)
        if raw_ready is None:
            raise ValueError("obs_ready_count cannot be None")
        if int(raw_ready) not in (0, 1):
            raise ValueError("obs_ready_count must be 0 or 1, got %r" % raw_ready)
        self.obs_ready_count = int(raw_ready) == 1
        if self.obs_ready_count and self.use_group_token:
            raise ValueError(
                "obs_ready_count=1 is MAPPO/mat_dec only; group_token already "
                "packs ready"
            )
        if self._append_pairwise_comm:
            kind = getattr(args, "comm_mask_kind", None)
            if kind != "pairwise":
                raise ValueError(
                "maco_pursuit mat_comm appends an N×N group adjacency "
                "to share_obs; set --comm_mask_kind pairwise. "
                    "Refusing bits: the last n_agents entries of concat obs "
                    "are not comm bits."
                )

        self._env_kwargs = dict(
            n_agents=int(args.n_agents),
            n_preys=int(args.n_preys),
            episode_limit=int(args.episode_limit),
            map_size=int(args.map_size),
            catch_reward=float(args.catch_reward),
            catch_fail_reward=float(args.catch_fail_reward),
            sight_range=int(args.sight_range),
            seed=int(args.seed),
        )
        if self._env_kwargs["n_agents"] < 1:
            raise ValueError("n_agents must be >= 1")
        if self._env_kwargs["n_preys"] < 1:
            raise ValueError("n_preys must be >= 1")
        if self._env_kwargs["episode_limit"] < 1:
            raise ValueError("episode_limit must be >= 1")
        if self._env_kwargs["map_size"] < 1:
            raise ValueError("map_size must be >= 1")
        if self._env_kwargs["sight_range"] < 0:
            raise ValueError("sight_range must be >= 0")

        catch_min_raw = getattr(args, "catch_min", 2)
        if catch_min_raw is None:
            raise ValueError("catch_min cannot be None")
        self.catch_min = int(catch_min_raw)
        if self.catch_min < 2 or self.catch_min > 4:
            raise ValueError(
                "catch_min must be in 2..4, got %r" % catch_min_raw
            )
        self._env = PursuitEnvCatchMin(catch_min=self.catch_min, **self._env_kwargs)
        switch = int(args.fail_penalty_switch_steps)
        if switch < -1:
            raise ValueError(
                "fail_penalty_switch_steps must be -1 (off) or >= 0, got %r" % switch
            )
        self._fail_after = None
        self._switch_at = None
        self._switched = False
        if switch == -1:
            if getattr(args, "catch_fail_reward_after", None) is not None:
                raise ValueError(
                    "catch_fail_reward_after requires fail_penalty_switch_steps >= 0"
                )
        else:
            if not hasattr(args, "catch_fail_reward_after"):
                raise ValueError(
                    "fail_penalty_switch_steps>=0 requires --catch_fail_reward_after"
                )
            after = args.catch_fail_reward_after
            if after is None:
                raise ValueError(
                    "fail_penalty_switch_steps>=0 requires --catch_fail_reward_after"
                )
            nthr = int(args.n_rollout_threads)
            if nthr < 1:
                raise ValueError("n_rollout_threads must be >= 1, got %r" % nthr)
            if switch % nthr != 0:
                raise ValueError(
                    "fail_penalty_switch_steps (%r) must be divisible by "
                    "n_rollout_threads (%r)" % (switch, nthr)
                )
            self._fail_after = float(after)
            self._switch_at = switch // nthr
            self._switched = False
        if int(self._env.n_agents) != self._env_kwargs["n_agents"]:
            raise ValueError(
                "PursuitEnv n_agents %r != args.n_agents %r"
                % (self._env.n_agents, self._env_kwargs["n_agents"])
            )
        expected_obs = _expected_obs_size(
            self._env_kwargs["map_size"], self._env_kwargs["sight_range"]
        )
        got_obs = int(self._env.get_obs_size())
        if got_obs != expected_obs:
            raise ValueError(
                "obs_size %r != expected %r (map_size=%r sight_range=%r)"
                % (got_obs, expected_obs, self._env_kwargs["map_size"],
                   self._env_kwargs["sight_range"])
            )
        if int(self._env.get_total_actions()) != 9:
            raise ValueError(
                "n_actions %r != 9" % self._env.get_total_actions()
            )

        self.n_agents = int(self._env.n_agents)
        self.episode_limit = int(self._env.episode_limit)
        self.n_actions = int(self._env.get_total_actions())
        self.algorithm_name = algo

        obs_dim = got_obs
        if self.obs_ready_count:
            obs_dim = got_obs + 1
        concat_state_dim = int(self.n_agents * obs_dim)
        if (not self.obs_ready_count) and int(self._env.get_state_size()) != self.n_agents * got_obs:
            raise ValueError(
                "state_size %r != n_agents*obs_dim %r"
                % (self._env.get_state_size(), self.n_agents * got_obs)
            )
        state_dim = concat_state_dim
        if self._append_pairwise_comm:
            state_dim = concat_state_dim + self.n_agents * self.n_agents
            if self.use_group_token:
                from mat.algorithms.utils.mat_comm_group_token import pack_width

                state_dim = state_dim + pack_width(self.n_agents)

        self.observation_space = [[obs_dim] for _ in range(self.n_agents)]
        self.share_observation_space = [[state_dim] for _ in range(self.n_agents)]
        self.action_space = [Discrete(self.n_actions) for _ in range(self.n_agents)]

        print(
            "[MACOPursuitMATEnv] algo=%s n_agents=%d n_preys=%d map=%d "
            "horizon=%d sight=%d catch=%s fail=%s obs_dim=%d share_obs_dim=%d "
            "n_actions=%d catch_min=%d fail_switch_at_env_steps=%s fail_after=%s "
            "append_pairwise_comm=%s groups=%s group_token=%s token_hear_radius=%s "
            "obs_ready_count=%s"
            % (
                self.algorithm_name,
                self.n_agents,
                self._env_kwargs["n_preys"],
                self._env_kwargs["map_size"],
                self.episode_limit,
                self._env_kwargs["sight_range"],
                self._env_kwargs["catch_reward"],
                self._env_kwargs["catch_fail_reward"],
                obs_dim,
                state_dim,
                self.n_actions,
                self.catch_min,
                self._switch_at,
                self._fail_after,
                self._append_pairwise_comm,
                (
                    "fixed_"
                    + "_".join(str(s) for s in self._fixed_sizes)
                    + (
                        "_nearest"
                        if self._fixed_assign == "nearest"
                        else ""
                    )
                    if self._fixed_sizes is not None
                    else (
                        "same_prey_catcher_clique"
                        if self._append_pairwise_comm
                        else "off"
                    )
                ),
                int(self.use_group_token),
                RECRUIT_RADIUS if self.use_group_token else 0,
                int(self.obs_ready_count),
            ),
            flush=True,
        )

        self._ep_step = 0
        self._ep_return = 0.0
        self._ep_joint = 0
        self._ep_solo = 0

    def _maybe_switch_fail_penalty(self):
        if self._switch_at is None or self._switched:
            return
        if int(self._env._total_steps) < int(self._switch_at):
            return
        old = float(self._env.catch_fail_reward)
        self._env.catch_fail_reward = float(self._fail_after)
        self._switched = True
        print(
            "[MACOPursuitMATEnv] fail_penalty switch env_steps=%d %s -> %s"
            % (self._env._total_steps, old, self._env.catch_fail_reward),
            flush=True,
        )

    def _ready_counts(self):
        inner = self._env
        return ready_count_from_same_prey_catchers(
            inner.agent_positions_idx,
            inner.prey_positions_idx,
            inner.alive_agents,
            inner.alive_preys,
        )

    def _obs(self):
        obs = np.asarray(self._env.get_obs(), dtype=np.float32)
        if self.obs_ready_count:
            ready = self._ready_counts().reshape(self.n_agents, 1)
            obs = np.concatenate([obs, ready], axis=-1)
        return obs

    def _group_token_pack(self):
        from mat.algorithms.utils.mat_comm_group_token import build_group_token_pack

        inner = self._env
        assigned = assign_same_prey_catchers(
            inner.agent_positions_idx,
            inner.prey_positions_idx,
            inner.alive_agents,
            inner.alive_preys,
            radius=CATCH_RADIUS,
        )
        recruit = assign_same_prey_catchers(
            inner.agent_positions_idx,
            inner.prey_positions_idx,
            inner.alive_agents,
            inner.alive_preys,
            radius=RECRUIT_RADIUS,
        )
        return build_group_token_pack(
            inner.agent_positions_idx,
            inner.prey_positions_idx,
            inner.alive_agents,
            inner.alive_preys,
            assigned,
            recruit,
        )

    def _share_obs(self):
        obs = self._obs()
        state = obs.reshape(-1)
        if self._append_pairwise_comm:
            parts = [state]
            if self.use_group_token:
                parts.append(self._group_token_pack())
            mask = self.get_comm_mask().reshape(-1)
            n2 = self.n_agents * self.n_agents
            if mask.size != n2:
                raise ValueError(
                    "get_comm_mask flattened size %d != n_agents**2=%d"
                    % (mask.size, n2)
                )
            parts.append(mask)
            state = np.concatenate(parts, axis=0)
        elif self.use_group_token:
            raise ValueError("group_token pack requires pairwise comm tail")
        return np.tile(state, (self.n_agents, 1))

    def _refresh_fixed_members(self):
        from mat.algorithms.utils.mat_comm_groups import (
            id_group_members,
            nearest_group_members,
        )

        if self._fixed_sizes is None:
            self._fixed_members = None
            return
        if self._fixed_assign == "id":
            self._fixed_members = id_group_members(self._fixed_sizes)
            return
        inner = self._env
        self._fixed_members = nearest_group_members(
            inner.agent_positions_idx,
            self._fixed_sizes,
            inner.alive_agents,
        )

    def get_comm_mask(self):
        """N×N groups: fixed partition, or same-prey catcher cliques."""
        inner = self._env
        if self._fixed_sizes is not None:
            from mat.algorithms.utils.mat_comm_groups import fixed_group_adjacency

            if self._fixed_members is None:
                self._refresh_fixed_members()
            return fixed_group_adjacency(
                self.n_agents,
                self._fixed_sizes,
                inner.alive_agents,
                members=self._fixed_members,
            )
        return group_adj_from_same_prey_catchers(
            inner.agent_positions_idx,
            inner.prey_positions_idx,
            inner.alive_agents,
            inner.alive_preys,
        )

    def _avail(self):
        return np.asarray(self._env.get_avail_actions(), dtype=np.float32)

    def _ascii_board(self):
        n = int(self._env.map_size)
        grid = [["." for _ in range(n)] for _ in range(n)]
        for prey_i in range(self._env.n_preys):
            if self._env.alive_preys[prey_i] < 0.5:
                continue
            px = int(self._env.prey_positions_idx[prey_i, 0])
            py = int(self._env.prey_positions_idx[prey_i, 1])
            grid[px][py] = "P"
        for i in range(self.n_agents):
            if self._env.alive_agents[i] < 0.5:
                continue
            ax = int(self._env.agent_positions_idx[i, 0])
            ay = int(self._env.agent_positions_idx[i, 1])
            grid[ax][ay] = format(i, "X")
        return "\n".join("".join(row) for row in grid)

    def _count_catches(self, acts):
        neighbors = self._env.neighbors
        solos = 0
        joints = 0
        events = []
        for prey_i in range(self._env.n_preys):
            if self._env.alive_preys[prey_i] < 0.5:
                continue
            px = int(self._env.prey_positions_idx[prey_i, 0])
            py = int(self._env.prey_positions_idx[prey_i, 1])
            hitters = []
            for i, a in enumerate(acts):
                if int(a) >= 4:
                    continue
                ax = int(self._env.agent_positions_idx[i, 0])
                ay = int(self._env.agent_positions_idx[i, 1])
                if abs(ax - px) + abs(ay - py) > 1:
                    continue
                dx, dy = neighbors[int(a)]
                if ax + dx == px and ay + dy == py:
                    hitters.append((i, int(a), ax, ay))
            n_hit = len(hitters)
            if n_hit == 0:
                continue
            kind = "joint" if n_hit >= self.catch_min else "undershoot"
            if n_hit >= self.catch_min:
                joints += 1
            else:
                solos += 1
            events.append(
                {
                    "kind": kind,
                    "prey": prey_i,
                    "agents": [h[0] for h in hitters],
                    "acts": [h[1] for h in hitters],
                    "prey_xy": (px, py),
                    "agent_xy": [(h[2], h[3]) for h in hitters],
                    "board": self._ascii_board() if n_hit >= self.catch_min else "",
                }
            )
        return solos, joints, events

    def reset(self):
        self._env.reset()
        self._refresh_fixed_members()
        self._ep_step = 0
        self._ep_return = 0.0
        self._ep_joint = 0
        self._ep_solo = 0
        return self._obs(), self._share_obs(), self._avail()

    def step(self, actions):
        acts = [int(a) for a in np.asarray(actions).reshape(-1)]
        if len(acts) != self.n_agents:
            raise ValueError("expected %d actions, got %d" % (self.n_agents, len(acts)))
        self._maybe_switch_fail_penalty()
        solos, joints, catch_events = self._count_catches(acts)
        reward, terminated, env_info = self._env.step(acts)

        self._ep_step += 1
        self._ep_return += float(reward)
        self._ep_joint += int(joints)
        self._ep_solo += int(solos)
        episode_done = bool(terminated)
        truncated = bool(self._ep_step >= self.episode_limit and env_info.get("prey_left", 0) > 0)
        bad_transition = bool(truncated)

        alive = np.asarray(self._env.alive_agents, dtype=np.float32)
        dones = np.array(
            [bool(episode_done or alive[i] < 0.5) for i in range(self.n_agents)],
            dtype=bool,
        )
        rewards = np.full((self.n_agents, 1), float(reward), dtype=np.float32)
        n_alive = int(alive.sum())
        prey_left = int(env_info.get("prey_left", self._env.rest_prey))
        won = bool(env_info.get("battle_won", False))

        info = [
            {
                "bad_transition": bad_transition,
                "episode_steps": self._ep_step,
                "episode_return": self._ep_return,
                "joint_catches": self._ep_joint,
                "solo_fails": self._ep_solo,
                "prey_left": prey_left,
                "n_alive_agents": n_alive,
                "won": won,
                "episode_done": episode_done,
                "catch_fail_reward": float(self._env.catch_fail_reward),
                "catch_events": catch_events,
            }
            for _ in range(self.n_agents)
        ]
        return self._obs(), self._share_obs(), rewards, dones, info, self._avail()

    def seed(self, seed=None):
        if seed is not None:
            seed = int(seed)
            random.seed(seed)
            np.random.seed(seed)
            self._env.reset()
            self._ep_step = 0
            self._ep_return = 0.0
            self._ep_joint = 0
            self._ep_solo = 0
        return seed

    def close(self):
        self._env.close()

    def save_replay(self):
        pass
