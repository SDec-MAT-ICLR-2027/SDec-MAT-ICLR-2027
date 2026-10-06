"""MAT-side wrapper for Level-Based Foraging (B vs C test).

ForagingEnv speaks Gymnasium. MAT's runners expect the SMAC-style API.
This module is the only translation layer; the LBF rules are unchanged.

Pinned-roster contract (coalition-ambiguity env):
    8x8, 4 agents at levels 1,1,2,3 (via --player_levels, shuffle off),
    6 foods all level 4, horizon 40, penalty 0, full obs.
    Minimal coalitions per food: {3,1}, {3,1}', {1,1,2}.
"""

import os
import sys

import numpy as np
from gymnasium.spaces import Box, Discrete

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)
_LBF_ROOT = os.path.join(_PROJECT_ROOT, "lb-foraging")
if _LBF_ROOT not in sys.path:
    sys.path.insert(0, _LBF_ROOT)

from lbforaging.foraging.environment import Action, ForagingEnv  # noqa: E402

COMM_ALGOS = ("mat_comm", "mat_comm_gru")

# Gym Action.{NONE,NORTH,SOUTH,WEST,EAST}; LOAD is replaced by LOAD[food_k]
# when food_indexed_load=1. Gym still only understands Action.LOAD.
N_MOVE = 5

ENV_ARG_KEYS = (
    "grid_size",
    "n_agents",
    "n_foods",
    "min_player_level",
    "max_player_level",
    "min_food_level",
    "max_food_level",
    "sight",
    "max_episode_steps",
    "force_coop",
    "normalize_reward",
    "penalty",
    "player_levels",
    "pin_player_levels",
    "food_sight",
    "agent_sight",
    "food_levels",
    "exact_match",
)

ENV_ARG_DEFAULTS = {
    "grid_size": 8,
    "n_agents": 4,
    "n_foods": 6,
    "min_player_level": 1,
    "max_player_level": 1,
    "min_food_level": 2,
    "max_food_level": 4,
    "sight": 8,
    "max_episode_steps": 50,
    "force_coop": 0,
    "normalize_reward": 1,
    "penalty": 0.0,
    "player_levels": "",
    "pin_player_levels": -1,
    "food_sight": -1,
    "agent_sight": -1,
    "food_levels": "",
    "step_cost": 0.0,
    "exact_match": 0,
}

DIAG_KEYS = (
    "solved",
    "ep_length",
    "collected_lvl1",
    "collected_lvl2",
    "collected_lvl3",
    "collected_lvl4",
    "collected_lvl3plus",
    "spawned_lvl1",
    "spawned_lvl2",
    "spawned_lvl3",
    "spawned_lvl4",
    "spawned_lvl3plus",
    "n_coop_foods",
    "split_rate",
    "failed_load_rate",
    "n_load_actions",
    "n_failed_loads",
    "load_overshoot_rate",
    "load_undershoot_rate",
    "n_load_overshoot",
    "n_load_undershoot",
    "parallel_collect",
    "coalition_entropy",
    "n_coalition_types",
    "n_collected",
    "wasted_adjacency",
    "alloc_efficiency",
    "frac_c31",
    "frac_c112",
    "frac_c22",
    "t_first_adj",
    "parallel_work_rate",
    "group_size_mean",
    "group_frac_1",
    "group_frac_2",
    "group_frac_3",
    "group_frac_4",
    "foods_vis_pre",
    "foods_vis_post",
    "collect_return",
    "group_memory_reset_rate",
    "food_untouched",
    "food_solo",
    "food_async",
    "food_paired_fail",
    "food_cluster",
    "n_foods_left",
)


def env_kwargs_from(args):
    return {k: getattr(args, k, ENV_ARG_DEFAULTS[k]) for k in ENV_ARG_KEYS}


def _parse_int_list(val, n, name):
    if val is None or val == "":
        return None
    if isinstance(val, (list, tuple, np.ndarray)):
        xs = [int(x) for x in val]
    else:
        s = str(val).strip()
        if "," in s:
            xs = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
        else:
            xs = [int(s)]
    if len(xs) == 1:
        xs = xs * n
    if len(xs) != n:
        raise ValueError("%s has length %d, expected %d" % (name, len(xs), n))
    return xs


def _shannon_entropy(keys):
    if not keys:
        return 0.0
    values, counts = np.unique(keys, return_counts=True)
    del values
    p = counts.astype(np.float64) / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def make_lbf_env(kwargs):
    """Build the raw ForagingEnv. Penalty is subtracted on a failed LOAD."""
    grid = int(kwargs["grid_size"])
    penalty = float(kwargs["penalty"])
    if penalty < 0.0:
        raise ValueError("penalty must be >= 0, got %r" % penalty)
    if penalty != 0.0:
        print("WARNING: failed-load penalty=%s (RO trap is possible)" % penalty)
    sight = int(kwargs["sight"])
    food_sight = int(kwargs.get("food_sight", -1))
    agent_sight = int(kwargs.get("agent_sight", -1))
    if food_sight < 0:
        food_sight = sight
    if agent_sight < 0:
        agent_sight = sight
    if food_sight < 0 or agent_sight < 0:
        raise ValueError("food_sight and agent_sight must be >= 0")
    if food_sight > grid:
        raise ValueError("food_sight (%r) > grid_size (%r)" % (food_sight, grid))
    if agent_sight > grid:
        raise ValueError("agent_sight (%r) > grid_size (%r)" % (agent_sight, grid))
    if sight < 0 or sight > grid:
        raise ValueError("sight (%r) must be in [0, grid_size=%r]" % (sight, grid))
    n_agents = int(kwargs["n_agents"])
    roster = _parse_int_list(kwargs.get("player_levels", ""), n_agents, "player_levels")
    pin = int(kwargs.get("pin_player_levels", -1))
    if pin < 0:
        pin = 1 if roster is not None else 0
    pin = bool(int(pin))
    if roster is not None:
        min_player = roster
        max_player = roster
    else:
        min_player = int(kwargs["min_player_level"])
        max_player = int(kwargs["max_player_level"])
        if pin:
            raise ValueError(
                "pin_player_levels=1 requires --player_levels so the roster "
                "is explicit, not a sampled range"
            )
    n_foods = int(kwargs["n_foods"])
    food_roster = _parse_int_list(
        kwargs.get("food_levels", ""), n_foods, "food_levels"
    )
    if food_roster is not None:
        min_food = food_roster
        max_food = food_roster
    else:
        min_food = int(kwargs["min_food_level"])
        max_food = kwargs["max_food_level"]
        max_food = None if max_food is None or int(max_food) <= 0 else int(max_food)
    exact_raw = kwargs["exact_match"]
    try:
        exact = int(exact_raw)
    except (TypeError, ValueError):
        raise ValueError("exact_match must be 0 or 1, got %r" % (exact_raw,))
    if exact not in (0, 1):
        raise ValueError("exact_match must be 0 or 1, got %r" % (exact_raw,))
    return ForagingEnv(
        players=n_agents,
        min_player_level=min_player,
        max_player_level=max_player,
        min_food_level=min_food,
        max_food_level=max_food,
        field_size=(grid, grid),
        max_num_food=int(kwargs["n_foods"]),
        sight=sight,
        max_episode_steps=int(kwargs["max_episode_steps"]),
        force_coop=bool(int(kwargs["force_coop"])),
        normalize_reward=bool(int(kwargs["normalize_reward"])),
        grid_observation=False,
        observe_agent_levels=True,
        penalty=penalty,
        shuffle_player_levels=not pin,
        food_sight=food_sight,
        agent_sight=agent_sight,
        exact_match=bool(exact),
    )


def _food_buckets(field):
    levels = field[field > 0].astype(np.int64)
    n1 = int(np.sum(levels == 1))
    n2 = int(np.sum(levels == 2))
    n3 = int(np.sum(levels == 3))
    n4 = int(np.sum(levels == 4))
    n3p = int(np.sum(levels >= 3))
    return n1, n2, n3, n4, n3p


def _is_split(env):
    """1 if ≥2 agents sit at different understaffed foods this timestep."""
    food_idx = np.argwhere(env.field > 0)
    needy = []
    for r, c in food_idx:
        level = int(env.field[r, c])
        adj = env.adjacent_players(int(r), int(c))
        if len(adj) >= 1 and len(adj) < level:
            needy.append(adj)
    if len(needy) < 2:
        return 0.0
    ids = {id(p) for group in needy for p in group}
    return 1.0 if len(ids) >= 2 else 0.0


def _slot_visible(slot):
    """True if this food/agent slot is visible.

    3-d gym triple: y >= 0. 4-d actor slot: last entry is an explicit vis bit.
    """
    slot = np.asarray(slot)
    if slot.shape[-1] >= 4:
        return float(slot[..., 3]) >= 0.5
    return float(slot[..., 0]) >= 0.0


def _slot_width(obs, n_slots):
    dim = int(np.asarray(obs).shape[-1])
    if dim == 4 * n_slots:
        return 4
    if dim == 3 * n_slots:
        return 3
    raise ValueError(
        "obs last dim %d is not 3*%d or 4*%d" % (dim, n_slots, n_slots)
    )


def _n_vis_foods(obs, n_foods, n_slots=None):
    obs = np.asarray(obs)
    if n_slots is None:
        if int(obs.shape[-1]) % 3 != 0:
            raise ValueError(
                "n_slots required for non-triple obs dim %d" % obs.shape[-1]
            )
        width = 3
    else:
        width = _slot_width(obs, n_slots)
    return int(
        sum(1 for i in range(n_foods) if _slot_visible(obs[width * i : width * i + width]))
    )


def _add_slot_visibility(raw, n_slots):
    """Replace gym (-1, ·, ·) with (y, x, lvl, vis). Invisible -> (0,0,0,0)."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[-1] != 3 * n_slots:
        raise ValueError(
            "expected gym triples dim %d, got %d" % (3 * n_slots, raw.shape[-1])
        )
    trips = raw.reshape(raw.shape[0], n_slots, 3)
    vis = trips[..., 0] >= 0.0
    out = np.zeros((raw.shape[0], n_slots, 4), dtype=np.float32)
    out[..., :3] = np.where(vis[..., None], trips, 0.0)
    out[..., 3] = vis.astype(np.float32)
    return out.reshape(raw.shape[0], 4 * n_slots)


def _connected_components(adj):
    n = int(adj.shape[0])
    seen = [False] * n
    comps = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            j = stack.pop()
            comp.append(j)
            for k in range(n):
                if (not seen[k]) and adj[j, k] > 0.5:
                    seen[k] = True
                    stack.append(k)
        comps.append(sorted(comp))
    return comps


def _chebyshev(pos_a, pos_b):
    return max(abs(int(pos_a[0]) - int(pos_b[0])), abs(int(pos_a[1]) - int(pos_b[1])))


def _tightest_k(ids, positions, k):
    """k members of ids with the smallest pairwise Chebyshev sum. Tie: IDs."""
    from itertools import combinations

    best = None
    for cand in combinations(sorted(int(i) for i in ids), k):
        cost = 0
        for a, b in combinations(cand, 2):
            cost += _chebyshev(positions[a], positions[b])
        key = (cost, cand)
        if best is None or key < best:
            best = key
    return list(best[1])


def _cap_neighborhoods(adj, max_group_size, positions=None, mode="id"):
    """Safety bound on 1-hop neighborhood size. Rows may overlap / be directed.

    Each agent keeps itself plus at most (max_group_size-1) neighbors.
    id=lowest neighbor IDs. spatial=nearest Chebyshev, ID tiebreak.
    max_group_size < 1 means no cap (return adj).
    """
    adj = np.asarray(adj, dtype=np.float32)
    n = int(adj.shape[0])
    if max_group_size < 1:
        out = adj.copy()
        np.fill_diagonal(out, 1.0)
        return out
    if mode not in ("id", "spatial"):
        raise ValueError("group cap mode must be id or spatial, got %r" % (mode,))
    if mode == "spatial" and positions is None:
        raise ValueError("spatial cap requires positions")
    out = np.eye(n, dtype=np.float32)
    k_other = max(int(max_group_size) - 1, 0)
    for i in range(n):
        nbrs = [j for j in range(n) if j != i and adj[i, j] > 0.5]
        if mode == "spatial":
            nbrs.sort(key=lambda j: (_chebyshev(positions[i], positions[j]), j))
        else:
            nbrs.sort()
        for j in nbrs[:k_other]:
            out[i, j] = 1.0
    return out


def _cap_groups(comps, max_group_size, positions=None, mode="id"):
    if max_group_size <= 0:
        return [list(c) for c in comps]
    if mode not in ("id", "spatial"):
        raise ValueError("group cap mode must be id or spatial, got %r" % (mode,))
    if mode == "spatial" and positions is None:
        raise ValueError("spatial cap requires positions")
    groups = []
    for comp in comps:
        ids = [int(i) for i in comp]
        if mode == "id":
            ids = sorted(ids)
            if len(ids) <= max_group_size:
                groups.append(ids)
                continue
            for start in range(0, len(ids), max_group_size):
                groups.append(ids[start : start + max_group_size])
            continue
        remaining = list(ids)
        while len(remaining) > max_group_size:
            tight = _tightest_k(remaining, positions, max_group_size)
            groups.append(sorted(tight))
            taken = set(tight)
            remaining = [i for i in remaining if i not in taken]
        if remaining:
            groups.append(sorted(remaining))
    return groups


def _merge_group_obs(raw_obs, groups, n_foods, n_agents):
    """Slot-wise union. Slot i stays slot i (global entity ID).

    If any member of a group can see a food/agent slot, every member's
    merged obs gets that same slot. Coordinates are global, so this is a
    copy, not a frame transform. Works on gym triples or 4-d vis-bit slots.
    """
    merged = raw_obs.copy()
    n_slots = n_foods + n_agents
    width = _slot_width(raw_obs, n_slots)
    for group in groups:
        if len(group) <= 1:
            continue
        for slot in range(n_slots):
            base = width * slot
            chosen = None
            for i in group:
                trip = raw_obs[i, base : base + width]
                if _slot_visible(trip):
                    payload = trip[:3]
                    if chosen is None:
                        chosen = trip.copy()
                    elif not np.allclose(chosen[:3], payload):
                        raise ValueError(
                            "global slot %d disagrees inside group %s: %s vs %s"
                            % (slot, group, chosen[:3].tolist(), payload.tolist())
                        )
            if chosen is not None:
                for i in group:
                    merged[i, base : base + width] = chosen
        for slot in range(n_slots):
            base = width * slot
            if not any(
                _slot_visible(raw_obs[i, base : base + width]) for i in group
            ):
                continue
            for i in group:
                if not _slot_visible(merged[i, base : base + width]):
                    raise RuntimeError(
                        "group %s: slot %d visible to a member but missing "
                        "on agent %d after merge"
                        % (group, slot, i)
                    )
    return merged


def _n_foods_with_adj(env):
    n = 0
    for r, c in np.argwhere(env.field > 0):
        if env.adjacent_players(int(r), int(c)):
            n += 1
    return n


def _parallel_collect(times, window=3):
    """1 if any two collection steps fall within `window` timesteps."""
    ts = sorted(int(t) for t in times)
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            if ts[j] - ts[i] <= window:
                return 1.0
    return 0.0


def _load_attempt_stats(env, actions):
    """LOAD attempts among agents adjacent to food, grouped by that food.

    Adjacent means loaders only — same filter as ForagingEnv.step.
    Overshoot/undershoot are about the summed loader level vs the food,
    independent of whether exact_match turns overshoot into a failed load.
    """
    n_load = 0
    n_fail = 0
    n_over = 0
    n_under = 0
    remaining = []
    for player, action in zip(env.players, actions):
        if int(action) != Action.LOAD.value:
            continue
        if env.adjacent_food(*player.position) <= 0:
            continue
        n_load += 1
        remaining.append(player)
    remaining = set(remaining)
    exact = bool(env.exact_match)
    while remaining:
        player = remaining.pop()
        loc = env.adjacent_food_location(*player.position)
        if loc is None:
            continue
        frow, fcol = loc
        food = int(env.field[frow, fcol])
        adj = [
            p
            for p in env.adjacent_players(frow, fcol)
            if p is player or p in remaining
        ]
        remaining -= set(adj)
        s = sum(p.level for p in adj)
        n = len(adj)
        if s > food:
            n_over += n
        elif s < food:
            n_under += n
        loaded = s == food if exact else s >= food
        if not loaded:
            n_fail += n
    return n_load, n_fail, n_over, n_under


class LBFMATWrapper:
    """Level-Based Foraging exposed through MAT's SMAC-style env API."""

    def __init__(self, args):
        self._kwargs = env_kwargs_from(args)
        self._env = make_lbf_env(self._kwargs)
        self.n_agents = self._env.n_agents
        raw_fil = getattr(args, "food_indexed_load", 0)
        try:
            fil = int(raw_fil)
        except (TypeError, ValueError):
            raise ValueError(
                "food_indexed_load must be 0 or 1, got %r" % (raw_fil,)
            )
        if fil not in (0, 1):
            raise ValueError("food_indexed_load must be 0 or 1, got %r" % (raw_fil,))
        self.food_indexed_load = bool(fil)
        self._n_foods = int(self._env.max_num_food)
        if self.food_indexed_load:
            self.n_actions = N_MOVE + self._n_foods
        else:
            self.n_actions = N_MOVE + 1
        self.episode_limit = int(self._kwargs["max_episode_steps"])

        gym_obs_space = self._env.observation_space[0]
        self._n_slots = self._n_foods + self.n_agents
        self._slot_width = 3
        self.obs_dim = int(np.prod(gym_obs_space.shape))
        if self.obs_dim != 3 * self._n_slots:
            raise ValueError(
                "gym obs dim %d != 3*(n_foods+n_agents)=%d"
                % (self.obs_dim, 3 * self._n_slots)
            )
        self.state_dim = self.obs_dim * self.n_agents
        self.algorithm_name = getattr(args, "algorithm_name", "")
        self._append_pairwise_comm = self.algorithm_name in COMM_ALGOS
        self.comm_radius = int(getattr(args, "comm_radius", -1))
        self.group_obs = bool(int(getattr(args, "group_obs", 0)))
        self.max_group_size = int(getattr(args, "max_group_size", 0))
        self.group_cap_mode = str(getattr(args, "group_cap_mode", "id"))
        if self.group_cap_mode not in ("id", "spatial"):
            raise ValueError(
                "group_cap_mode must be id or spatial, got %r" % self.group_cap_mode
            )
        if self.group_obs:
            if self.comm_radius < 0:
                raise ValueError(
                    "group_obs=1 requires --comm_radius >= 0, got %r"
                    % self.comm_radius
                )
            if self.max_group_size < 1:
                raise ValueError(
                    "group_obs=1 requires --max_group_size >= 1, got %r"
                    % self.max_group_size
                )
            # Secondary merge arm: explicit vis bit, no (-1) sentinel in actor obs.
            self._slot_width = 4
            self.obs_dim = 4 * self._n_slots
        raw_c = getattr(args, "commitment", 0)
        try:
            cflag = int(raw_c)
        except (TypeError, ValueError):
            raise ValueError("commitment must be 0 or 1, got %r" % (raw_c,))
        if cflag not in (0, 1):
            raise ValueError("commitment must be 0 or 1, got %r" % (raw_c,))
        self.use_commitment = bool(cflag)
        if self.use_commitment:
            if self.algorithm_name != "mat_comm":
                raise ValueError("commitment=1 is mat_comm only, got %r" % self.algorithm_name)
            n_tgt = int(getattr(args, "n_commit_targets", 0))
            if n_tgt != self._n_foods:
                raise ValueError(
                    "n_commit_targets (%r) must equal n_foods (%r)"
                    % (n_tgt, self._n_foods)
                )
            from mat.algorithms.utils.mat_comm_commit import pack_size
            self.state_dim = self.state_dim + pack_size(self.n_agents, self._n_foods)
        if self._append_pairwise_comm:
            kind = getattr(args, "comm_mask_kind", None)
            if kind != "pairwise":
                raise ValueError(
                    "LBF mat_comm appends an N×N Chebyshev adjacency to "
                    "share_obs; set --comm_mask_kind pairwise. Refusing bits: "
                    "the last n_agents entries of concat obs are not comm bits."
                )
            if self.comm_radius < 0:
                raise ValueError(
                    "LBF mat_comm requires --comm_radius >= 0, got %r"
                    % self.comm_radius
                )
            self.state_dim = self.state_dim + self.n_agents * self.n_agents
        self.step_cost = float(getattr(args, "step_cost", ENV_ARG_DEFAULTS["step_cost"]))
        if self.step_cost < 0.0:
            raise ValueError("step_cost must be >= 0, got %r" % self.step_cost)

        # Group action memory: appends the group's own joint action from t-1
        # to every member's actor obs. n_agents * n_actions one-hot (zero for
        # agents outside the group), zeroed at episode start and whenever the
        # group's exact member set changed since its last recorded action.
        # Folded into `obs`, not `share_obs`: for mat_comm this reaches the
        # decoder indirectly through the encoder's obs_rep (attn2's query),
        # the same channel the group_obs union already uses; for MAPPO's
        # dec_actor MLP it is read directly since that path consumes `obs`.
        # It is plain float data replayed from the buffer, never a tensor
        # produced by this step's own forward pass, so there is nothing to
        # detach.
        self.group_action_memory = bool(int(getattr(args, "group_action_memory", 0)))
        if self.group_action_memory:
            if not self.group_obs or self.max_group_size < 1:
                raise ValueError(
                    "group_action_memory requires group_obs=1 and "
                    "max_group_size >= 1"
                )
            self._memory_dim = self.n_agents * self.n_actions
            self.obs_dim = self.obs_dim + self._memory_dim
        else:
            self._memory_dim = 0
        self._group_memory = np.zeros(
            (self.n_agents, self._memory_dim), dtype=np.float32
        )
        self._last_groups = None
        self._memory_reset_count = 0
        self._memory_total_count = 0

        self._collect_return = 0.0
        print(
            "[LBFMATWrapper] algo=%s obs_dim=%d share_obs_dim=%d n_actions=%d "
            "food_indexed_load=%s append_pairwise_comm=%s comm_radius=%s "
            "group_obs=%s max_group_size=%s cap_mode=%s step_cost=%s "
            "group_action_memory=%s exact_match=%s commitment=%s"
            % (
                self.algorithm_name,
                self.obs_dim,
                self.state_dim,
                self.n_actions,
                int(self.food_indexed_load),
                self._append_pairwise_comm,
                self.comm_radius if (self._append_pairwise_comm or self.group_obs) else None,
                self.group_obs,
                self.max_group_size if self.max_group_size >= 1 else None,
                self.group_cap_mode if self.max_group_size >= 1 else None,
                self.step_cost,
                self.group_action_memory,
                int(self._env.exact_match),
                int(self.use_commitment),
            ),
            flush=True,
        )

        if self.group_obs or self.group_action_memory:
            self.observation_space = [
                Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
                for _ in range(self.n_agents)
            ]
        else:
            self.observation_space = [
                Box(
                    low=gym_obs_space.low,
                    high=gym_obs_space.high,
                    shape=(self.obs_dim,),
                    dtype=np.float32,
                )
                for _ in range(self.n_agents)
            ]
        self.share_observation_space = [
            Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.state_dim,),
                dtype=np.float32,
            )
            for _ in range(self.n_agents)
        ]
        self.action_space = [
            Discrete(self.n_actions) for _ in range(self.n_agents)
        ]
        if self.group_obs:
            print(
                "[LBFMATWrapper] group_obs=1 SECONDARY merge arm: slot-wise "
                "union plus per-slot vis bit (y,x,lvl,vis). Invisible is "
                "(0,0,0,0), not (-1,·,·). Primary CT-SDE is group_obs=0.",
                flush=True,
            )
        self._spawned = (0, 0, 0, 0, 0)
        self._n_coop_foods = 0
        self._collected = [0, 0, 0, 0, 0]
        self._split_steps = 0
        self._n_steps = 0
        self._n_load = 0
        self._n_fail = 0
        self._n_overshoot = 0
        self._n_undershoot = 0
        self._collect_times = []
        self._adj_by_pos = {}
        self._collections = []
        self._productive_adj = 0
        self._collected_level_sum = 0
        self._t_first_adj = None
        self._parallel_steps = 0
        self._group_size_sum = 0.0
        self._group_frac = [0, 0, 0, 0]
        self._foods_vis_pre_sum = 0.0
        self._foods_vis_post_sum = 0.0
        self._group_metric_steps = 0
        self._memory_reset_count = 0
        self._memory_total_count = 0

    def _reset_episode_stats(self):
        n1, n2, n3, n4, n3p = _food_buckets(self._env.field)
        max_player = max(p.level for p in self._env.players) if self._env.players else 0
        levels = self._env.field[self._env.field > 0]
        n_coop = int(np.sum(levels > max_player))
        self._spawned = (n1, n2, n3, n4, n3p)
        self._n_coop_foods = n_coop
        self._collected = [0, 0, 0, 0, 0]
        self._split_steps = 0
        self._n_steps = 0
        self._n_load = 0
        self._n_fail = 0
        self._n_overshoot = 0
        self._n_undershoot = 0
        self._collect_times = []
        self._adj_by_pos = {}
        self._collections = []
        self._productive_adj = 0
        self._collected_level_sum = 0
        self._t_first_adj = 0.0 if _n_foods_with_adj(self._env) >= 1 else None
        self._parallel_steps = 0
        self._group_size_sum = 0.0
        self._group_frac = [0, 0, 0, 0]
        self._foods_vis_pre_sum = 0.0
        self._foods_vis_post_sum = 0.0
        self._group_metric_steps = 0
        self._memory_reset_count = 0
        self._memory_total_count = 0
        self._collect_return = 0.0
        self._food_visitors = [set() for _ in range(self._n_foods)]
        self._food_max_adj = np.zeros(self._n_foods, dtype=np.int64)
        self._food_pair_steps = np.zeros(self._n_foods, dtype=np.int64)
        self._food_collected = np.zeros(self._n_foods, dtype=np.int64)
        self._slot_pos = {}
        slots = self._env.food_slots
        if slots is not None:
            for k, s in enumerate(slots):
                if s is None:
                    continue
                self._slot_pos[(int(s[0]), int(s[1]))] = k
        self._note_group_stats()

    def _gym_obs(self):
        obs = self._env._make_gym_obs()
        return np.stack(obs).astype(np.float32)

    def _radius_adj(self):
        """Raw Chebyshev adjacency. Diagonal on. No size cap."""
        pos = np.asarray(
            [p.position for p in self._env.players], dtype=np.int32
        )
        delta = np.abs(pos[:, None, :] - pos[None, :, :])
        dist = delta.max(-1)
        adj = dist <= self.comm_radius
        np.fill_diagonal(adj, True)
        return adj.astype(np.float32)

    def _positions(self):
        return np.asarray([p.position for p in self._env.players], dtype=np.int32)

    def _groups(self):
        if self.comm_radius < 0:
            return [[i] for i in range(self.n_agents)]
        return _cap_groups(
            _connected_components(self._radius_adj()),
            self.max_group_size,
            positions=self._positions(),
            mode=self.group_cap_mode,
        )

    def _obs(self):
        raw = self._gym_obs()
        obs = raw
        if self.group_obs:
            raw = _add_slot_visibility(raw, self._n_slots)
            obs = _merge_group_obs(
                raw, self._groups(), self._env.max_num_food, self.n_agents
            )
        if self.group_action_memory:
            obs = np.concatenate([obs, self._group_memory], axis=1)
        return obs

    def _update_group_memory(self, groups_prev, actions_prev, groups_next):
        """Carry each group's joint action forward if its member set is unchanged.

        `groups_prev` is the partition that produced `actions_prev` (captured
        when the obs those actions were sampled from was built). `groups_next`
        is the partition for the obs about to be returned. An agent's memory
        resets to zero if its group at `groups_next` is not the exact same
        member set it had in `groups_prev`, or if that prior group had no
        recorded joint action (episode start).
        """
        prev_membership = {}
        prev_joint = {}
        for g in groups_prev:
            gs = frozenset(g)
            for i in g:
                prev_membership[i] = gs
            if len(g) >= 2:
                vec = np.zeros(self._memory_dim, dtype=np.float32)
                for i in g:
                    vec[i * self.n_actions + int(actions_prev[i])] = 1.0
                prev_joint[gs] = vec

        memory = np.zeros((self.n_agents, self._memory_dim), dtype=np.float32)
        for g in groups_next:
            if len(g) < 2:
                continue
            gs = frozenset(g)
            for i in g:
                self._memory_total_count += 1
                if prev_membership.get(i) == gs and gs in prev_joint:
                    memory[i] = prev_joint[gs]
                else:
                    self._memory_reset_count += 1
        self._group_memory = memory

    def get_comm_mask(self):
        """Raw Chebyshev adjacency for mat_comm. No component cliques.

        max_group_size is a per-row neighborhood cap (safety), not a
        partition. Diagonal stays on.
        """
        adj = self._radius_adj()
        if self.max_group_size < 1:
            return adj
        return _cap_neighborhoods(
            adj,
            self.max_group_size,
            positions=self._positions(),
            mode=self.group_cap_mode,
        )

    def _note_group_stats(self):
        raw = self._gym_obs()
        n_foods = int(self._env.max_num_food)
        n_slots = n_foods + self.n_agents
        pre = [_n_vis_foods(raw[i], n_foods, n_slots) for i in range(self.n_agents)]
        self._foods_vis_pre_sum += float(np.mean(pre))
        groups = self._groups()
        sizes = np.zeros(self.n_agents, dtype=np.int64)
        post = []
        for g in groups:
            for i in g:
                sizes[i] = len(g)
            if self.group_obs:
                merged_one = _merge_group_obs(
                    raw, [g], n_foods, self.n_agents
                )[g[0]]
            else:
                merged_one = raw[g[0]]
            post.append(_n_vis_foods(merged_one, n_foods, n_slots))
        self._foods_vis_post_sum += float(np.mean(post)) if post else 0.0
        self._group_size_sum += float(sizes.mean())
        for k in (1, 2, 3, 4):
            self._group_frac[k - 1] += float(np.mean(sizes == k))
        self._group_metric_steps += 1

    def _commitment_pack(self):
        """Visible food slots from raw gym triples. Positions in grid cells."""
        from mat.algorithms.utils.mat_comm_commit import pack_commitment

        n_t = self._n_foods
        agent_xy = self._positions().astype(np.float32)
        target_xy = np.zeros((n_t, 2), dtype=np.float32)
        alive = np.zeros(n_t, dtype=np.float32)
        raw = self._gym_obs()
        width = 3
        vis = np.zeros((self.n_agents, n_t), dtype=np.float32)
        for k in range(n_t):
            yx = self._living_slot_yx(k)
            if yx is not None:
                target_xy[k, 0] = float(yx[0])
                target_xy[k, 1] = float(yx[1])
                alive[k] = 1.0
            for i in range(self.n_agents):
                slot = raw[i, width * k : width * k + width]
                if _slot_visible(slot):
                    vis[i, k] = 1.0
        return pack_commitment(
            agent_xy, target_xy, alive, vis, int(self._kwargs["grid_size"])
        )

    def _share_obs(self):
        obs = self._gym_obs()
        flat = obs.reshape(-1)
        if self.use_commitment:
            flat = np.concatenate([flat, self._commitment_pack()], axis=0)
        if self._append_pairwise_comm:
            mask = self.get_comm_mask().reshape(-1)
            n2 = self.n_agents * self.n_agents
            if mask.size != n2:
                raise ValueError(
                    "get_comm_mask flattened size %d != n_agents**2=%d"
                    % (mask.size, n2)
                )
            flat = np.concatenate([flat, mask], axis=0)
        return np.tile(flat, (self.n_agents, 1)).astype(np.float32)

    def _living_slot_yx(self, k):
        slots = self._env.food_slots
        if slots is None or k < 0 or k >= len(slots):
            return None
        slot = slots[k]
        if slot is None:
            return None
        y, x = int(slot[0]), int(slot[1])
        if int(self._env.field[y, x]) <= 0:
            return None
        return (y, x)

    def _is_load_k_adjacent(self, player, k):
        yx = self._living_slot_yx(k)
        if yx is None:
            return False
        py, px = player.position
        return abs(int(py) - yx[0]) + abs(int(px) - yx[1]) == 1

    def _gym_would_load_slot(self, player, k):
        """True iff gym LOAD would hit food slot k (NESW first adjacent)."""
        if not self._is_load_k_adjacent(player, k):
            return False
        loc = self._env.adjacent_food_location(*player.position)
        yx = self._living_slot_yx(k)
        if loc is None or yx is None:
            return False
        return (int(loc[0]), int(loc[1])) == yx

    def _decode_to_gym(self, actions):
        """Map wrapper actions onto gym {NONE..EAST, LOAD}.

        LOAD[k] becomes gym LOAD only if the agent is 4-adjacent to slot k
        *and* LBF's NESW pick is that cell. Otherwise NONE — never load a
        different food than the one named in the action.
        """
        actions = np.asarray(actions).reshape(-1).astype(int)
        if actions.shape[0] != self.n_agents:
            raise ValueError(
                "expected %d actions, got %r" % (self.n_agents, actions.shape)
            )
        for a in actions:
            if int(a) < 0 or int(a) >= self.n_actions:
                raise ValueError(
                    "action %d out of range [0, %d)" % (int(a), self.n_actions)
                )
        if not self.food_indexed_load:
            return actions
        gym_a = np.empty(self.n_agents, dtype=int)
        for i, a in enumerate(actions):
            a = int(a)
            if a < N_MOVE:
                gym_a[i] = a
                continue
            k = a - N_MOVE
            if self._gym_would_load_slot(self._env.players[i], k):
                gym_a[i] = Action.LOAD.value
            else:
                gym_a[i] = Action.NONE.value
        return gym_a

    def _avail(self):
        masks = np.zeros((self.n_agents, self.n_actions), dtype=np.int64)
        valid = self._env._valid_actions
        if valid is None:
            raise RuntimeError("avail requested before LBF _gen_valid_moves")
        if not self.food_indexed_load:
            for i, player in enumerate(self._env.players):
                for action in valid[player]:
                    masks[i, action.value] = 1
            return masks
        for i, player in enumerate(self._env.players):
            for action in valid[player]:
                if action.value < N_MOVE:
                    masks[i, action.value] = 1
            for k in range(self._n_foods):
                if self._is_load_k_adjacent(player, k):
                    masks[i, N_MOVE + k] = 1
        return masks

    def _coalition_diag(self):
        keys = [",".join(str(i) for i in c["agents"]) for c in self._collections]
        n_col = len(self._collections)
        leftover = [
            (int(r), int(c)) for r, c in np.argwhere(self._env.field > 0)
        ]
        wasted = float(sum(self._adj_by_pos.get(pos, 0) for pos in leftover))
        if self._productive_adj > 0:
            alloc = float(self._collected_level_sum) / float(self._productive_adj)
        else:
            alloc = 0.0
        n_c31 = sum(1 for c in self._collections if c["levels"] == (1, 3))
        n_c112 = sum(1 for c in self._collections if c["levels"] == (1, 1, 2))
        n_c22 = sum(1 for c in self._collections if c["levels"] == (2, 2))
        return {
            "coalition_entropy": _shannon_entropy(keys),
            "n_coalition_types": float(len(set(keys))),
            "n_collected": float(n_col),
            "wasted_adjacency": wasted,
            "alloc_efficiency": alloc,
            "frac_c31": float(n_c31 / n_col) if n_col else 0.0,
            "frac_c112": float(n_c112 / n_col) if n_col else 0.0,
            "frac_c22": float(n_c22 / n_col) if n_col else 0.0,
        }

    def _diag(self):
        n_load = self._n_load
        fail_rate = (self._n_fail / n_load) if n_load > 0 else 0.0
        solved = float(self._env.field.sum() == 0)
        out = {
            "solved": solved,
            "ep_length": float(self._n_steps),
            "collected_lvl1": float(self._collected[0]),
            "collected_lvl2": float(self._collected[1]),
            "collected_lvl3": float(self._collected[2]),
            "collected_lvl4": float(self._collected[3]),
            "collected_lvl3plus": float(self._collected[4]),
            "spawned_lvl1": float(self._spawned[0]),
            "spawned_lvl2": float(self._spawned[1]),
            "spawned_lvl3": float(self._spawned[2]),
            "spawned_lvl4": float(self._spawned[3]),
            "spawned_lvl3plus": float(self._spawned[4]),
            "n_coop_foods": float(self._n_coop_foods),
            "split_rate": float(self._split_steps / max(self._n_steps, 1)),
            "failed_load_rate": float(fail_rate),
            "n_load_actions": float(self._n_load),
            "n_failed_loads": float(self._n_fail),
            "load_overshoot_rate": float(
                (self._n_overshoot / n_load) if n_load > 0 else 0.0
            ),
            "load_undershoot_rate": float(
                (self._n_undershoot / n_load) if n_load > 0 else 0.0
            ),
            "n_load_overshoot": float(self._n_overshoot),
            "n_load_undershoot": float(self._n_undershoot),
            "parallel_collect": float(_parallel_collect(self._collect_times)),
            "t_first_adj": float(
                self._t_first_adj
                if self._t_first_adj is not None
                else self.episode_limit
            ),
            "parallel_work_rate": float(self._parallel_steps / max(self._n_steps, 1)),
            "group_size_mean": float(
                self._group_size_sum / max(self._group_metric_steps, 1)
            ),
            "group_frac_1": float(
                self._group_frac[0] / max(self._group_metric_steps, 1)
            ),
            "group_frac_2": float(
                self._group_frac[1] / max(self._group_metric_steps, 1)
            ),
            "group_frac_3": float(
                self._group_frac[2] / max(self._group_metric_steps, 1)
            ),
            "group_frac_4": float(
                self._group_frac[3] / max(self._group_metric_steps, 1)
            ),
            "foods_vis_pre": float(
                self._foods_vis_pre_sum / max(self._group_metric_steps, 1)
            ),
            "foods_vis_post": float(
                self._foods_vis_post_sum / max(self._group_metric_steps, 1)
            ),
            "collect_return": float(self._collect_return),
            "group_memory_reset_rate": float(
                self._memory_reset_count / max(self._memory_total_count, 1)
            ),
        }
        out.update(self._coalition_diag())
        out.update(self._food_outcome_diag())
        return out

    def _note_food_occupancy(self, y, x, agent_ids):
        k = self._slot_pos.get((int(y), int(x)))
        if k is None:
            return
        ids = [int(i) for i in agent_ids]
        self._food_visitors[k].update(ids)
        n = len(ids)
        if n > self._food_max_adj[k]:
            self._food_max_adj[k] = n
        if n >= 2:
            self._food_pair_steps[k] += 1

    def _food_outcome_diag(self):
        """Leftover-food modes. Distinguishes async / split / cluster.

        untouched: no agent ever adjacent.
        solo: exactly one agent ever adjacent, never two at once.
        async: two+ agents visited, never co-occupied.
        paired_fail: co-occupied at least once but still leftover.
        cluster: leftover (or any leftover food) with max simultaneous adj >= 3.
        """
        untouched = solo = async_ = paired_fail = cluster = 0
        left = 0
        for k in range(self._n_foods):
            if self._food_collected[k] > 0:
                continue
            yx = self._living_slot_yx(k)
            if yx is None:
                continue
            left += 1
            n_vis = len(self._food_visitors[k])
            mx = int(self._food_max_adj[k])
            pairs = int(self._food_pair_steps[k])
            if n_vis == 0:
                untouched += 1
            elif pairs == 0 and n_vis >= 2:
                async_ += 1
            elif pairs == 0:
                solo += 1
            else:
                paired_fail += 1
            if mx >= 3:
                cluster += 1
        return {
            "food_untouched": float(untouched),
            "food_solo": float(solo),
            "food_async": float(async_),
            "food_paired_fail": float(paired_fail),
            "food_cluster": float(cluster),
            "n_foods_left": float(left),
        }

    def _infos(self, bad_transition):
        diag = self._diag()
        info = dict(diag)
        info["bad_transition"] = bool(bad_transition)
        info["won"] = bool(diag["solved"] >= 1.0)
        return [dict(info) for _ in range(self.n_agents)]

    def reset(self):
        self._env.reset()
        self._reset_episode_stats()
        if self.group_action_memory:
            self._group_memory = np.zeros(
                (self.n_agents, self._memory_dim), dtype=np.float32
            )
            self._last_groups = self._groups()
        return self._obs(), self._share_obs(), self._avail()

    def step(self, actions):
        actions = np.asarray(actions).reshape(-1).astype(int)
        if actions.shape[0] != self.n_agents:
            raise ValueError(
                "expected %d actions, got %r" % (self.n_agents, actions.shape)
            )
        groups_prev = self._last_groups
        gym_actions = self._decode_to_gym(actions)

        field_before = self._env.field.copy()
        self._split_steps += _is_split(self._env)
        n_load, n_fail, n_over, n_under = _load_attempt_stats(
            self._env, gym_actions
        )
        self._n_load += n_load
        self._n_fail += n_fail
        self._n_overshoot += n_over
        self._n_undershoot += n_under

        obs, rewards, done, truncated, _ = self._env.step(gym_actions)
        del obs
        self._n_steps += 1

        disappeared = (field_before > 0) & (self._env.field == 0)
        n_gone = int(np.sum(disappeared))
        if n_gone > 0:
            self._collect_times.extend([int(self._n_steps)] * n_gone)
        for level in field_before[disappeared]:
            lvl = int(level)
            if lvl == 1:
                self._collected[0] += 1
            elif lvl == 2:
                self._collected[1] += 1
            elif lvl == 3:
                self._collected[2] += 1
                self._collected[4] += 1
            elif lvl == 4:
                self._collected[3] += 1
                self._collected[4] += 1
            else:
                self._collected[4] += 1

        n_adj_foods = _n_foods_with_adj(self._env)
        if self._t_first_adj is None and n_adj_foods >= 1:
            self._t_first_adj = float(self._n_steps)
        if n_adj_foods >= 2:
            self._parallel_steps += 1
        self._note_group_stats()
        if self.group_action_memory:
            groups_next = self._groups()
            self._update_group_memory(groups_prev, actions, groups_next)
            self._last_groups = groups_next

        for r, c in np.argwhere(self._env.field > 0):
            pos = (int(r), int(c))
            adj = self._env.adjacent_players(int(r), int(c))
            n_adj = len(adj)
            self._adj_by_pos[pos] = self._adj_by_pos.get(pos, 0) + n_adj
            ids = [i for i, p in enumerate(self._env.players) if p in adj]
            self._note_food_occupancy(r, c, ids)
        for col in self._env.last_collections:
            pos = col["pos"]
            n_now = len(col["agents"])
            self._adj_by_pos[pos] = self._adj_by_pos.get(pos, 0) + n_now
            self._collections.append(col)
            self._productive_adj += self._adj_by_pos[pos]
            self._collected_level_sum += int(col["level"])
            k = self._slot_pos.get((int(pos[0]), int(pos[1])))
            if k is not None:
                self._food_collected[k] = 1
                ids = []
                for a in col["agents"]:
                    if isinstance(a, (int, np.integer)):
                        ids.append(int(a))
                    else:
                        for i, p in enumerate(self._env.players):
                            if p is a:
                                ids.append(i)
                                break
                self._note_food_occupancy(pos[0], pos[1], ids)

        team_collect = float(np.sum(rewards))
        self._collect_return += team_collect
        team_r = team_collect - self.step_cost
        done = bool(done or truncated)
        hit_horizon = self._env.current_step >= self.episode_limit
        solved = self._env.field.sum() == 0
        bad_transition = bool(hit_horizon and not solved)

        rewards_out = np.full((self.n_agents, 1), team_r, dtype=np.float32)
        dones = np.full(self.n_agents, done, dtype=bool)
        infos = self._infos(bad_transition)
        return self._obs(), self._share_obs(), rewards_out, dones, infos, self._avail()

    def seed(self, seed=None):
        self._env.seed(seed)

    def close(self):
        self._env.close()
