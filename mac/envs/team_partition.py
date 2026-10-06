"""Team Partition: regroup, then split across zones to match a quota exactly.

`n_agents` interchangeable agents spawn on two rings around a staging cell --
`near_wave_size` of them `spawn_distance` cells away, the rest `wave_gap` cells
further out -- and fly in with MOVE. Once `commit_quorum` agents stand on the
staging cell (by default the quota total, n_agents - n_spare) the GATE opens,
and the agents present must split across `n_zones` coloured zones so that zone
c receives exactly K_c of them. The quota K (every K_c >= 1) is public and is
redrawn every episode. The team has `commit_window` rounds to act once the gate
opens, and the gate must open by `gate_deadline`; otherwise the episode ends
as a miss. Which agents end up deciding is the environment's draw, so an
id-keyed lookup table cannot memorise the split.

Observation (per agent)
    quota one-hot (n_zones x possible quota values), own at_staging bit, own
    comm bit (on the staging cell with at least one other agent). At the
    decision every present agent's observation is identical; `get_obs` asserts
    it (symmetry_assert).
State (centralised critic)
    quota one-hot, every agent's distance to the staging cell, round counter.
Actions
    0 STAY, 1..n_zones GO_c (inert until the gate is open), n_zones+1 MOVE (one
    step toward the staging cell; the path is not learned).
Reward (shared)
    the split, ending the episode:
        +zone_reward per zone whose count matches its quota
        +all_reward if every zone matches, else -wrong_penalty
    each other round: a step cost from step_cost (on the staging cell) to
    step_cost_far (far edge), scaled by the team's mean distance; -dodge_penalty
    if the gate is open and nobody goes; +shaping_weight per cell of progress
    toward the staging cell (potential-based)
    a miss (window expired, gate never opened, or time limit): -miss_penalty
Info
    is_success / exact_match (the success metric), abs_error, n_committed,
    n_present, blocked_commits, analytic_ceiling (best symmetric success for
    this quota), coordinated_cap, and further per-episode diagnostics.

Paper configuration (mac/presets.py): 9 agents (6 near at distance 2, 3 far at
distance 5), 3 zones, quota summing to 6 (3 spares), gate quorum 6 by round 4,
commit window 1, zone_reward = all_reward = 10, wrong_penalty 20, miss_penalty
25, time_limit 20. Rule-based levels there: rush in and split exactly to quota
+55.96 (success 1.0); each present agent picks a zone with p = K/N +11.30
(success 0.149); assemble and never commit -6.75.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product as _iproduct
from math import factorial

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - gymnasium is a hard dep of the repo
    import gym
    from gym import spaces


# ======================================================================
# Base partition game (one-shot split at a fixed staging cell)
#
# TeamPartitionEnv below subclasses this and adds the regroup phase. Nothing
# here may be edited without re-checking that TeamPartitionEnv's dynamics are
# unchanged -- the trained baselines depend on them bit for bit.
# ======================================================================


def full_counts(n_agents: int, quota) -> tuple:
    """(K_STAY, K_1, ..., K_m) from the zone quota vector."""
    q = tuple(int(x) for x in quota)
    stay = n_agents - sum(q)
    if stay < 0:
        raise ValueError(f"quota {q} sums above n_agents={n_agents}")
    return (stay,) + q


def _multinomial_coefficient(n_agents: int, counts) -> int:
    c = factorial(n_agents)
    for k in counts:
        c //= factorial(int(k))
    return c


def max_exact_prob(n_agents: int, quota) -> float:
    """Equation (2), multinomial form: max_p P(X == K) for X ~ Multinomial(N, p).

    ell(p) = ln C + sum_c K_c ln p_c is strictly concave on the open simplex, so
    the stationary point of the Lagrangian is the unique maximiser and sits at
    p_c = K_c / N. Hence

        max_p P(X = K) = N! / prod_c K_c! * prod_c (K_c/N)^{K_c},

    taking 0^0 = 1. This is the best attainable *reliability* of any symmetric
    shared policy and it is the ceiling every CTDE baseline should converge to
    when lambda = 0.
    """
    counts = full_counts(n_agents, quota)
    out = float(_multinomial_coefficient(n_agents, counts))
    for k in counts:
        if k:
            out *= (k / n_agents) ** k
    return out


def exact_prob(n_agents: int, quota, p) -> float:
    """P(X == K) for X ~ Multinomial(n_agents, p). p includes the STAY slot."""
    counts = full_counts(n_agents, quota)
    p = np.asarray(p, dtype=float)
    if p.shape[0] != len(counts):
        raise ValueError(f"p has length {p.shape[0]}, expected {len(counts)}")
    out = float(_multinomial_coefficient(n_agents, counts))
    for k, pc in zip(counts, p):
        if k:
            out *= pc ** k
    return out


def quota_space(n_agents: int, n_zones: int, allow_zero_quota: bool = False):
    """Every admissible quota vector, as a sorted tuple of tuples.

    Constraints:
      * sum_c K_c == N            -- the whole team is allocated
      * K_c >= 1 (default)        -- headline config, tight and interpretable
        or K_c >= 0 with max_c K_c < N when `allow_zero_quota`
      * max_c K_c < N always      -- Condition 1's interiority, generalised: a
                                     quota of N is hit exactly by p = e_c

    Enumerated rather than rejection-sampled so that reset() draws uniformly from
    a space the analytics can average over exactly. Size is C(N-1, m-1) at most,
    which is 10 for the headline N=6, m=3.
    """
    if n_agents < n_zones:
        raise ValueError(f"n_agents={n_agents} < n_zones={n_zones}: no admissible quota")
    lo = 0 if allow_zero_quota else 1
    out = []
    for vec in _iproduct(range(lo, n_agents + 1), repeat=n_zones):
        if sum(vec) != n_agents:
            continue
        if max(vec) >= n_agents:  # generalised interiority
            continue
        out.append(tuple(vec))
    if not out:
        raise ValueError(
            f"no admissible quota vector for n_agents={n_agents}, n_zones={n_zones}, "
            f"allow_zero_quota={allow_zero_quota}"
        )
    return tuple(sorted(out))


def mean_ceiling(n_agents: int, n_zones: int, allow_zero_quota: bool = False) -> float:
    """Average of max_p P(X=K) over K drawn uniformly from `quota_space`.

    This is the number a symmetric CTDE learner should converge to under the
    default (lambda_coef = 0) config. N = 6, m = 3 gives 0.145062...
    """
    space = quota_space(n_agents, n_zones, allow_zero_quota)
    return float(np.mean([max_exact_prob(n_agents, k) for k in space]))


class _TeamPartitionBase(gym.Env):
    """The parts of the v0 env class that v2 builds on: construction, the quota
    and zone draw in reset, and the small shared methods. Copied verbatim."""

    metadata = {"render_modes": ["human"]}

    STAY = 0  # actions 1 .. n_zones are GO_1 .. GO_n_zones

    # Okabe-Ito, colour-blind safe. Render only.
    ZONE_COLOURS = ("#E69F00", "#0072B2", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
    ZONE_NAMES = ("amber", "blue", "green", "pink", "sky", "vermilion")

    def __init__(
        self,
        n_agents: int = 6,
        n_zones: int = 3,
        grid_w: int = 9,
        grid_h: int = 6,
        lambda_coef: float = 0.0,
        step_cost: float = 0.0,
        wrong_penalty: float = 0.0,
        fixed_quota=None,
        fixed_distances=None,
        allow_zero_quota: bool = False,
        symmetry_assert: bool = True,
        n_agents_max: int = 12,
        time_limit: int = 1,
        seed: int | None = None,
        render_flag: bool = False,
        render_theme: str = "paper",
        render_save_dir: str | None = None,
        render_dpi: int = 150,
        render_pause: float = 0.35,
        render_prefix: str = "team_partition",
        render_note: str = "",
        render_terrain: bool = True,
        render_terrain_contrast: float = 0.72,
        render_terrain_seed: int = 84,
        map_name: str | None = None,
    ):
        # `map_name` is a label, not a setting: src/main.py names the sacred
        # results directory after env_args.map_name when present. env_args is
        # forwarded verbatim to the constructor, so the env has to accept and
        # ignore it.
        del map_name

        super().__init__()

        if n_zones < 2:
            raise ValueError("n_zones must be >= 2: a one-way 'split' is not a split")
        if n_agents < n_zones:
            raise ValueError(f"n_agents={n_agents} < n_zones={n_zones}")

        self.n_agents = int(n_agents)
        self.n_zones = int(n_zones)
        self.n_actions = self.n_zones + 1
        self.G_W = int(grid_w)
        self.G_H = int(grid_h)
        self.lambda_coef = float(lambda_coef)
        # Used by team_partition, which subclasses this env and plays
        # over many rounds. Both default to 0 so v0's one-shot scoring is
        # unchanged: step_cost is never charged here (every step resolves), and
        # wrong_penalty = 0 leaves r(X) = 1[X == K] - sum_c lambda_c X_c.
        self.step_cost = float(step_cost)
        self.wrong_penalty = float(wrong_penalty)
        if self.step_cost < 0.0 or self.wrong_penalty < 0.0:
            raise ValueError("step_cost and wrong_penalty must be >= 0")
        self.allow_zero_quota = bool(allow_zero_quota)
        self.symmetry_assert = bool(symmetry_assert)
        self.n_agents_max = int(n_agents_max)
        self._time_limit = int(time_limit)
        self.render_flag = bool(render_flag)

        # --- rendering options (see render()) -------------------------------
        # Named constructor arguments so they can be set from an env_args block,
        # and plain attributes so a driver (scripts/render_policy.py) can flip
        # them on an already-constructed env.
        self.render_theme = str(render_theme)
        self.render_save_dir = render_save_dir
        self.render_dpi = int(render_dpi)
        self.render_pause = float(render_pause)
        self.render_prefix = str(render_prefix)
        self.render_note = render_note
        # Ground: procedural dust/scrub terrain. render_terrain=False falls back
        # to the flat background, and the contrast knob dials the texture down if
        # it competes with the foreground in print.
        self.render_terrain = bool(render_terrain)
        self.render_terrain_contrast = float(render_terrain_contrast)
        # Which patch of procedural ground this benchmark sits on. Every
        # benchmark uses a different seed so the figures do not share a texture.
        self.render_terrain_seed = int(render_terrain_seed)
        self._fig = self._ax = self._axp = None
        self._frame_paths = []
        self._episode_idx = -1
        self._frame_idx = 0

        # Row width of get_comm_mask(). 1 = per-agent connectivity bits (this
        # env), n_agents = a full pairwise adjacency. src/run.py sizes the
        # replay buffer's `comm_mask` field from this, via GymmaWrapper which
        # reads it off the unwrapped env. It would default to 1 anyway, but the
        # contract is load-bearing enough to state rather than inherit.
        self.comm_mask_shape = 1

        if self.G_W < self.n_zones:
            raise ValueError(
                f"grid_w={self.G_W} cannot hold {self.n_zones} zones in one row"
            )
        if self.G_H < 2:
            raise ValueError("grid_h must be >= 2 so the zones sit above the staging cell")

        self._quota_space = quota_space(self.n_agents, self.n_zones, self.allow_zero_quota)

        # Condition 1 guard. lambda_c = lambda_coef * d_c with d_c <= 1 and
        # sum_c K_c = N, so the cost of the target is at most lambda_coef * N.
        # r(K) = 1 - cost(K) must stay above the safe boundary r(0) = 0.
        if self.lambda_coef * self.n_agents >= 1.0:
            raise ValueError(
                f"lambda_coef={self.lambda_coef} violates Condition 1: the target's "
                f"travel cost can reach lambda_coef * N = "
                f"{self.lambda_coef * self.n_agents:.4f} >= 1, sinking r(K) to or below "
                f"the boundary r(0) = 0. Require lambda_coef < 1/{self.n_agents} "
                f"= {1.0 / self.n_agents:.4f}."
            )

        self.fixed_quota = None
        if fixed_quota is not None:
            fq = tuple(int(x) for x in fixed_quota)
            if fq not in self._quota_space:
                raise ValueError(
                    f"fixed_quota={fq} is not admissible: it must have length "
                    f"{self.n_zones}, sum to {self.n_agents}, and satisfy "
                    f"max_c K_c < {self.n_agents} (Condition 1, generalised)"
                    + ("" if self.allow_zero_quota else " with every K_c >= 1")
                )
            self.fixed_quota = fq

        self.fixed_distances = None
        if fixed_distances is not None:
            fd = np.asarray(fixed_distances, dtype=np.float32)
            if fd.shape != (self.n_zones,) or np.any(fd <= 0) or np.any(fd > 1):
                raise ValueError(
                    f"fixed_distances must be {self.n_zones} values in (0, 1]; got {fd}"
                )
            self.fixed_distances = fd

        # Staging cell: bottom row, horizontal centre. All N agents occupy it.
        self.staging = (self.G_W // 2, 0)
        self._zone_row = self.G_H - 1
        # Longest reachable trip to the zone row, used to normalise d_c so that a
        # corner zone reads exactly 1.0.
        self._max_trip = max(
            abs(c - self.staging[0]) for c in range(self.G_W)
        ) + (self._zone_row - self.staging[1])

        # Observation layout. Every field is common knowledge -- there is no
        # per-agent slot anywhere in this vector, by construction.
        #   quota one-hot per zone : n_zones * (N + 1)
        #   distances              : n_zones
        #   lambdas                : n_zones
        #   team size             : 1
        self.obs_dim = self.n_zones * (self.n_agents + 1) + 2 * self.n_zones + 1
        # Nothing is hidden in this game, so the centralised critic is handed
        # exactly the same vector -- honest, and it makes the point that the
        # barrier is decentralised execution, not partial observability.
        self.state_dim = self.obs_dim

        self.action_space = spaces.Tuple(
            tuple(spaces.Discrete(self.n_actions) for _ in range(self.n_agents))
        )
        self.observation_space = spaces.Tuple(
            tuple(
                spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
                for _ in range(self.n_agents)
            )
        )

        self._rng = np.random.default_rng(seed)

        # Episode state (set by reset)
        self.quota: tuple | None = None
        self.zone_cols: tuple | None = None
        self.d_norm: np.ndarray | None = None
        self.lam: np.ndarray | None = None
        self._t = 0
        self._n_stalls = 0
        self._last_counts: tuple | None = None
        self._last_actions: list | None = None

    @property
    def last_counts(self):
        """The realised count vector, which `step`'s info dict cannot carry.

        `info` is restricted to summable scalars by the parallel runner's
        aggregation (see `step`), so the vector itself lives here.
        """
        return self._last_counts

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t = 0
        self._n_stalls = 0
        self._last_counts = None
        self._last_actions = None

        # Condition 1: the target is set randomly in each game, and is never
        # attainable by a deterministic symmetric policy (max_c K_c < N).
        if self.fixed_quota is not None:
            self.quota = self.fixed_quota
        else:
            idx = int(self._rng.integers(0, len(self._quota_space)))
            self.quota = self._quota_space[idx]

        # Geometry. Zones occupy distinct columns of the top row; both the
        # columns and the resulting distances are public.
        cols = self._rng.choice(self.G_W, size=self.n_zones, replace=False)
        self.zone_cols = tuple(int(c) for c in np.sort(cols))

        if self.fixed_distances is not None:
            self.d_norm = self.fixed_distances.astype(np.float32).copy()
        else:
            sx, sy = self.staging
            self.d_norm = np.array(
                [
                    (abs(c - sx) + (self._zone_row - sy)) / self._max_trip
                    for c in self.zone_cols
                ],
                dtype=np.float32,
            )

        # lambda_c = per-unit travel cost x distance. This is the Theorem 2 knob.
        self.lam = (self.lambda_coef * self.d_norm).astype(np.float32)

        self._episode_idx += 1
        self._frame_idx = 0
        if self.render_flag:
            # The reset frame is the decision moment itself -- the whole team
            # still in the staging cell with the quotas already public -- which
            # is the frame the paper wants next to the outcome.
            self.render()

        return self.get_obs(), {}

    def get_avail_actions(self):
        """Every action is always available to every agent.

        Per-agent action masks would be a symmetry-breaking channel (Conditions
        2/3), so there are none. The shared GymmaWrapper returns all-ones
        regardless (src/envs/gymma.py), which matches this exactly.
        """
        return [[1] * self.n_actions for _ in range(self.n_agents)]

    def get_env_info(self):
        return {
            "n_agents": self.n_agents,
            "n_actions": self.n_actions,
            "obs_shape": self.obs_dim,
            "state_shape": self.state_dim,
            "episode_limit": self._time_limit,
            "comm_mask_shape": self.comm_mask_shape,
        }

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def close(self):
        fig = getattr(self, "_fig", None)
        if fig is not None:
            import matplotlib.pyplot as plt

            plt.close(fig)
            self._fig = self._ax = self._axp = None
        return None


# ======================================================================
# Analytic references
# ======================================================================


def _multinomial_pmf(n: int, counts, p) -> float:
    """Mult(counts; n, p) over the zones only (no STAY slot)."""
    if sum(counts) != n:
        return 0.0
    return exact_prob(n, counts, np.concatenate([[0.0], np.asarray(p, dtype=float)]))


def _compositions(n: int, m: int):
    for v in _iproduct(range(n + 1), repeat=m):
        if sum(v) == n:
            yield v


@lru_cache(maxsize=None)
def coordinated_success_prob(quota, transit_prob: float) -> float:
    """Success probability of a perfectly coordinated team under transit
    interference: every agent goes to its assigned zone (so the counts match the
    quota before transit), and each one independently ends in a uniformly random
    other zone with probability `transit_prob`. Exact, by dynamic programming
    over the agents. This is the ceiling of MAT, CTCE or an oracle; it is
    (1 - transit_prob)^N plus the small chance that misroutes cancel out.
    """
    quota = tuple(int(x) for x in quota)
    m = len(quota)
    q = float(transit_prob)
    if q == 0.0:
        return 1.0
    dist = {(0,) * m: 1.0}
    for c, k in enumerate(quota):
        for _ in range(k):
            nxt = {}
            for counts, pr in dist.items():
                for z in range(m):
                    pz = (1.0 - q) if z == c else q / (m - 1)
                    cc = list(counts)
                    cc[z] += 1
                    cc = tuple(cc)
                    nxt[cc] = nxt.get(cc, 0.0) + pr * pz
            dist = nxt
    return float(dist.get(quota, 0.0))


def compensated_symmetric_p(quota, transit_prob: float) -> np.ndarray:
    """The zone distribution a symmetric agent should aim for so that, after
    transit interference, it lands in zone c with probability K_c / N.

    Transit interference maps an aim p to a landing distribution
    p'_c = p_c (1 - q - q/(m-1)) + q/(m-1). Inverting it at p' = K/N is always
    possible for the shipped settings, which is why the symmetric ceiling
    max_p P(X = K) is unchanged by this noise.
    """
    quota = np.asarray(quota, dtype=float)
    m = len(quota)
    q = float(transit_prob)
    target = quota / quota.sum()
    p = (target - q / (m - 1)) / (1.0 - q - q / (m - 1))
    if np.any(p < 0.0):
        raise ValueError(f"transit_prob={q} is too large to compensate for quota {tuple(quota)}")
    return p / p.sum()


def class_ceiling(sizes, quota, n_restarts: int = 6, seed: int = 0) -> float:
    """Best symmetric success when the team is split into observable classes.

    Agents whose observation histories coincide form a class; a shared policy
    can give each class its own zone distribution p_j, but agents inside a class
    still sample independently. With class sizes n_j (summing to N) and quota K,

        P(success) = sum over decompositions K = sum_j k_j, |k_j| = n_j,
                     of prod_j Mult(k_j; n_j, p_j),

    and this returns its maximum over the p_j (without transit interference,
    which can only lower it -- so this is an upper bound either way). One class
    reproduces `max_exact_prob` exactly; N singleton classes reach 1.0.

    Two exact reductions keep this cheap:

    * STAY is left out of every p_j: sum_c K_c = N means success needs every
      agent to go, and moving STAY mass onto the zones can only raise each
      term, so the optimum never uses it.
    * A singleton class enters P(success) LINEARLY (sum_c p_jc * P(rest hits
      K - e_c)), so its optimum is a vertex: a deterministic zone. Singletons
      are therefore enumerated as a count vector s <= K and removed, and only
      classes of size >= 2 are searched numerically, on the quota K - s.

    The remaining objective is not concave, so that search is seeded with every
    decomposition's closed-form optimum (p_j = k_j / n_j, evaluated exactly) and
    refined by multi-start Nelder-Mead in softmax coordinates.
    """
    quota = tuple(int(x) for x in quota)
    sizes = [int(s) for s in sizes if int(s) > 0]
    if sum(sizes) != sum(quota):
        raise ValueError(f"class sizes {sizes} must sum to sum(quota) = {sum(quota)}")
    n_single = sum(1 for s in sizes if s == 1)
    rest = tuple(sorted(s for s in sizes if s > 1))
    if not rest:
        return 1.0
    best = 0.0
    for s in _compositions(n_single, len(quota)):
        if all(s[c] <= quota[c] for c in range(len(quota))):
            sub = tuple(quota[c] - s[c] for c in range(len(quota)))
            best = max(best, _class_ceiling_core(rest, sub, int(n_restarts), int(seed)))
    return best


@lru_cache(maxsize=None)
def _class_ceiling_core(sizes: tuple, quota: tuple, n_restarts: int, seed: int) -> float:
    """class_ceiling for classes that all have size >= 2 (see above)."""
    from scipy.optimize import minimize

    m = len(quota)
    J = len(sizes)

    def decompositions(j, remaining):
        if j == J - 1:
            if sum(remaining) == sizes[j]:
                yield (tuple(remaining),)
            return
        for k in _compositions(sizes[j], m):
            if all(k[c] <= remaining[c] for c in range(m)):
                tail_quota = tuple(remaining[c] - k[c] for c in range(m))
                for tail in decompositions(j + 1, tail_quota):
                    yield (k,) + tail

    decs = list(decompositions(0, quota))
    if not decs:
        return 0.0

    def value(ps):
        return float(sum(
            np.prod([_multinomial_pmf(sizes[j], d[j], ps[j]) for j in range(J)])
            for d in decs
        ))

    best = 0.0
    starts = []
    for d in decs:
        ps = [np.asarray(d[j], dtype=float) / sizes[j] for j in range(J)]
        best = max(best, value(ps))
        z = []
        for p in ps:
            q = np.clip(p, 1e-6, None)
            q = q / q.sum()
            z.append(np.log(q[1:]) - np.log(q[0]))
        starts.append(np.concatenate(z))

    def unpack(z):
        z = np.clip(np.asarray(z, dtype=float).reshape(J, m - 1), -60.0, 60.0)
        e = np.exp(np.concatenate([np.zeros((J, 1)), z], axis=1))
        return [row / row.sum() for row in e]

    rng = np.random.default_rng(seed)
    for _ in range(n_restarts):
        starts.append(rng.normal(0.0, 2.0, size=J * (m - 1)))
    for z0 in starts:
        res = minimize(lambda z: -value(unpack(z)), z0, method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-13, "maxiter": 4000})
        best = max(best, value(unpack(res.x)))
    return float(min(best, 1.0))


# ======================================================================
# Environment
# ======================================================================


class TeamPartitionEnv(_TeamPartitionBase):
    """v0's partition, preceded by a regroup at the staging cell, with signal
    interference on the way in and in transit to the zones."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_agents: int = 9,
        n_zones: int = 3,
        grid_w: int = 9,
        grid_h: int = 6,
        lambda_coef: float = 0.0,
        step_cost: float = 0.1,
        wrong_penalty: float = 20.0,
        zone_reward: float = 10.0,
        all_reward: float = 100.0,
        dodge_penalty: float = 0.2,
        fixed_quota=None,
        fixed_distances=None,
        allow_zero_quota: bool = False,
        symmetry_assert: bool = True,
        time_limit: int = 20,
        spawn_distance: int = 2,
        wave_gap: int = 3,
        near_wave_size: int = 6,
        interference_prob: float = 0.0,
        drift_at_staging: bool = False,
        staging_drift_prob: float | None = None,
        transit_interference_prob: float = 0.0,
        shaping_weight: float = 1.0,
        step_cost_far: float | None = 5.0,
        n_quota_readers: int | None = None,
        commit_gate: bool = True,
        commit_quorum: int | None = None,
        commit_window: int | None = 1,
        gate_deadline: int | None = None,
        miss_penalty: float = 25.0,
        public_regrouped=False,
        expose_own_wave: bool = False,
        n_spare: int = 3,
        k_min: int = 1,
        comm_radius: int = 2,
        seed: int | None = None,
        render_flag: bool = False,
        map_name: str | None = None,
        render_theme: str = "paper",
        render_save_dir: str | None = None,
        render_dpi: int = 150,
        render_pause: float = 0.45,
        render_prefix: str = "team_partition",
        render_note: str = "",
        render_trail: int = 6,
        render_terrain: bool = True,
        render_terrain_contrast: float = 0.72,
        render_terrain_seed: int = 84,
    ):
        # Defaults are the HEADLINE configuration (src/config/envs/
        # team_partition_v2.yaml), unlike v0 whose constructor defaults are the
        # one-shot special case. Anything that builds the env without kwargs --
        # the CTCE runner's gym.make, notebooks -- therefore plays the same game
        # as the EPyMARL and MAT runs.
        super().__init__(
            n_agents=n_agents,
            n_zones=n_zones,
            grid_w=grid_w,
            grid_h=grid_h,
            lambda_coef=lambda_coef,
            step_cost=step_cost,
            wrong_penalty=wrong_penalty,
            fixed_quota=fixed_quota,
            fixed_distances=fixed_distances,
            allow_zero_quota=allow_zero_quota,
            symmetry_assert=symmetry_assert,
            time_limit=time_limit,
            seed=seed,
            render_flag=render_flag,
            map_name=map_name,
            render_theme=render_theme,
            render_save_dir=render_save_dir,
            render_dpi=render_dpi,
            render_pause=render_pause,
            render_prefix=render_prefix,
            render_note=render_note,
            render_terrain=render_terrain,
            render_terrain_contrast=render_terrain_contrast,
            render_terrain_seed=render_terrain_seed,
        )
        # Render only: how many past positions each drone's trail shows.
        self.render_trail = int(render_trail)
        self._render_step = None      # what the last step did, for the frame
        self._render_return = 0.0
        self._in_reset = False

        # Graded reward, transplanted from charging_rendezvous (its
        # REWARD_SATISFY / REWARD_SUCCESS_ALL / COLLAPSE_BETA). See the module
        # docstring section "WHY THE REWARD IS GRADED".
        self.zone_reward = float(zone_reward)
        self.all_reward = float(all_reward)
        self.dodge_penalty = float(dodge_penalty)

        # --- aerial's distance-scaled time cost -------------------------------
        # None keeps the ORIGINAL rule: step_cost only on rounds in which nobody
        # moved, plus dodge_penalty once assembled. That rule makes the assembled
        # state the ONLY punishable one, so the safe play is to stop one cell
        # short of the staging cell and spam blocked GOs -- measured +9.20,
        # against -2.52 for assembling with 2% per-round GO leakage. Three arms
        # learned exactly that in results/overnight/v2_stage1_aerial (CommNet
        # regrouped_ever 0.19 -> 0.0035 while its return ROSE 11.1 -> 13.8).
        #
        # A float turns on the live aerial env's rule instead: a cost charged
        # EVERY non-resolving round, scaled by the team's mean distance to the
        # staging cell -- step_cost at the cell, step_cost_far at the far edge of
        # the grid. Idling is then never free anywhere, and the bleed slows as
        # the team closes in, which is what orders the degenerate outcomes
        # solve > wait at the staging cell > roam near it. aerial runs -0.1 near
        # / -0.3 far against a prize of ~20 (0.5% / 1.5% per round); the shipped
        # 0.1 / 1.5 against all_reward 100 is that ratio at our scale.
        # dodge_penalty stays available on top but should be 0 under this rule:
        # aerial deleted its own ready-wait penalty precisely because it ranked
        # "wait on the target" BELOW "roam around" and produced a degenerate
        # freeze.
        self.step_cost_far = None if step_cost_far is None else float(step_cost_far)
        if self.step_cost_far is not None and self.step_cost_far < self.step_cost:
            raise ValueError(
                f"step_cost_far ({self.step_cost_far}) must be >= step_cost "
                f"({self.step_cost}): the cost has to RISE with distance, or "
                "roaming becomes cheaper than assembling, which is the failure "
                "this knob exists to remove."
            )

        # --- aerial's hidden K ------------------------------------------------
        # None (or n_agents) = every agent reads the quota, which is what makes
        # the split a (quota, id) lookup table and let MAPPO w/ID reach 0.978.
        # An int m < n_agents hands the quota to exactly m agents per episode,
        # drawn without replacement -- a fixed PUBLIC size with PRIVATE
        # membership, the same construction as near_wave_size, and the same
        # reason: an independently drawn per-agent probability is worth exactly
        # zero. The rest read the sentinel -1 in every quota slot, which is
        # aerial's `target_requirement = -1.0` for a worker out of comm range.
        #
        # This caps an ID-equipped INDEPENDENT policy exactly. The uninformed
        # agents can do no better than a fixed per-id profile v (they never
        # observe the quota, in this episode or any other), the informed fill
        # K - v, and the attempt succeeds iff K >= v componentwise:
        #     m =  0      1      2      3+
        #        0.100  0.300  0.600  1.000
        # A policy that POOLS the quota is unaffected: SDec-MAT's encoder
        # attends across agents, so the informed agent's slots reach every
        # decode position.
        #
        # !! Condition 3 is broken at the coordination state by this flag, in
        # the same way and for the same reason that obs_agent_id breaks it: the
        # informed and the uninformed no longer observe the same thing. That is
        # acceptable only on the WITH-ID protocol (the MAC-suite figure), where
        # ids have already voided it. Leave it off for any ID-free run that
        # quotes the 0.145062 barrier. symmetry_assert below is relaxed to check
        # equality WITHIN each visibility class.
        if n_quota_readers is None:
            self.n_quota_readers = n_agents
        else:
            self.n_quota_readers = int(n_quota_readers)
            if not 1 <= self.n_quota_readers <= n_agents:
                raise ValueError(
                    f"n_quota_readers must be in [1, {n_agents}], got "
                    f"{n_quota_readers}. 0 readers makes the task unsolvable by "
                    "anything -- nobody ever learns the quota."
                )
        self._quota_readers = None
        if self.zone_reward < 0.0 or self.all_reward < 0.0:
            raise ValueError("zone_reward and all_reward must be >= 0")
        if self.dodge_penalty < 0.0:
            raise ValueError(f"dodge_penalty must be >= 0, got {dodge_penalty}")
        if self.zone_reward == 0.0 and self.all_reward == 0.0:
            raise ValueError("zone_reward and all_reward cannot both be 0: the "
                             "partition would carry no reward at all")
        # Condition 1 with the graded reward. r(X) = zone_reward * #matched +
        # all_reward * 1[X == K] - sum_c lambda_c X_c is still a function of the
        # count vector alone, and it is still uniquely maximised at X = K
        # (every quota has K_c >= 1, so X = 0 matches no zone and scores 0 for
        # the reward part). What has to be re-checked is that the travel cost
        # cannot sink r(K) to or below r(0) = 0 on the new scale.
        r_at_K = self.zone_reward * self.n_zones + self.all_reward
        if self.lambda_coef * self.n_agents >= r_at_K:
            raise ValueError(
                f"lambda_coef={self.lambda_coef} violates Condition 1 on the graded "
                f"scale: the target's travel cost can reach lambda_coef * N = "
                f"{self.lambda_coef * self.n_agents:.4f} >= r(K) = {r_at_K:.4f}, "
                f"sinking r(K) to or below r(0) = 0."
            )

        # --- the two rules that decide whether an agent id is enough ---
        # commit_gate     True  -> a GO is inert until the team has assembled
        #                          (mistiming is free; an id then solves the task)
        #                 False -> v0's and aerial's rule: any GO from the
        #                          staging cell RESOLVES, so committing at the
        #                          wrong moment is a real, scored failure
        # public_regrouped True  -> the observation carries an exact public
        #                          "everyone is here" flag, which hands every
        #                          policy the timing for free
        #                 False -> aerial/charging's own-only view: the agent
        #                          sees its own at_staging and its own comm bit
        #                          and must infer the moment
        self.commit_gate = bool(commit_gate)
        # True | False | "count". "count" is aerial's n_free_connected and is
        # what a sub-roster quorum needs: see get_obs_agent.
        if isinstance(public_regrouped, str):
            if public_regrouped != "count":
                raise ValueError(
                    f"public_regrouped must be True, False or 'count', got "
                    f"{public_regrouped!r}"
                )
            self.public_regrouped = "count"
        else:
            self.public_regrouped = bool(public_regrouped)
        # own_wave: which spawn ring this agent started on. OFF by default. The
        # cohort it identifies is a symmetry break that is provably unreachable
        # by gradient (the gain from tilting two cohorts apart grows as eps^2 --
        # P(X=K) is an EVEN function of the tilt -- measured at +4e-5 for
        # eps=0.01 against +/-0.018 of eval noise), so the field buys nothing and
        # is one more per-agent quantity to have to justify. The RINGS stay: their
        # real job is to make the ARRIVAL TIME uncertain, which is what keeps the
        # readiness signal uninformative (one ring instead of two lets an
        # independent+id policy reach 68% of the range instead of 22%).
        self.expose_own_wave = bool(expose_own_wave)

        # --- the quota is satisfiable by a SUBSET (aerial's K < n_workers) ---
        #
        # This is the change that makes aerial's separation transplantable at
        # all. In v0 the quota sums to N, so success needs EVERY agent and no
        # subset of the team can ever be the right answer. An autoregressive
        # decode chain then buys nothing over independent actors on the timing
        # question -- both have to wait for all N -- which is why an agent id
        # was a complete solution here (MAPPO+id measured 0.958) while it is
        # not in aerial.
        #
        # With sum_c K_c = N - n_spare and n_spare > 0, the chain can commit
        # exactly the right agents FROM AMONG THOSE CURRENTLY ON THE STAGING
        # CELL and leave the rest on STAY. Independent actors cannot: each has
        # to guess, alone, whether it is one of the chosen and whether now is
        # the moment.
        #
        # k_min >= 2 is the established way to keep the horizon from becoming a
        # free re-roll -- pickup_consensus uses exactly this, and v0's yaml
        # names it as the alternative to full allocation ("that is exactly the
        # exploit pickup_consensus must exclude with k_min >= 2"). At K_c = 1 a
        # tiny commit rate would turn "nobody acted" into a costless retry.
        self.n_spare = int(n_spare)
        self.k_min = int(k_min)
        self.comm_radius = int(comm_radius)
        if self.n_spare < 0 or self.n_spare >= self.n_agents:
            raise ValueError(f"n_spare must lie in [0, n_agents), got {n_spare}")
        if self.k_min < 1:
            raise ValueError(f"k_min must be >= 1, got {k_min}")
        if self.n_spare > 0 and self.k_min < 2 and self.dodge_penalty <= 0.0:
            raise ValueError(
                "n_spare > 0 with k_min < 2 and dodge_penalty = 0 makes an "
                "all-STAY round a FREE re-roll: conditioned on anyone committing "
                "it is almost surely exactly one agent, so a long horizon would "
                "raise the ceiling. pickup_consensus excludes this with "
                "k_min >= 2; v2 can also price it, because a declining round "
                "costs step_cost + dodge_penalty. Either set k_min >= 2 or keep "
                "dodge_penalty > 0 (and check with multistep_ceiling that the "
                "ceiling really is flat in the horizon)."
            )
        if self.comm_radius < 0:
            raise ValueError(f"comm_radius must be >= 0, got {comm_radius}")
        n_commit = self.n_agents - self.n_spare
        if self.k_min * self.n_zones > n_commit:
            raise ValueError(
                f"infeasible: k_min({self.k_min}) * n_zones({self.n_zones}) = "
                f"{self.k_min * self.n_zones} > n_agents - n_spare = {n_commit}"
            )

        # --- WHO IS PRESENT WHEN THE GATE OPENS (aerial's real device) ---
        #
        # The gate used to open only on at_staging.all(), i.e. on the FULL
        # ROSTER. That single fact is what let an agent id solve v2: with the
        # acting set always {0..N-1} and the quota public, the optimal policy
        # is the static table zone = f(quota, my_id). Measured directly -- a
        # 60-entry table over the 10 admissible quotas scores exact_match
        # 1.0000 over 3000 episodes, and MAPPO w/ID found it at 120k steps
        # (test_exact_match 0.9975, results/sacred/mappo/team-partition-v0/26).
        #
        # aerial never had that property, and the same table transplanted onto
        # the live aerial env scores only 0.2507 (wind 0.1) / 0.4413 (oracle
        # timing, no wind), with undershoot 0.56/episode. The reason is that
        # aerial's lift fires on N_comm >= K with K < n_workers, so WHICH
        # workers are on the target when it fires varies episode to episode and
        # a table keyed on (K, id) cannot align with it.
        #
        # commit_quorum ports that: the gate opens as soon as this many agents
        # are on the staging cell, not when all of them are. With n_spare > 0
        # (sum K = N - n_spare) and interference on to scatter the arrivals,
        # the acting set genuinely varies -- measured, the subset present at
        # quorum takes all C(6,5) = 6 values at interference_prob >= 0.05,
        # against 2 values at 0.0. An autoregressive decode chain handles this
        # natively (it conditions on who has already acted); an independent
        # policy with an id cannot, which is the separation the benchmark is
        # supposed to measure.
        #
        # None => n_agents - n_spare. At the shipped n_spare = 0 that is N, so
        # the default reproduces the old all-roster gate bit for bit and the
        # v0 reduction tests are unaffected.
        # --- THE WAVE PASSES (charging's device) ---
        #
        # Without this the gate, once open, stays open for the rest of the
        # episode, so an independent policy can simply WAIT until the full
        # roster has assembled and then play the static (quota, id) table.
        # Measured 2026-09-14: that strategy scores 1.0000 in every clean-arrival
        # configuration, against 0.07-0.22 for the same table forced to act when
        # the gate first opens. Waiting is the whole exploit.
        #
        # It cannot be closed with noise. Deterministic arrival makes the roster
        # predictable (just count rounds); stochastic arrival makes each agent's
        # arrival its own history class, which hands a recurrent policy an
        # arrival-RANK rule worth 1.0000 (that is what QMIX w/ID found, 0.9855).
        # Both exploits live on the same axis and closing one opens the other.
        #
        # charging_rendezvous does not have the problem because its
        # opportunity is TRANSIENT: the ports are free now, the batteries are
        # draining, and a drone that waits for a better moment dries out. This
        # is that rule -- the attempt must be made within `commit_window` rounds
        # of the gate first opening, after which the episode ends as a miss.
        # None = the old behaviour (the gate never closes).
        self.commit_window = None if commit_window is None else int(commit_window)
        # --- charging's "strict loss for idle capacity while demand exists" ---
        #
        # commit_window has a side effect that must be paid for: the episode now
        # ENDS as soon as the window expires, so letting the wave pass costs one
        # round instead of the whole horizon. Measured 2026-09-14 on the first
        # windowed preset, that made NOT TRYING the best play on the board:
        #
        #     wait (let the window expire)   +16.58   <- best
        #     uniform-random commit           -3.77
        #     symmetric commit                -2.12
        #     coordinated                   +114.88
        #
        # The +16.58 is the approach shaping, banked and never charged back
        # because the episode stops. The gradient therefore pointed away from
        # committing from step one and EVERY arm collapsed to no_attempt 1.000 --
        # SDec-MAT included (entropy 0.0007 at 492k). That is Theorem 2(i)
        # catching our own method, the same way wrong_penalty=100 did.
        #
        # This charges the miss, so that trying beats not trying and the
        # exploration budget goes into the count decision.
        self.miss_penalty = float(miss_penalty)
        if self.miss_penalty < 0.0:
            raise ValueError(f"miss_penalty must be >= 0, got {miss_penalty}")
        if self.commit_window is not None and self.commit_window < 1:
            raise ValueError(
                f"commit_window must be >= 1 (the round the gate opens is itself "
                f"playable), got {commit_window}")
        self.commit_quorum = (n_commit if commit_quorum is None
                              else int(commit_quorum))
        if not n_commit <= self.commit_quorum <= self.n_agents:
            raise ValueError(
                f"commit_quorum must lie in [n_agents - n_spare, n_agents] = "
                f"[{n_commit}, {self.n_agents}], got {self.commit_quorum}. Below "
                f"n_commit the quota could not be filled by the agents present; "
                f"above n_agents it could never open."
            )
        # Rebuild the quota space over the SUBSET total, with the k_min floor.
        if self.n_spare > 0 or self.k_min > 1:
            space = tuple(
                q for q in _iproduct(range(self.k_min, n_commit + 1),
                                     repeat=self.n_zones)
                if sum(q) == n_commit and max(q) < n_commit
            )
            if not space:
                raise ValueError(
                    f"no admissible quota with n_zones={self.n_zones}, "
                    f"k_min={self.k_min}, total={n_commit}"
                )
            self._quota_space = space
            if self.fixed_quota is not None and tuple(self.fixed_quota) not in space:
                raise ValueError(
                    f"fixed_quota={tuple(self.fixed_quota)} is not admissible for "
                    f"k_min={self.k_min}, total={n_commit}"
                )

        # --- THE GATE ITSELF IS THE OPPORTUNITY (added 2026-09-15) ---
        # The gate must OPEN by this round or the episode ends as a miss.
        # None => spawn_distance, i.e. the round the near wave lands.
        #
        # !! THIS IS WHAT BLOCKS THE ELECTED-COMMITTEE EXPLOIT, and it is the
        # only thing that does. commit_quorum was meant to deny a static
        # (quota, id) table by making the acting set a random 6-subset of 8 --
        # but WHO IS PRESENT was chosen by the POLICY, not the environment, so
        # an id-equipped team simply had two agents never move: the remaining
        # six then formed the quorum by themselves, the roster stopped varying,
        # and the table worked again. Measured: exact 1.0000, return +141.51
        # against a cap of 144.88, and it reproduced QMIX w/ID's log fingerprint
        # exactly (n_hist_classes 2.96 vs 2.943, regrouped_ever 0.000 vs 0.0000,
        # gate_round 4.12 vs 2 for a team that rushes). QMIX rode it to
        # exact-match 0.9305 at 1M.
        #
        # The deadline closes it WITHOUT touching the reward, because being
        # present at the deadline requires having moved every round, and no
        # agent can tell whether it is in the near wave -- so the only way to
        # have commit_quorum agents on the cell in time is for EVERYONE to
        # rush, and then the deciders are the near wave, which the environment
        # drew. The acting set becomes env-controlled instead of
        # policy-controlled, which is the property commit_quorum was supposed
        # to provide.
        #
        # Measured over 2000 episodes at the shipped preset:
        #     honest (everyone rushes)        gate by round 2   1.0000
        #     sacrifice (2 agents never move) gate by round 2   0.0325
        # 0.0325 is C(6,6)/C(8,6) = 1/28: the sacrifice only survives when all
        # six movers happen to be the near wave.
        #
        # !! Set to a large number (or raise spawn_distance without raising
        # this) to restore the old behaviour; a test pins the exploit's score
        # in both directions.
        # !! The default is the EARLIEST ROUND THE QUORUM CAN BE MET with every
        # agent rushing, which is not always spawn_distance: if the quorum is
        # larger than the near wave it cannot be met until the FAR wave lands,
        # at spawn_distance + wave_gap. Hard-coding spawn_distance made the
        # n_spare = 0 reduction (quorum = full roster) unsolvable -- the gate
        # could never open in time and every episode truncated as a miss, which
        # tests/test_team_partition_v2_quorum.py caught immediately.
        if gate_deadline is None:
            # The far wave lands at spawn_distance + wave_gap. The SACRIFICE
            # cannot assemble before then (its movers are a random subset, so
            # ~1.5 of the six are far), while honest play assembles when the
            # NEAR wave lands, at spawn_distance. So set the deadline as LATE as
            # possible while still landing strictly before the far wave: that
            # blocks the sacrifice and gives honest play every round of slack
            # the geometry allows.
            #
            # !! The slack is not optional. At wave_gap 1 the deadline is forced
            # to 2, and a policy that SAMPLES at test time cannot reliably get
            # six agents to move twice in a row: MAPPO and CommNet both read
            # gate_missed 1.0000, so the figure measured navigation instead of
            # counting. Measured P(gate in time), 400 eps:
            #     gap/deadline   p(MOVE)=.85   =.95   =1.0   SACRIFICE
            #        1 / 2          0.195      0.613  1.000    0.033
            #        3 / 4          0.985      0.998  1.000    0.033
            # which is why wave_gap ships at 3.
            far_round = int(spawn_distance) + int(wave_gap)
            if self.commit_quorum > int(near_wave_size):
                # The quorum NEEDS the far wave (n_spare = 0 is this case), so
                # the deadline cannot precede it or the task is unsolvable.
                gate_deadline = far_round
            else:
                gate_deadline = max(int(spawn_distance), far_round - 1)
        self.gate_deadline = int(gate_deadline)
        if self.gate_deadline < 1:
            raise ValueError(
                f"gate_deadline must be >= 1, got {self.gate_deadline}")

        self.spawn_distance = int(spawn_distance)
        self.wave_gap = int(wave_gap)
        self.near_wave_size = int(near_wave_size)
        self.interference_prob = float(interference_prob)
        # aerial's wind hits every worker, including the ones already standing
        # on the target -- only those ATTEMPTING a pickup are exempt. v2 shipped
        # the opposite: interference touched agents in flight only, so the
        # staging cell was an ABSORBING set and, once assembled, the team could
        # sit there for free. That is what makes "wait for the full roster, then
        # play the (quota, id) table" a safe strategy and hands an id-equipped
        # independent policy the task. With drift on, the assembled set churns,
        # the full roster is rarely present all at once, and -- because a GO is
        # live as soon as commit_quorum agents are here -- waiting for it is a
        # real gamble rather than a free option.
        self.drift_at_staging = bool(drift_at_staging)
        # !! THE TWO NOISE SOURCES DO OPPOSITE JOBS AND MUST BE SET SEPARATELY.
        #
        #   in flight (interference_prob)  scatters ARRIVAL TIMES. That is a
        #       HISTORY signal: a recurrent policy can rank itself by when it
        #       arrived and split on that rank. Measured on the shipped preset
        #       at 0.20, an arrival-rank policy scores exact_match 1.0000 with
        #       3.18 distinct observation histories per episode -- and QMIX w/ID
        #       found exactly that (0.9975 by 140k, committed 4.00 with
        #       regrouped_ever 0.000). The module docstring's own table prices
        #       it: 1.82 classes -> leak 0.341, 2.32 -> 0.483, 2.57 -> 0.561.
        #
        #   at the staging cell (staging_drift_prob)  varies WHO IS PRESENT when
        #       the gate opens. That is what denies the memoryless (quota, id)
        #       lookup table, and it costs no arrival-time information because
        #       every agent it touches has already arrived.
        #
        # Tying them to one number forces a choice between the two exploits.
        # None keeps the old single-knob behaviour (= interference_prob).
        self.staging_drift_prob = (float(interference_prob)
                                   if staging_drift_prob is None
                                   else float(staging_drift_prob))
        if not 0.0 <= self.staging_drift_prob <= 1.0:
            raise ValueError(
                f"staging_drift_prob must lie in [0, 1], got {staging_drift_prob}")
        self.transit_interference_prob = float(transit_interference_prob)
        self.shaping_weight = float(shaping_weight)

        if self.lambda_coef > 0.0 and self.fixed_distances is None:
            raise ValueError(
                "lambda_coef > 0 requires fixed_distances in v2: the observation does "
                "not carry the per-episode zone distances, so with randomised zone "
                "columns each zone's cost would be an unobserved variable. Pin the "
                "distances (v0's lambda sweep does the same)."
            )
        if not 0.0 <= self.interference_prob <= 1.0:
            raise ValueError(f"interference_prob must lie in [0, 1], got {interference_prob}")
        if not 0.0 <= self.transit_interference_prob < 1.0:
            raise ValueError(
                f"transit_interference_prob must lie in [0, 1), got {transit_interference_prob}"
            )
        if self.shaping_weight < 0.0:
            raise ValueError(f"shaping_weight must be >= 0, got {shaping_weight}")
        if self.spawn_distance < 1:
            raise ValueError("spawn_distance must be >= 1: at 0 the agents start at the "
                             "staging cell, which is v0")
        if self.wave_gap < 0:
            raise ValueError(f"wave_gap must be >= 0, got {wave_gap}")
        if not 0 <= self.near_wave_size <= self.n_agents:
            raise ValueError(
                f"near_wave_size must lie in [0, n_agents={self.n_agents}], "
                f"got {near_wave_size}"
            )
        self._far_distance = self.spawn_distance + self.wave_gap
        if self._time_limit < self._far_distance + 1:
            raise ValueError(
                f"time_limit={self._time_limit} leaves no round to go after a "
                f"{self._far_distance}-step approach from the far wave; need "
                f">= spawn_distance + wave_gap + 1"
            )

        # Actions: v0's STAY (0) and GO_1..GO_m (1..m), plus MOVE (m + 1).
        self.MOVE = self.n_zones + 1
        self.n_actions = self.n_zones + 2
        self.action_space = spaces.Tuple(
            tuple(spaces.Discrete(self.n_actions) for _ in range(self.n_agents))
        )

        sx, sy = self.staging
        # Cells an agent may occupy in flight: the whole grid below the zone row.
        self._flight_rows = self._zone_row
        def _ring(d):
            return tuple(
                (x, y)
                for y in range(self._flight_rows)
                for x in range(self.G_W)
                if abs(x - sx) + abs(y - sy) == d
            )

        # TWO WAVES. Agents start on one of two Manhattan rings: the near ring at
        # `spawn_distance`, or the far ring `wave_gap` cells beyond it. With
        # wave_gap >= 1 the two groups reach the staging cell in different
        # rounds, so the team assembles as a STAGGERED ARRIVAL of two cohorts
        # rather than all at once -- the structural symmetry break, in place of
        # relying on noise to produce one by accident.
        self._wave_cells = (_ring(self.spawn_distance), _ring(self._far_distance))
        self._spawn_cells = tuple(sorted(set(self._wave_cells[0] + self._wave_cells[1])))
        for d, cells in ((self.spawn_distance, self._wave_cells[0]),
                         (self._far_distance, self._wave_cells[1])):
            if not cells:
                raise ValueError(
                    f"no cell at Manhattan distance {d} from the staging cell "
                    f"{self.staging} fits below the zone row of a "
                    f"{self.G_W}x{self.G_H} grid"
                )
        self._max_staging_dist = max(
            abs(x - sx) + abs(y - sy)
            for y in range(self._flight_rows)
            for x in range(self.G_W)
        )

        # Quota one-hot over the values a quota can actually take (1..4 for the
        # headline N=6, m=3), so no slot is permanently zero.
        self._quota_values = tuple(sorted({int(k) for q in self._quota_space for k in q}))
        self._quota_slot = {k: i for i, k in enumerate(self._quota_values)}
        self._n_onehot = self.n_zones * len(self._quota_values)

        # Observation: quota one-hot, own at_staging, own comm bit, and -- only if
        # expose_own_wave -- the spawn ring (masked to 0 once regrouped).
        self.obs_dim = self._n_onehot + 2 + int(self.expose_own_wave)
        # State (critic only): quota one-hot, every agent's distance to the
        # staging cell, and the round counter.
        self.state_dim = self._n_onehot + self.n_agents + 1
        self.observation_space = spaces.Tuple(
            tuple(
                spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
                for _ in range(self.n_agents)
            )
        )

        # Episode state (set by reset).
        self.pos: np.ndarray | None = None
        self._wave: np.ndarray | None = None
        self._hist: list | None = None
        self._traj: list | None = None
        self._first_arrival: np.ndarray | None = None
        self._diag: dict | None = None

    def __repr__(self):
        return (
            f"<TeamPartitionEnv N={self.n_agents} zones={self.n_zones} "
            f"grid={self.G_W}x{self.G_H} spawn_distance={self.spawn_distance} "
            f"wave_gap={self.wave_gap} near_wave_size={self.near_wave_size} "
            f"interference={self.interference_prob} "
            f"transit_interference={self.transit_interference_prob}>"
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _dists(self) -> np.ndarray:
        sx, sy = self.staging
        return np.abs(self.pos[:, 0] - sx) + np.abs(self.pos[:, 1] - sy)

    def _regrouped(self) -> bool:
        return bool(np.all(self._dists() == 0))

    def _n_present(self) -> int:
        """Agents standing on the staging cell. aerial's n_free_connected."""
        return int(np.count_nonzero(self._dists() == 0))

    def _gate_open(self) -> bool:
        """Is a GO live this round? True once commit_quorum agents have arrived.

        At the default quorum (n_agents - n_spare, i.e. N when n_spare = 0) this
        is exactly `_regrouped`; with a spare it opens EARLY, on a subset whose
        membership varies, which is the property that stops a static
        (quota, id) table from being optimal. See commit_quorum in __init__.
        """
        return self._n_present() >= self.commit_quorum

    def _step_toward_staging(self, i: int):
        """One step toward the staging cell. The controller of aerial/charging:
        x wins ties. The staging cell is on the bottom row and every flight cell
        lies below the zone row, so the step stays on the grid and never enters
        the zone row."""
        x, y = int(self.pos[i, 0]), int(self.pos[i, 1])
        sx, sy = self.staging
        dx, dy = sx - x, sy - y
        if dx == 0 and dy == 0:
            return
        if abs(dx) >= abs(dy):
            x += 1 if dx > 0 else -1
        else:
            y += 1 if dy > 0 else -1
        self.pos[i] = (x, y)

    def _interfere(self, i: int, prob: float | None = None) -> bool:
        """Signal interference: with probability `prob` (default
        interference_prob) the agent loses its fix and ends the round one cell
        off, uniformly among the valid neighbours. Returns True if it fired.

        Agents already on the staging cell are charged staging_drift_prob
        instead -- see the note in __init__ for why the two rates are separate.
        """
        if prob is None:
            prob = self.interference_prob
        if self._rng.random() >= prob:
            return False
        x, y = int(self.pos[i, 0]), int(self.pos[i, 1])
        options = [
            (x + ddx, y + ddy)
            for ddx, ddy in ((0, 1), (0, -1), (1, 0), (-1, 0))
            if 0 <= x + ddx < self.G_W and 0 <= y + ddy < self._flight_rows
        ]
        self.pos[i] = options[int(self._rng.integers(len(options)))]
        return True

    def _transit(self, zone: int) -> int:
        """Transit interference: the agent lands in its chosen zone, or with
        probability transit_interference_prob in a uniformly random other one."""
        if (self.transit_interference_prob > 0.0
                and self._rng.random() < self.transit_interference_prob):
            others = [z for z in range(self.n_zones) if z != zone]
            return others[int(self._rng.integers(len(others)))]
        return zone

    def _record_observation_round(self):
        """Log what the agents observe at the start of a round."""
        at = self._dists() == 0
        for i in range(self.n_agents):
            self._hist[i].append(int(at[i]))
            if at[i] and self._first_arrival[i] < 0:
                self._first_arrival[i] = self._t
        self._traj.append(self.pos.copy())
        # Only a regroup at the start of a round that can still be played
        # counts; one first observed after the horizon had no round left.
        if at.all() and self._diag["regroup_round"] < 0 and self._t < self._time_limit:
            self._diag["regroup_round"] = self._t
        # Under a sub-roster quorum the ALL-N regroup is no longer the event the
        # episode turns on -- the gate opening is. Tracked separately so the two
        # readings stay distinguishable in the logs: `regrouped_ever` still means
        # "every agent stood on the staging cell at once", `gate_open_ever` means
        # "the team was ever able to act". They coincide when commit_quorum is
        # the full roster.
        if (int(at.sum()) >= self.commit_quorum and self._diag["gate_round"] < 0
                and self._t < self._time_limit):
            self._diag["gate_round"] = self._t

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        # v0's reset draws the quota and the zone columns and then calls
        # get_obs(), which in v2 needs positions. A provisional all-at-staging
        # placement satisfies it; it is overwritten below and never returned.
        self.pos = np.tile(np.asarray(self.staging, dtype=int), (self.n_agents, 1))
        self._in_reset = True          # render only: no frame of the provisional layout
        super().reset(seed=seed, options=options)
        self._in_reset = False

        # RNG order: quota, zone columns (both inside v0's reset), then WHICH
        # agents are in the near wave, then a spawn cell inside each ring.
        #
        # EXACTLY near_wave_size agents are near, drawn without replacement. The
        # SIZE is therefore a constant of the environment -- public, known to
        # every policy before the episode starts -- while WHICH agents hold
        # those slots is private. That combination is the whole point; see
        # "WHERE THE SYMMETRY BREAK COMES FROM" in the module docstring. Drawing
        # each agent's wave independently instead (the first draft did) provably
        # yields NOTHING: the waves are then i.i.d., so agent i's wave carries no
        # information about the others', the agents stay independent, X is again
        # Multinomial(N, q) with q the wave-mixture, and max_p P(X = K) is
        # exactly 0.145062 -- verified numerically at 100.0% of the ceiling for
        # near_wave_prob in {0.15, 0.25, 0.35, 0.50}.
        near = np.zeros(self.n_agents, dtype=bool)
        if self.near_wave_size > 0:
            near[self._rng.choice(self.n_agents, size=self.near_wave_size,
                                  replace=False)] = True
        self._wave = near.astype(int)
        cells = []
        for i in range(self.n_agents):
            ring = self._wave_cells[0] if near[i] else self._wave_cells[1]
            cells.append(ring[int(self._rng.integers(len(ring)))])
        self.pos = np.array(cells, dtype=int)

        # Which agents read the manifest this episode. Fixed public size, fresh
        # private membership -- drawn AFTER the wave so the two draws stay
        # independent and either can be turned off without moving the other.
        if self.n_quota_readers >= self.n_agents:
            self._quota_readers = np.ones(self.n_agents, dtype=bool)
        else:
            readers = np.zeros(self.n_agents, dtype=bool)
            readers[self._rng.choice(self.n_agents, size=self.n_quota_readers,
                                     replace=False)] = True
            self._quota_readers = readers

        self._hist = [[] for _ in range(self.n_agents)]
        self._traj = []
        self._first_arrival = np.full(self.n_agents, -1, dtype=int)
        self._diag = {"approach_rounds": 0, "interference_hits": 0, "regroup_round": -1,
                      "gate_round": -1, "blocked_commits": 0, "dodge_rounds": 0}
        self._record_observation_round()
        self._render_step, self._render_return = None, 0.0
        if self.render_flag:
            self.render()
        return self.get_obs(), {}

    def step(self, actions):
        a = [int(x) for x in actions]
        if len(a) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} actions, got {len(a)}")
        if any(x < 0 or x >= self.n_actions for x in a):
            raise ValueError(f"actions out of range [0, {self.n_actions - 1}]: {a}")

        d_before = self._dists()
        at_staging = d_before == 0
        regrouped = bool(at_staging.all())
        # What the GATE reads. Equal to `regrouped` at the default quorum; with
        # a spare it opens on a varying SUBSET, which is aerial's structure and
        # the thing that denies an id-keyed lookup table. See commit_quorum.
        gate_open = int(at_staging.sum()) >= self.commit_quorum

        # A GO_c resolves the attempt only once the team has REGROUPED. Before
        # that a GO is inert -- exactly as a GO issued while still in flight
        # already was -- and the agent simply holds its cell for the round.
        #
        # WHY (measured, 2026-09-12). The rule used to be v0's "any GO from the
        # staging cell resolves". On the EPyMARL path mappo/tarmac/commnet are
        # evaluated by SAMPLING (soft_policies ignores test_greedy), so residual
        # policy entropy makes an already-arrived agent draw GO while stragglers
        # are still inbound. That ends the episode with n_committed < N, which a
        # quota summing to N can never match. The damage grows with the arrival
        # spread, so it scaled with interference_prob and swamped the thing this
        # benchmark exists to measure:
        #
        #   interference  regrouped_ever  exact-match   P(regroup) x 0.1451
        #     0.03            0.70-0.81     0.081-0.115        0.101-0.117
        #     0.20            0.25          0.018              0.036
        #
        # i.e. exact-match was P(regroup) x the symmetric ceiling almost exactly:
        # the split itself was already being played at its ceiling and every
        # point of the shortfall was navigation. Gating restores that (scripted
        # probe, 3000 eps/row, an arrived agent defecting to GO w.p. eps_go):
        #
        #   interf  eps_go |  regroup  exact   vs ceiling
        #     0.20    0.15 |   0.4973  0.0717      49.4%    <- old rule
        #     0.20    0.15 |   1.0000  0.1457     100.4%    <- gated
        #
        # The ceiling itself is untouched. At the coordination state the game is
        # still one attempt: the first round in which anyone GOes resolves it,
        # and a round in which nobody does is v0's all-STAY decline, which
        # multistep_ceiling already proves cannot raise max_p E[return] for any
        # horizon. Gating only removes a failure mode that lives BEFORE the
        # coordination state and has nothing to do with Condition 3.
        if self.commit_gate:
            # Only an agent that has ARRIVED can depart -- at the full-roster
            # quorum that is everyone, so this reduces to the old rule, but with
            # a spare the stragglers must be excluded explicitly.
            goers = ([i for i in range(self.n_agents)
                      if at_staging[i] and 1 <= a[i] <= self.n_zones]
                     if gate_open else [])
            blocked = 0 if gate_open else sum(
                1 for i in range(self.n_agents)
                if at_staging[i] and 1 <= a[i] <= self.n_zones
            )
        else:
            # aerial's rule: a GO from the staging cell resolves the attempt
            # whenever it is issued. A GO from an agent still in flight has no
            # effect (it cannot depart from where it is not).
            goers = [i for i in range(self.n_agents)
                     if at_staging[i] and 1 <= a[i] <= self.n_zones]
            blocked = 0
        misroutes = 0
        self._diag["blocked_commits"] += blocked
        if gate_open and not goers:
            self._diag["dodge_rounds"] += 1

        if goers:
            # v0's resolution: the realised count vector against the quota. An
            # agent that did not go is in no zone, so going before everyone has
            # arrived cannot meet a quota summing to N. Transit interference may
            # move an agent to another zone on the way.
            landed = {}
            for i in goers:
                zone = self._transit(a[i] - 1)
                misroutes += int(zone != a[i] - 1)
                landed[i] = zone
            counts = tuple(
                sum(1 for i in goers if landed[i] == c) for c in range(self.n_zones)
            )
            exact = counts == self.quota
            # GRADED, as charging_coordination: per-unit credit for each zone
            # whose count matches, plus a completion bonus only when every zone
            # does. Both terms read the count vector alone and are jointly
            # maximised only at X = K, so Condition 1 is untouched; what changes
            # is that PARTIAL competence now earns partial return, which is what
            # spreads the arms along the return axis instead of collapsing them
            # onto two discrete values.
            n_matched = sum(1 for c in range(self.n_zones)
                            if counts[c] == self.quota[c])
            reward = (
                self.zone_reward * n_matched
                + self.all_reward * float(exact)
                - self.wrong_penalty * float(not exact)
                - float(np.dot(self.lam, counts))
            )
            terminated = True
            nobody_moved = False
            self._last_actions = [landed[i] + 1 if i in landed else self.STAY
                                  for i in range(self.n_agents)]
        else:
            counts = (0,) * self.n_zones
            exact = False
            movers = [i for i in range(self.n_agents)
                      if not at_staging[i] and a[i] == self.MOVE]
            for i in movers:
                self._step_toward_staging(i)
            # v0's all-STAY round: nobody moved and nobody went. A GO_c issued
            # while flying in also leaves the agent where it is.
            nobody_moved = not movers

            # Only agents in flight are exposed; the staging cell holds the rest.
            hits = 0
            in_flight_now = self._dists() > 0
            for i in range(self.n_agents):
                if in_flight_now[i]:
                    hits += int(self._interfere(i, self.interference_prob))
                elif self.drift_at_staging:
                    hits += int(self._interfere(i, self.staging_drift_prob))
            self._diag["interference_hits"] += hits
            if movers:
                self._diag["approach_rounds"] += 1

            if self.step_cost_far is None:
                # Original rule: charged only when the round was a full stall.
                reward = -self.step_cost if nobody_moved else 0.0
            else:
                # aerial's rule: charged EVERY round, scaled by how far the team
                # still is from the staging cell, so idling is never free
                # anywhere on the grid and roaming costs strictly more than
                # waiting at the cell. Computed on the distances AFTER the moves
                # and the interference, i.e. on the state the round ends in.
                frac = float(self._dists().mean()) / self._max_staging_dist
                reward = -(self.step_cost
                           + (self.step_cost_far - self.step_cost) * frac)
            # An assembled team that declines pays this on top. Under the
            # distance-scaled rule it should be 0: aerial deleted its own
            # ready-wait penalty because charging the team for being assembled
            # ranked "wait on the target" below "roam around".
            if gate_open and not goers:
                reward -= self.dodge_penalty
            if self.shaping_weight > 0.0:
                # Potential-based on the full transition (movement AND
                # interference), so it telescopes exactly to
                # w * (sum of start distances - sum of final distances).
                reward += self.shaping_weight * float(d_before.sum() - self._dists().sum())
            terminated = False
            self._last_actions = None

        self._last_counts = counts
        self._t += 1
        if nobody_moved:
            self._n_stalls += 1

        if not terminated:
            self._record_observation_round()

        # The env enforces its own horizon. On the MAT path there is no gymnasium
        # TimeLimit around it, so this line is what ends a stalling episode.
        # The wave passes: once the gate has been open for commit_window rounds
        # the opportunity is gone and the episode ends as a miss.
        window_expired = (
            self.commit_window is not None
            and self._diag["gate_round"] >= 0
            and self._t - self._diag["gate_round"] >= self.commit_window
        )
        # THE GATE ITSELF IS THE OPPORTUNITY: if the quorum was never assembled
        # by the deadline, the moment has passed. See gate_deadline in
        # __init__ -- this is what denies the elected-committee exploit, by
        # making it impossible to reach the quorum unless EVERY agent rushed,
        # which in turn makes the deciding set the environment's draw rather
        # than the policy's choice.
        gate_missed = (self._diag["gate_round"] < 0
                       and self._t >= self.gate_deadline)
        truncated = (not terminated) and (self._t >= self._time_limit
                                          or window_expired or gate_missed)
        if truncated and self.miss_penalty > 0.0:
            reward -= self.miss_penalty

        arrivals = self._first_arrival[self._first_arrival >= 0]
        # EVERY value must be a summable scalar (parallel_runner sums infos over
        # the union of keys); vectors are exploded per zone, as in v0.
        info = {
            "exact_match": float(exact),
            # Stable-Baselines3's name for the same flag: its EvalCallback records
            # a success curve from it, which is how the CTCE arm gets one.
            "is_success": float(exact),
            "n_committed": int(len(goers)),
            # How many agents were ON the staging cell this round. Under a
            # sub-roster quorum this is the size of the acting set, and the gap
            # between it and n_agents is the thing an id-keyed table cannot
            # track. Equal to n_agents whenever the quorum is the full roster.
            "n_present": float(at_staging.sum()),
            "abs_error": float(sum(abs(x - k) for x, k in zip(counts, self.quota))),
            "no_attempt": float(not terminated),
            "stall_rounds": float(self._n_stalls),
            # Reference lines: the memoryless symmetric ceiling (v0's closed
            # form) and a perfectly coordinated team's ceiling under transit
            # interference, both for this episode's quota.
            # !! THE CEILING IS OVER THE AGENTS THAT ACTUALLY DECIDE, not the
            # roster. With a sub-roster quorum the absent agents are not
            # choosing among {zones, STAY} -- they are simply not there, and
            # pricing them as choosers understates the barrier badly. Measured
            # 2026-09-14 on the shipped preset (N=8, quorum 6, sum K = 6):
            # over the roster 0.0452, over the deciding six 0.1451, and a
            # scripted symmetric policy at p = K/6 lands on 0.1477 over 4000
            # episodes. 0.1451 is v0's 0.145062 -- the barrier is preserved,
            # which is the whole point of keeping sum(K) = commit_quorum.
            "analytic_ceiling": max_exact_prob(self.commit_quorum, self.quota),
            "coordinated_cap": coordinated_success_prob(self.quota,
                                                        self.transit_interference_prob),
            # Gating makes a resolve imply a regroup, so premature_commit is
            # now identically 0; it is kept so old and new logs line up. The
            # live signal is blocked_commits: GOes the policy issued before the
            # team had assembled, which the env absorbed instead of scoring.
            "premature_commit": float(terminated and not gate_open),
            "regrouped_at_commit": float(terminated and gate_open),
            "blocked_commits": float(self._diag["blocked_commits"]),
            # Zones whose realised count matched, 0..n_zones. The graded part of
            # the reward; exact_match is the all-zones case of it.
            "zones_matched": float(
                sum(1 for c in range(self.n_zones) if counts[c] == self.quota[c])
            ),
            "dodge_rounds": float(self._diag["dodge_rounds"]),
            "approach_rounds": float(self._diag["approach_rounds"]),
            # regroup_round reads time_limit when the team never regrouped in a
            # playable round; regrouped_ever is the unambiguous flag.
            "regrouped_ever": float(self._diag["regroup_round"] >= 0),
            "regroup_round": float(self._diag["regroup_round"]
                                   if self._diag["regroup_round"] >= 0
                                   else self._time_limit),
            "gate_open_ever": float(self._diag["gate_round"] >= 0),
            # 1.0 when the quorum was never assembled in time -- the
            # signature of a team that held agents back (the exploit).
            "gate_missed": float(self._diag["gate_round"] < 0
                                 and self._t >= self.gate_deadline),
            "gate_round": float(self._diag["gate_round"]
                                if self._diag["gate_round"] >= 0
                                else self._time_limit),
            "interference_hits": float(self._diag["interference_hits"]),
            "transit_misroutes": float(misroutes),
            "arrival_spread": float(arrivals.max() - arrivals.min()) if arrivals.size > 1 else 0.0,
            # Size of the near (early) cohort. Constant by construction -- logged
            # so a config change shows up in the curves rather than silently.
            "near_wave_size": float(self._wave.sum()) if self._wave is not None else 0.0,
            # Distinct observation histories among the agents. 1 = identical
            # histories, v0's situation.
            "n_hist_classes": float(len({tuple(h) for h in self._hist})),
        }
        for c in range(self.n_zones):
            info[f"count_{c}"] = int(counts[c])
            info[f"quota_{c}"] = int(self.quota[c])

        # Render only: what this step did, for the live frame.
        self._render_return += float(reward)
        self._render_step = dict(
            actions=list(a), goers=list(goers), blocked=int(blocked),
            reward=float(reward), terminated=bool(terminated),
            truncated=bool(truncated), window_expired=bool(window_expired),
            gate_missed=bool(gate_missed), exact=bool(exact))

        if self.render_flag:
            self.render()

        return self.get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observations -- Condition 3 at the coordination state
    # ------------------------------------------------------------------

    def _quota_onehot(self) -> np.ndarray:
        onehot = np.zeros((self.n_zones, len(self._quota_values)), dtype=np.float32)
        for c, k in enumerate(self.quota):
            onehot[c, self._quota_slot[int(k)]] = 1.0
        return onehot.reshape(-1)

    def _in_disk(self) -> np.ndarray:
        """Agents inside the staging cell's radio disk (Manhattan comm_radius).

        DELIBERATELY WIDER THAN THE DOCK, exactly as aerial_ordering runs
        comm_radius 3 against a lift that requires standing on the target cell.
        An agent on final approach therefore hears the negotiation a step or two
        before it can act on it, so the readiness count OVER-REPORTS: it says
        "ready" while some of those it counted still cannot depart. That gap is
        what stops a timing rule from being read straight off the signal.
        Set comm_radius=0 to collapse the disk onto the staging cell and make
        the count exact.
        """
        return self._dists() <= self.comm_radius

    def get_comm_mask(self):
        """Range-based, as aerial_ordering and charging_rendezvous.

        bit[i] = 1 iff agent i is inside the staging cell's radio disk and at
        least one other agent is too, so the communication clique is exactly the
        set of agents that have arrived and can still act, and it GROWS IN WAVES
        as the cohorts come in (charging's phrase for the same structure). An
        agent still far out is a singleton and cannot pre-negotiate with anyone.

        v0 returns all-ones on purpose -- with Condition 3 holding everywhere,
        communication is provably useless there, and that is the experiment. v2
        is the applied member of the pair, so it carries the live graph the
        SDec-POMDP framework is about: a decode chain restricted to this clique
        can count its own members, which is precisely what independent actors
        cannot do however many agent ids they are given.
        """
        if self.pos is None:
            return np.zeros((self.n_agents, 1), dtype=np.float32)
        disk = self._in_disk()
        bits = disk & (int(disk.sum()) >= 2)
        return bits.astype(np.float32).reshape(self.n_agents, 1)

    def get_obs_agent(self, agent_id: int):
        """[quota one-hot, own at_staging, regrouped, own wave].

        Two entries are agent-specific and BOTH are switched off by the regroup:

        at_staging  1 once this agent has reached the staging cell, so it is 1
                    for everyone exactly when regrouped is.
        own_wave    1 if this agent started on the near ring, but ONLY while the
                    team has not yet assembled: it is multiplied by
                    (1 - regrouped). The wave is announced on the way IN, not at
                    the decision.

        So at the coordination state every observation is again bit-identical
        (at_staging = regrouped = 1, own_wave = 0, the rest public) and
        Condition 3 holds -- `get_obs` asserts it. The break therefore lives in
        the HISTORY, not in the decision-time observation, which is what makes
        it invisible to a memoryless policy and keeps 0.145062 a live reference
        line on the same plot.

        The field is exposed rather than left to be inferred from arrival timing
        because timing is confounded by the policy's own MOVE choices: a diffuse
        policy fragments its arrivals into 5+ history classes (measured on MAPPO
        at 20k steps) and the clean two-cohort signal is lost in its own noise.
        """
        if self.quota is None or self.pos is None:
            return np.zeros(self.obs_dim, dtype=np.float32)
        at = self._dists() == 0
        regrouped = float(at.all())
        # Field 2 is the agent's OWN communication bit by default -- it is in the
        # clique iff it has arrived and at least one peer has too (aerial's
        # `in_comm` / charging's own comm bit). With public_regrouped=True it is
        # instead the exact public "everyone is here" flag, which hands the
        # timing to every policy for free and is what let MAPPO+id solve v2.
        if self.public_regrouped == "count":
            # aerial's n_free_connected: HOW MANY are assembled, not whether
            # everyone is. The distinction is load-bearing once the gate opens
            # on a subset -- a boolean "all here" flag hands the policy a
            # roster it can key a static table on, whereas a count leaves WHICH
            # agents are present unobservable. Gated on the radio disk, so a
            # disconnected agent reads 0.0, exactly as aerial's field 3 does.
            field2 = (float(at.sum()) / self.n_agents
                      if self._dists()[agent_id] <= self.comm_radius else 0.0)
        elif self.public_regrouped:
            field2 = regrouped
        else:
            field2 = float(at[agent_id] and int(at.sum()) >= 2)
        tail = [float(at[agent_id]), field2]
        if self.expose_own_wave:
            own_wave = (float(self._wave[agent_id]) * (1.0 - regrouped)
                        if self._wave is not None else 0.0)
            tail.append(own_wave)
        # aerial's hidden requirement: an agent that did not read the manifest
        # sees the sentinel in every quota slot, exactly as one of its workers
        # reads target_requirement = -1.0 when out of comm range of the sensor.
        if self._quota_readers is not None and not self._quota_readers[agent_id]:
            quota_field = np.full(self.n_zones * len(self._quota_values), -1.0,
                                  dtype=np.float32)
        else:
            quota_field = self._quota_onehot()
        return np.concatenate([
            quota_field,
            np.array(tail, dtype=np.float32),
        ]).astype(np.float32)

    def get_obs(self):
        """All agent observations, with Condition 3 enforced where it must hold.

        Built per agent through `get_obs_agent` for the same reason v0 does it:
        so that the tripwire exercises the code a future edit would touch. On
        the way in the observations legitimately differ (at_staging); once the
        team has regrouped -- the coordination state s of the theory -- they
        must be identical, and this raises if they are not.
        """
        obs = tuple(self.get_obs_agent(i) for i in range(self.n_agents))
        if self.symmetry_assert and self.pos is not None and self._gate_open():
            # With a hidden quota the team splits into two OBSERVATION CLASSES
            # of publicly known sizes (readers and non-readers), so Condition 3
            # can only be asserted within a class -- the flag voids it across
            # classes deliberately, the way obs_agent_id does. With every agent
            # reading the manifest this is the original all-pairs check.
            readers = (self._quota_readers if self._quota_readers is not None
                       else np.ones(self.n_agents, dtype=bool))
            # A sub-roster quorum splits the team a SECOND way: the agents on
            # the staging cell are the ones facing the decision, the stragglers
            # are not, and their at_staging bits legitimately differ. Condition
            # 3 is the claim that the DECIDING agents cannot tell each other
            # apart, so it is asserted within the present class -- exactly as
            # aerial's workers on the target cell all read the same 5 fields
            # while a worker still in transit does not.
            present = self._dists() == 0
            for cls in (True, False):
                members = [i for i in range(self.n_agents)
                           if bool(readers[i]) is cls and present[i]]
                for i in members[1:]:
                    if not np.array_equal(obs[members[0]], obs[i]):
                        raise AssertionError(
                            "Condition 3 violated at the regrouped state: "
                            f"observations differ between agent {members[0]} and "
                            f"agent {i}, which are in the same visibility class. "
                            "The split must be decided from identical observations."
                        )
        return obs

    def get_state(self):
        """Critic-only state: the quota, every agent's distance to the staging
        cell, and the round counter."""
        if self.quota is None or self.pos is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        return np.concatenate([
            self._quota_onehot(),
            (self._dists() / self._max_staging_dist).astype(np.float32),
            np.array([self._t / self._time_limit], dtype=np.float32),
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # Trajectories (render and figures)
    # ------------------------------------------------------------------

    def transit_trajectory(self):
        """The decision-free walk each agent made to the zone it landed in,
        (T+1, N, 2). Agents that did not go stay where they were."""
        if self._last_actions is None:
            raise RuntimeError("no GO yet -- the episode has not resolved")
        sx, sy = self.staging
        paths = []
        for i, act in enumerate(self._last_actions):
            here = (int(self.pos[i, 0]), int(self.pos[i, 1]))
            if act == self.STAY:
                paths.append([here])
                continue
            zx = self.zone_cols[act - 1]
            path = [(sx, sy)]
            for y in range(sy + 1, self._zone_row + 1):
                path.append((sx, y))
            step = 1 if zx >= sx else -1
            for x in range(sx + step, zx + step, step):
                path.append((x, self._zone_row))
            paths.append(path)
        horizon = max(len(p) for p in paths)
        out = np.zeros((horizon, self.n_agents, 2), dtype=np.int32)
        for i, path in enumerate(paths):
            for t in range(horizon):
                out[t, i] = path[min(t, len(path) - 1)]
        return out

    def episode_trajectory(self):
        """Every position of the episode: the approach, then the transit to the
        zones if the attempt resolved. Shape (T, N, 2)."""
        approach = np.stack(self._traj).astype(np.int32)
        if self._last_actions is None:
            return approach
        return np.concatenate([approach, self.transit_trajectory()[1:]], axis=0)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_kit(self):
        """The shared drawing toolkit (mac/envs/render.py), imported lazily so
        that matplotlib is only needed when rendering."""
        kit = getattr(self, "_render_kit_cache", None)
        if kit is None:
            from . import render as kit

            self._render_kit_cache = kit
        return kit

    @staticmethod
    def _spread(n, k, radius):
        """Offset for the k-th of n agents sharing one cell."""
        if n <= 1:
            return 0.0, 0.0
        ang = 2.0 * np.pi * k / n + np.pi / 2.0
        return radius * np.cos(ang), radius * np.sin(ang)

    def _clipped(self, ax, draw):
        """Run `draw(ax)` and clip whatever it added to the playing field, so a
        radio diamond around the bottom-row staging cell stays on the grid."""
        from matplotlib.patches import Rectangle

        n_patches, n_lines = len(ax.patches), len(ax.lines)
        draw(ax)
        box = Rectangle((-0.5, -0.5), self.G_W, self.G_H, transform=ax.transData)
        for art in list(ax.patches[n_patches:]) + list(ax.lines[n_lines:]):
            art.set_clip_path(box)

    def render(self):
        """Live view of the current step, in the style of the other MAC envs.

        Call it after reset() and after every step(): each call draws one frame
        (a window, or a PNG in `render_save_dir`). The frame shows
          * the three zones on the top row with their public quotas, and once
            the team has split, how many agents landed in each;
          * the staging cell, its radio range, and a link from every agent in
            the communication group to it;
          * every agent, with a short trail of where it came from; agents on
            the staging cell are marked, and an agent whose GO was ignored
            because the gate was still closed is tagged;
          * the gate: closed (with the step it must open by), open (the team
            has its commit window), or passed;
          * on the split itself, each departing agent's path to the zone it
            landed in, and the outcome.
        `render_summary()` draws the whole episode in one static figure.
        """
        if self._in_reset or self.pos is None:
            return None
        dr = self._render_kit()
        fig = getattr(self, "_fig", None)
        if fig is None:
            fig, ax, th = dr.stage(figsize=(9.6, 8.2), theme_name=self.render_theme,
                                   save_dir=self.render_save_dir,
                                   window_title="Team Partition")
            self._fig, self._ax = fig, ax
        else:
            ax = self._ax
            th = dr.reuse(fig, ax, self.render_theme)
        fig.texts.clear()
        fig.patches.clear()

        from collections import defaultdict
        from matplotlib.patches import FancyBboxPatch

        W, H, zr = self.G_W, self.G_H, self._zone_row
        sx, sy = self.staging
        last = self._render_step or {}
        resolved = bool(last.get("terminated"))
        landed = self._last_actions if resolved else None   # zone + 1 per goer
        # Agents that split have left the staging cell, although the env keeps
        # their last position there (the transit is decision-free).
        present = self._n_present() - (len(last.get("goers", [])) if resolved else 0)
        gate_open = self._gate_open() and not (last.get("terminated") or last.get("truncated"))
        dr.field(ax, 0, W - 1, 0, H - 1, th, grid_step=1, major_every=5, pad=0.62,
                 terrain=self.render_terrain, terrain_seed=self.render_terrain_seed,
                 terrain_contrast=self.render_terrain_contrast)

        # --- zones: coloured pads with their public quota --------------------
        for c, col in enumerate(self.zone_cols):
            colour = self.ZONE_COLOURS[c % len(self.ZONE_COLOURS)]
            ax.add_patch(FancyBboxPatch(
                (col - 0.44, zr - 0.44), 0.88, 0.88, boxstyle="round,pad=0,rounding_size=0.12",
                facecolor=colour, edgecolor=colour, lw=1.6, alpha=0.38, zorder=1.5))
            if resolved:
                got, need = self._last_counts[c], self.quota[c]
                tag, tc = f"{got}/{need}", th["ok"] if got == need else th["bad"]
            else:
                tag, tc = f"K={self.quota[c]}", th["ink"]
            ax.text(col, zr + 0.58, tag, ha="center", va="bottom", color=tc,
                    fontsize=9.5, fontweight="bold", zorder=9,
                    path_effects=dr.stroke(th["field_top"], 2.8))

        # --- staging cell and its radio range --------------------------------
        self._clipped(ax, lambda a: dr.comm_disk(a, sx, sy, self.comm_radius, th,
                                                 manhattan=True))
        ax.add_patch(FancyBboxPatch(
            (sx - 0.46, sy - 0.46), 0.92, 0.92, boxstyle="round,pad=0,rounding_size=0.10",
            facecolor=th["ok"] if gate_open else th["panel"],
            edgecolor=th["ok"] if present >= self.commit_quorum else th["panel_edge"],
            lw=1.6, alpha=0.30 if gate_open else 0.75, zorder=1.6))
        ax.text(sx, sy - 0.56, "staging", ha="center", va="top", color=th["muted"],
                fontsize=7.4, fontweight="bold", zorder=9,
                path_effects=dr.stroke(th["field_top"], 2.4))

        # --- trails and radio links ------------------------------------------
        comm = np.asarray(self.get_comm_mask()).reshape(-1) > 0.5
        hist = list(self._traj or []) + [self.pos]
        if self.render_trail > 1:
            for i in range(self.n_agents):
                pts = [tuple(int(v) for v in p[i]) for p in hist[-self.render_trail:]]
                if len(set(pts)) > 1:
                    dr.trail(ax, pts, dr.agent_color(th, i), th, alpha=0.55)
        for i in range(self.n_agents):
            p = (int(self.pos[i, 0]), int(self.pos[i, 1]))
            if comm[i] and p != (sx, sy) and not (landed and landed[i] != self.STAY):
                dr.comm_link(ax, p, (sx, sy), th, alpha=0.55, lw=1.1)

        # --- the split: each departing agent's path to its zone --------------
        where = {}
        if landed:
            for i, act in enumerate(landed):
                if act == self.STAY:
                    continue
                zone = act - 1
                zx = self.zone_cols[zone]
                colour = self.ZONE_COLOURS[zone % len(self.ZONE_COLOURS)]
                off = 0.09 * (zone - (self.n_zones - 1) / 2.0)   # keep the paths apart
                ax.plot([sx + off, sx + off, zx], [sy, zr + off, zr + off], color=colour,
                        lw=2.0, alpha=0.6, ls=(0, (4, 3)), solid_capstyle="round", zorder=3)
                where[i] = (zx, zr)

        # --- agents ----------------------------------------------------------
        by_cell = defaultdict(list)
        for i in range(self.n_agents):
            by_cell[where.get(i, (int(self.pos[i, 0]), int(self.pos[i, 1])))].append(i)
        ignored = (set(i for i, x in enumerate(last.get("actions", []))
                       if 1 <= x <= self.n_zones) if last.get("blocked") else set())
        for cell, agents in by_cell.items():
            n_here = len(agents)
            ring, glyph = ((0.0, 0.27) if n_here == 1 else
                           (0.30, 0.17) if n_here <= 4 else (0.36, 0.13))
            for k, i in enumerate(agents):
                dx, dy = self._spread(n_here, k, ring)
                x, y = cell[0] + dx, cell[1] + dy
                at_staging = cell == (sx, sy)
                dr.drone(ax, x, y, size=glyph, color=dr.agent_color(th, i), th=th,
                         label=str(i + 1), halo=th["ok"] if at_staging else None,
                         halo_alpha=0.22, zorder=7)
                if i in ignored:
                    ax.text(x, y + glyph * 1.7, "GO ignored", ha="center", va="bottom",
                            color=th["warn"], fontsize=5.8, fontweight="bold", zorder=8,
                            path_effects=dr.stroke(th["field_top"], 2.2))

        # --- HUD -------------------------------------------------------------
        dr.header(
            fig, ax, "Team Partition",
            f"{self.n_agents} interchangeable agents · {self.n_zones} zones · quota "
            f"sums to {self.commit_quorum} ({self.n_agents - self.commit_quorum} spare) · "
            f"the gate opens when {self.commit_quorum} are on the staging cell · "
            f"radio r={self.comm_radius}",
            th,
        )
        gate_round = self._diag["gate_round"] if self._diag else -1
        if last.get("terminated") or last.get("truncated"):
            gate = ("passed", th["muted"])
        elif gate_open:
            left = (gate_round + self.commit_window - self._t
                    if self.commit_window is not None else None)
            gate = ("OPEN · split now" if left in (None, 1) else f"OPEN · {left} steps",
                    th["ok"])
        else:
            gate = (f"closed (by t={self.gate_deadline})", th["warn"])
        dr.stat_row(fig, [
            ("step", f"{self._t}/{self._time_limit}"),
            ("quota", " · ".join(str(k) for k in self.quota)),
            ("on staging", f"{present}/{self.commit_quorum}",
             th["ok"] if present >= self.commit_quorum else None),
            ("gate", gate[0], gate[1]),
            ("return", f"{self._render_return:+.2f}"),
        ], th)

        if resolved:
            counts = " ".join(f"{g}/{k}" for g, k in zip(self._last_counts, self.quota))
            pill = (f"EXACT SPLIT   {counts}", th["ok"]) if last.get("exact") else \
                   (f"WRONG SPLIT   {counts}", th["bad"])
        elif last.get("truncated"):
            why = ("the commit window passed" if last.get("window_expired") else
                   "the gate never opened" if last.get("gate_missed") else "time limit")
            pill = (f"MISSED   {why}", th["bad"])
        elif gate_open:
            pill = ("GATE OPEN   the agents present must split exactly to quota", th["ok"])
        else:
            pill = None
        if pill:
            dr.status_pill(fig, pill[0], pill[1], th)

        dr.legend_row(fig, [
            (dr.agent_color(th, 0), "agents"),
            (th["ok"], "on the staging cell"),
            (th["comm"], "radio link to the staging cell"),
            (self.ZONE_COLOURS[0], "zones (required count K)"),
        ], th)
        dr.timeline(fig, self._t, self._time_limit, th)
        dr.footer(fig, "Team Partition · MAC benchmark", th, right=self.render_note)

        path = dr.finish(
            fig, save_dir=self.render_save_dir,
            name=f"{self.render_prefix}_ep{max(0, self._episode_idx):02d}_t{self._frame_idx:03d}",
            dpi=self.render_dpi, pause=self.render_pause)
        self._frame_idx += 1
        if path:
            self._frame_paths.append(path)
        return path

    def render_summary(self, save_path: str | None = None, show: bool | None = None):
        """The whole episode in one static figure: map, start ring, approach
        trails, transits and the outcome.

        With `save_path` the figure is written to disk (e.g. the env-view panel
        of the paper) and not shown unless `show=True`.
        """
        try:
            import matplotlib
            if save_path is not None and not show:
                matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
        except ImportError:  # pragma: no cover
            print(f"[team-partition-v0] quota={self.quota} counts={self._last_counts}")
            return

        fig, ax = plt.subplots(figsize=(1.0 * self.G_W, 1.0 * self.G_H))
        ax.set_xlim(0, self.G_W)
        ax.set_ylim(0, self.G_H)
        ax.set_aspect("equal")
        ax.set_xticks(range(self.G_W + 1))
        ax.set_yticks(range(self.G_H + 1))
        ax.grid(True, color="#DDDDDD", linewidth=0.6)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        for (x, y) in self._spawn_cells:
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#BBBBBB", alpha=0.18, zorder=0))

        if self._traj:
            traj = self.episode_trajectory()
            for i in range(self.n_agents):
                jitter = 0.5 + 0.07 * (i - (self.n_agents - 1) / 2.0)
                xs = traj[:, i, 0] + jitter
                ys = traj[:, i, 1] + jitter
                ax.plot(xs, ys, color="#555555", alpha=0.5, linewidth=1.3, zorder=2)
                ax.plot(xs[0], ys[0], "s", color="#888888", markersize=4, zorder=3)
                ax.plot(xs[-1], ys[-1], "o", color="#222222", markersize=5, zorder=3)

        for c, col in enumerate(self.zone_cols):
            colour = self.ZONE_COLOURS[c % len(self.ZONE_COLOURS)]
            ax.add_patch(Rectangle((col, self._zone_row), 1, 1, facecolor=colour,
                                   alpha=0.55, zorder=1))
            got = "-" if self._last_counts is None else self._last_counts[c]
            ax.text(col + 0.5, self._zone_row + 0.5, f"{got}/{self.quota[c]}",
                    ha="center", va="center", fontsize=12, color="white",
                    fontweight="bold", zorder=4)

        sx, sy = self.staging
        ax.add_patch(Rectangle((sx, sy), 1, 1, facecolor="#999999", alpha=0.5, zorder=1))
        ax.text(sx + 0.5, sy + 0.15, "staging", ha="center", va="center", fontsize=7, zorder=4)

        title = f"quota={self.quota}  t={self._t}"
        if self._last_actions is not None:
            hit = "EXACT" if self._last_counts == self.quota else "MISS"
            title += f"   |   counts={self._last_counts}  {hit}"
        ax.set_title(title, fontsize=11)
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150)
        if show or (show is None and save_path is None):
            plt.show()
        plt.close(fig)

