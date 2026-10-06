"""Charging Rendezvous: drones arriving in waves must ration too few ports.

N interchangeable drones start on spawn rings around ONE charging station with
`capacity` ports at the centre of a grid. Each must fly in, charge to
`target_level` and leave before its battery runs dry. Drones arrive in cohorts
larger than the port count, so over and over the drones at the dock must decide
which exactly-C of them CONNECT -- if more ask than there are free ports, the
station rejects everyone.

The central design problem
--------------------------
The whole benchmark rests on an EXACT symmetry: identical battery + identical
drain => no observable sort key => permutation-equivariant CTDE (CommNet/TarMAC)
cannot produce exactly-C CONNECTs, while an autoregressive decoder (SDec-MAT) can.
Naive navigation DESTROYS that symmetry: drones starting at different distances
arrive at different times with different residual batteries, which hands the
policy a battery/rank sort-key that a *communicative* CTDE method can compute as
a symmetric function of messages (my_rank = #peers more needy than me) WITHOUT
any autoregressive decode. That would quietly defeat the benchmark.

Two ideas keep the stress test alive:

  1. STAGGER IN TIME, NOT IN RESIDUAL BATTERY. Each drone's starting battery is
     set to fund its own trip exactly:

         init_battery_i = base_arrival_battery + distance_i * drain_rate (+ jitter)

     A farther drone is dispatched with proportionally more charge, so NOBODY
     dies en route, and — with jitter sigma=0 — every drone ARRIVES at the same
     residual `base_arrival_battery`. Distances differ => arrival TIMES differ
     (staggered waves), but the battery a drone competes with at the dock does
     not => the sort-key gradient does not exist at the decision point.

  2. COHORTS bigger than the port count. Drones are partitioned into a few
     COHORTS, each sharing a spawn distance, so a cohort arrives together and
     tied. With cohort_size > capacity, the neediest present group is always
     larger than the free ports, so "pick exactly C among identical drones" —
     the thing only autoregressive decode solves — keeps recurring wave after
     wave. `base_arrival_battery` is sampled once per episode from a (low, high)
     range and applied to every drone, varying episode difficulty
     while preserving the exact within-cohort tie.

`battery_sigma` turns the residual heterogeneity into a TUNABLE knob:
sigma=0 => hardest (exact within-cohort ties); larger sigma => near-ties that a
rank-based CTDE can increasingly exploit — a clean ablation axis for "how much
symmetry-breaking before CTDE catches up to SDec-MAT?".

Communication (station-hub BITS, position-derived)
--------------------------------------------------
The comm mask is ONE BIT PER AGENT -- `get_comm_mask() -> (N, 1)`, the same
representation aerial_ordering.py uses -- not a pairwise [N, N] adjacency.
The charging station is the mesh hub (it is the ground radio), so

    comm_bit[i] = 1  iff  i is unsatisfied
                     AND  Manhattan(i, station) <= comm_radius
                     AND  at least one OTHER unsatisfied drone is also in range

Every consumer downstream (MAT's mat_comm_groups.py) turns per-agent bits into
the outer product bits (x) bits,
so the communication graph is ALWAYS exactly

    ONE clique   = every unsatisfied drone inside the station's radio disk
    + singletons = everybody still en route, plus everybody already satisfied

which is precisely the structure this sub-task needs. Contention for the C ports
happens inside the clique; a drone that has not arrived yet is alone and cannot
pre-negotiate with anyone -- it must fly into range to join the conversation.
Clique membership therefore grows in WAVES as cohorts arrive and shrinks as
drones satisfy and leave, sweeping 0 -> 3 -> 6 -> ... -> 0 inside one episode
with no pairwise bookkeeping anywhere.

The mesh is deliberately WIDER than the dock (comm_radius 1 > dock_radius 0): a
drone on final approach hears the negotiation a step or two before it can act on
it, so `on_charger` and `in_comm` are genuinely different bits rather than a
duplicated flag. Set comm_radius=0 to collapse the mesh onto the dock itself.

Actions (4)
-----------
    0 CONNECT     : request the port (masked unless docked & not charging &
                    not satisfied). Count-based, resolved against pre-DISCONNECT
                    free ports; admit ALL iff requests <= free else brownout &
                    reject ALL (never a subset).
    1 WAIT        : hold (a charging drone keeps charging; anyone else drains).
    2 DISCONNECT  : release a held port (masked unless charging). Vacated port
                    sits empty the rest of this step, claimable only next step.
    3 MOVE        : one step toward the station via a low-level controller
                    (masked unless off-dock).

Observation (own-only, 5 floats)
--------------------------------
    [on_charger, battery, is_charging, in_comm, satisfied]
    on_charger : 1.0 iff physically docked (Manhattan(agent, station) <=
                 dock_radius), i.e. actually arrived.
    battery    : own continuous battery in [0, 1].
    is_charging: 1.0 if currently occupying a port.
    in_comm    : 1.0 if inside the station's mesh with >=1 unsatisfied peer,
                 i.e. own comm bit (see above).
    satisfied  : 1.0 once the drone reached target and LEFT the problem.
    No position, no peer info, no free-port count, no cohort id — coordination
    facts must flow through comm_mask. Deliberately no distance field: MOVE is a
    low-level controller that already walks the drone in, so exposing distance
    would risk a navigation-derived sort key.

State (privileged — centralized critic only)
--------------------------------------------
    [pos_x/norm(N), pos_y/norm(N), battery(N), is_charging(N), satisfied(N),
     dist_to_station/norm(N), free_ports/C, n_in_comm/N, n_satisfied/N,
     t/time_limit, comm_bits(N)]   -> 7N + 4

    The comm bits are the LAST N entries, deliberately. That is the contract
    aerial_ordering.py established and that MAT's mat_comm variants read
    (`state[:, 0, -n_agent:]`), so a MAT run needs no extra plumbing to see the
    communication graph.

Reward (shared scalar)
----------------------
    Step cost   : -0.1 every step.
    Brownout    : -5.0 * max(0, n_req - free)  (reject-all on overload).
    Wasted port : -0.5 * free  when idle unsatisfied DOCKED drones exist but
                  nobody requested (the always-WAIT dodge is a strict loss).
    Satisfied   : +10 per drone reaching target this step.
    Terminal    : +100 when ALL are satisfied; -100 if any drone hits 0
                  battery.
    Approach    : +shaping_weight per cell of progress TOWARD the station (and
                  -shaping_weight per cell away), summed over unsatisfied drones.
                  Potential-based / order-preserving: it pulls drones onto the
                  dock without changing the optimal policy, because the low
                  battery + unsignaled navigation prologue otherwise leaves RL on
                  a flat pre-dock reward plateau it cannot climb. Set
                  shaping_weight=0.0 for the pure (unshaped) reward.

Diagnostics (`info`, cumulative per episode)
--------------------------------------------
Every step returns the running counters (see _reset_diagnostics); a runner reads
them on the terminal step to get episode totals. `docked_wait_steps` vs
`connects` separates a navigation failure from a selection failure, and
`brownouts` vs `wasted_port_steps` separates the two ways of botching the
selection: everyone asking at once, or nobody daring to ask.

Every environment tie (cohort assignment, spawn cell, wind) is broken by uniform
randomness, NEVER by agent index. `all_satisfied` is the success metric.

Paper configuration (mac/presets.py): 6 drones, 2 ports, cohorts (3, 3) at
distances (3, 6), arrival battery U[0.10, 0.15], target 0.80, comm_radius=1,
shaping 0.1, no wind, time_limit=100. Rule-based levels there: an oracle that
rations ports by need +159.96; fly in and never request a port -106.06.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ChargingRendezvousEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # ---- action ids ----
    CONNECT    = 0
    WAIT       = 1
    DISCONNECT = 2
    MOVE       = 3

    # ---- reward constants ----
    REWARD_SUCCESS_ALL = 100.0   # all drones satisfied (terminal)
    REWARD_DRYOUT      = -100.0  # any drone hits 0 battery (terminal)
    REWARD_SATISFY     = 10.0    # per drone reaching target this step
    BROWNOUT_ALPHA     = 5.0     # per excess requester at the overloaded station
    COLLAPSE_BETA      = 0.5     # per idle port while docked demand exists
    REWARD_STEP        = -0.1    # per-step penalty

    def __init__(
        self,
        n_agents=6,
        grid_size=9,
        capacity=2,
        cohort_sizes=(3, 3),
        cohort_distances=(3, 6),
        init_battery=(0.10, 0.15),  # BASE ARRIVAL battery: sampled once per
                                    # episode, applied to every drone, then each
                                    # drone's start is inflated by its trip cost
                                    # so it ARRIVES at this level. Pass a scalar
                                    # to pin it (e.g. for unit tests). This is the
                                    # SLACK/learnable default; [0.06,0.12] is the
                                    # HARD (emergency-swap) regime.
        battery_sigma=0.0,          # per-drone jitter on arrival battery; 0 =>
                                    # exact within-cohort ties (hardest).
        target_level=0.80,
        drain_rate=0.02,
        charge_rate=0.20,
        comm_radius=2,
        dock_radius=0,
        shaping_weight=0.1,         # potential-based reward per cell of progress
                                    # TOWARD the station; pulls drones to the dock
                                    # where the real coordination reward lives.
                                    # Order-preserving (see step()); 0.0 disables.
        wind_prob=0.0,              # prob an idle docked drone is blown off-dock
        time_limit=100,
        seed=None,
        render_flag=False,
        **kwargs,
    ):
        super().__init__()

        self.n_agents    = int(n_agents)
        self.G           = int(grid_size)
        self.capacity    = int(capacity)
        self.cohort_sizes     = [int(s) for s in cohort_sizes]
        self.cohort_distances = [int(d) for d in cohort_distances]

        if len(self.cohort_sizes) != len(self.cohort_distances):
            raise ValueError("cohort_sizes and cohort_distances must have equal length.")
        if sum(self.cohort_sizes) != self.n_agents:
            raise ValueError(
                f"sum(cohort_sizes)={sum(self.cohort_sizes)} must equal n_agents={self.n_agents}."
            )
        if any(s <= self.capacity for s in self.cohort_sizes):
            # Not fatal, but the exact-C stress test only bites when the neediest
            # cohort is larger than the port count. Warn loudly.
            print(
                f"[ChargingRendezvousEnv] WARNING: a cohort of size <= capacity "
                f"({self.capacity}) never forces a contested exact-C selection "
                f"(cohort_sizes={self.cohort_sizes})."
            )

        # Accept scalar (pinned) or (low, high) pair for base arrival battery.
        if np.isscalar(init_battery):
            self.init_battery_low  = float(init_battery)
            self.init_battery_high = float(init_battery)
        else:
            self.init_battery_low, self.init_battery_high = (
                float(init_battery[0]), float(init_battery[1])
            )
        self.battery_sigma = float(battery_sigma)
        self.target_level  = float(target_level)
        self.drain_rate    = float(drain_rate)
        self.charge_rate   = float(charge_rate)
        self.comm_radius   = int(comm_radius)
        self.dock_radius   = int(dock_radius)
        self.shaping_weight = float(shaping_weight)
        self.wind_prob     = float(wind_prob)
        self._time_limit   = int(time_limit)
        self.render_flag   = bool(render_flag)
        self._fig          = None
        self._rng          = np.random.default_rng(seed)

        # --- rendering options (see render()) -------------------------------
        # Read out of **kwargs so an env_args block can set them, and plain
        # attributes so a driver script can flip them on a live env.
        self.render_theme    = kwargs.get("render_theme", "paper")
        self.render_save_dir = kwargs.get("render_save_dir", None)
        self.render_dpi      = int(kwargs.get("render_dpi", 150))
        self.render_pause    = float(kwargs.get("render_pause", 0.35))
        self.render_prefix   = kwargs.get("render_prefix", "charging_rendezvous")
        # Ground: procedural dust/scrub terrain. `render_terrain=False`
        # falls back to the flat background, and the contrast knob dials
        # the texture down if it competes with the foreground in print.
        self.render_terrain = bool(kwargs.get("render_terrain", True))
        self.render_terrain_contrast = float(
            kwargs.get("render_terrain_contrast", 0.72))
        # Ground texture is procedural; the seed picks which patch of field
        # this benchmark sits on, so the figures do not share a texture.
        self.render_terrain_seed = int(kwargs.get("render_terrain_seed", 63))
        self.render_note     = kwargs.get("render_note", "")
        self.render_trail    = int(kwargs.get("render_trail", 6))
        self._frame_paths    = []

        self.station = (self.G // 2, self.G // 2)
        self._dist_norm = max(1, 2 * (self.G - 1))

        # Validate every ring has enough in-bounds cells for its cohort. Cohorts
        # sharing a distance pool their demand.
        demand = {}
        for sz, d in zip(self.cohort_sizes, self.cohort_distances):
            demand[d] = demand.get(d, 0) + sz
        for d, need in demand.items():
            ncells = len(self._ring_cells(d))
            if ncells < need:
                raise ValueError(
                    f"ring at distance {d} has only {ncells} in-bounds cells but "
                    f"{need} drones must spawn there; increase grid_size or spread cohorts."
                )

        # Communication is ONE BIT per agent (station-hub star), so the comm_mask
        # field is width 1. src/run.py and src/envs/gymma.py size the buffer off
        # this attribute -- an env exposing a pairwise [N, N] mask instead would
        # set it to n_agents (see grouped_pickup_consensus.py).
        self.comm_mask_shape = 1

        self.obs_dim   = 5
        self.n_actions = 4
        # state: pos_x, pos_y, battery, is_charging, satisfied, dist_to_station
        # (each N) + free_ports/C + n_in_comm/N + n_satisfied/N + t/limit
        # + comm_bits (N, and they MUST stay last -- see the module docstring)
        self.state_dim = 7 * self.n_agents + 4

        self.action_space = spaces.Tuple(
            tuple(spaces.Discrete(self.n_actions) for _ in range(self.n_agents))
        )
        self.observation_space = spaces.Tuple(
            tuple(
                spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
                for _ in range(self.n_agents)
            )
        )

        # Episode state (allocated in reset)
        self.pos          = None   # int[N, 2]
        self.battery      = None   # float[N]
        self.is_charging  = None   # bool[N]
        self.satisfied    = None   # bool[N]
        self.cohort_of    = None   # int[N]
        self.distance_of  = None   # int[N]
        self._t           = 0

        # Render-only bookkeeping: who asked for a port this step and how the
        # station resolved it. Never read by obs/state/step logic — it exists so
        # a frame can show the contention, not just the aftermath.
        self._last_actions    = None
        self._last_requesters = []
        self._last_connected  = []   # admitted this step
        self._last_satisfied  = []   # reached target this step
        self._last_event      = None
        self._trails          = None
        self._episode_idx     = -1
        self._frame_idx       = 0

        print(
            f"ChargingRendezvousEnv (Charging Rendezvous): N={self.n_agents} "
            f"grid={self.G}x{self.G} station={self.station} capacity={self.capacity} "
            f"cohorts(size@dist)={list(zip(self.cohort_sizes, self.cohort_distances))} "
            f"base_arrival=[{self.init_battery_low},{self.init_battery_high}] "
            f"sigma={self.battery_sigma} target={self.target_level} drain={self.drain_rate} "
            f"charge={self.charge_rate} comm_radius={self.comm_radius} dock_radius={self.dock_radius} "
            f"shaping={self.shaping_weight} wind_p={self.wind_prob} "
            f"obs_dim={self.obs_dim} state_dim={self.state_dim} "
            f"comm=bits(star@station, r={self.comm_radius})"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _ring_cells(self, d):
        """All in-bounds cells at exactly Manhattan distance d from the station."""
        sx, sy = self.station
        cells = []
        for x in range(self.G):
            for y in range(self.G):
                if abs(x - sx) + abs(y - sy) == d:
                    cells.append((x, y))
        return cells

    def _dist_to_station(self, i):
        return self._manhattan(self.pos[i], self.station)

    def _on_charger(self, i):
        return self._dist_to_station(i) <= self.dock_radius

    def _free_ports(self):
        """Free ports at the single station: capacity minus currently-charging."""
        occupied = sum(
            1 for i in range(self.n_agents)
            if self.is_charging[i] and not self.satisfied[i]
        )
        return self.capacity - occupied

    def _comm_bits(self):
        """Per-agent connectivity bits, [N] float, star topology on the station.

        bit[i] = 1 iff i is unsatisfied, sits within comm_radius (Manhattan) of
        the station, and is not the ONLY such drone -- a lone drone in the disk
        has nobody to talk to, so its bit stays 0 and it decodes as a singleton.
        Purely position-derived; see the module docstring for why the hub is the
        station rather than a pairwise range test between drones.
        """
        in_range = [
            (not self.satisfied[i]) and self._dist_to_station(i) <= self.comm_radius
            for i in range(self.n_agents)
        ]
        connected = sum(in_range) >= 2
        return [1.0 if (in_range[i] and connected) else 0.0
                for i in range(self.n_agents)]

    def _move_toward_station(self, i):
        """Low-level controller: one step toward the station. x wins ties.
        Reduces Manhattan distance by exactly 1 unless already docked."""
        x, y = self.pos[i]
        sx, sy = self.station
        dx, dy = sx - x, sy - y
        if dx == 0 and dy == 0:
            return
        if abs(dx) >= abs(dy):
            nx, ny = x + (1 if dx > 0 else -1), y
        else:
            nx, ny = x, y + (1 if dy > 0 else -1)
        if 0 <= nx < self.G and 0 <= ny < self.G:
            self.pos[i] = [nx, ny]

    def _reset_diagnostics(self):
        """Per-episode counters returned in `info` on every step.

        Cumulative over the episode, never per-step: MAT's runner (and PyMARL's
        parallel runner) read `info` only on the terminal step and average each
        scalar across episodes, so a per-step value would be meaningless there.

        The pair that diagnoses a stuck policy is `docked_wait_steps` vs
        `connects`:
          - docked_wait_steps == 0 -> nobody ever reaches the dock; the
            bottleneck is navigation, not port selection.
          - docked_wait_steps > 0 with connects == 0 -> the team arrives but
            never samples a legal request pattern. That is the exact-C joint
            exploration failure this benchmark exists to measure.
        `brownouts` and `wasted_port_steps` then separate the two ways of
        getting it wrong: asking together, or nobody daring to ask.
        """
        self._diag = {
            "brownouts":          0,   # steps where requests > free -> reject ALL
            "brownout_excess":    0,   # sum of (n_req - free) over those steps
            "wasted_port_steps":  0,   # free port + docked demand, nobody asked
            "wasted_ports":       0,   # sum of idle ports over those steps
            "connects":           0,   # drones actually admitted to a port
            "charge_steps":       0,   # drone-steps spent charging
            "docked_wait_steps":  0,   # drone-steps docked, unsatisfied, idle
            "mesh_size_sum":      0,   # sum of clique size, for a mean over steps
            "n_satisfied":        0,   # drones that reached target this episode
            "dryout":             0,   # 1 iff some drone hit 0 battery
            "all_satisfied":      0,   # 1 iff every drone finished
        }

    def _blow_off_dock(self, i):
        """Wind: displace an idle docked drone one cell AWAY from the station
        (into a random in-bounds neighbour that increases its distance)."""
        x, y = self.pos[i]
        d0 = self._dist_to_station(i)
        cands = []
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.G and 0 <= ny < self.G and \
                    self._manhattan((nx, ny), self.station) > d0:
                cands.append([nx, ny])
        if cands:
            self.pos[i] = cands[int(self._rng.integers(len(cands)))]

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t = 0

        # Assign drones to cohorts by a uniform permutation, so agent index
        # carries no structural meaning across episodes.
        perm = self._rng.permutation(self.n_agents)
        self.cohort_of   = np.zeros(self.n_agents, dtype=int)
        self.distance_of = np.zeros(self.n_agents, dtype=int)
        cursor = 0
        for m, (sz, d) in enumerate(zip(self.cohort_sizes, self.cohort_distances)):
            members = perm[cursor:cursor + sz]
            cursor += sz
            for i in members:
                self.cohort_of[i]   = m
                self.distance_of[i] = d

        # Place each drone on a distinct in-bounds cell at its cohort distance.
        occupied = set()
        self.pos = [None] * self.n_agents
        for i in self._rng.permutation(self.n_agents):
            d = int(self.distance_of[i])
            ring = [c for c in self._ring_cells(d) if c not in occupied]
            cell = ring[int(self._rng.integers(len(ring)))]
            occupied.add(cell)
            self.pos[i] = [cell[0], cell[1]]

        # Sample ONE base arrival battery this episode; each drone's start is
        # inflated by its own trip cost so it arrives at that level (nobody dies
        # en route), plus optional per-drone jitter.
        base = float(self._rng.uniform(self.init_battery_low, self.init_battery_high))
        self.battery = np.zeros(self.n_agents, dtype=np.float64)
        for i in range(self.n_agents):
            jitter = (float(self._rng.uniform(-self.battery_sigma, self.battery_sigma))
                      if self.battery_sigma > 0.0 else 0.0)
            self.battery[i] = base + self.distance_of[i] * self.drain_rate + jitter

        self.is_charging = np.zeros(self.n_agents, dtype=bool)
        self.satisfied   = np.zeros(self.n_agents, dtype=bool)

        self._reset_diagnostics()

        self._last_actions    = None
        self._last_requesters = []
        self._last_connected  = []
        self._last_satisfied  = []
        self._last_event      = None
        self._trails          = [[tuple(p)] for p in self.pos]
        self._episode_idx    += 1
        self._frame_idx       = 0

        if self.render_flag:
            self.render()

        return self.get_obs(), dict(self._diag)

    def step(self, actions):
        a = [int(x) for x in actions]
        reward = 0.0
        dryout = False

        self._last_actions    = list(a)   # render-only, see _last_event below
        self._last_requesters = []
        self._last_connected  = []
        self._last_satisfied  = []
        self._last_event      = None

        # Snapshot distances BEFORE movement for navigation shaping (below).
        dist_before = [self._dist_to_station(i) for i in range(self.n_agents)]

        # 1. MOVE — off-dock drones step toward the station (low-level controller).
        for i in range(self.n_agents):
            if self.satisfied[i]:
                continue
            if a[i] == self.MOVE and not self._on_charger(i):
                self._move_toward_station(i)

        # 1b. Navigation shaping (potential-based, order-preserving). Reward each
        #     cell of progress TOWARD the station for unsatisfied drones so the
        #     policy is pulled to the dock — where the coordination reward lives —
        #     instead of sitting on a flat pre-dock plateau. Because it is exactly
        #     the change in a distance potential, a round trip nets zero and a
        #     docked drone (dist 0) earns nothing, so it cannot be farmed and does
        #     not alter the optimal policy.
        if self.shaping_weight > 0.0:
            for i in range(self.n_agents):
                if not self.satisfied[i]:
                    reward += self.shaping_weight * (dist_before[i] - self._dist_to_station(i))

        # Snapshot free ports BEFORE this step's DISCONNECTs (no same-step
        # release-and-claim swap; a vacated port is claimable only next step).
        free_before = self._free_ports()

        # 2. DISCONNECT — charging drones release their port.
        for i in range(self.n_agents):
            if self.satisfied[i]:
                continue
            if a[i] == self.DISCONNECT and self.is_charging[i]:
                self.is_charging[i] = False

        # 3. CONNECT — count-based, resolved against free_before. Only DOCKED,
        #    unsatisfied, not-already-charging requesters count. Admit ALL iff
        #    requests <= free; else BROWNOUT and reject ALL (never a subset).
        docked_idle = [
            i for i in range(self.n_agents)
            if not self.satisfied[i] and not self.is_charging[i] and self._on_charger(i)
        ]
        requesters = [i for i in docked_idle if a[i] == self.CONNECT]
        n_req = len(requesters)
        self._last_requesters = list(requesters)

        self._diag["docked_wait_steps"] += len(docked_idle)

        if n_req == 0:
            if free_before > 0 and len(docked_idle) > 0:
                reward += -self.COLLAPSE_BETA * free_before      # R_collapse (wasted port)
                self._diag["wasted_port_steps"] += 1
                self._diag["wasted_ports"] += free_before
                self._last_event = ("wasted", free_before, len(docked_idle))
        elif n_req <= free_before:
            for i in requesters:
                self.is_charging[i] = True
            self._diag["connects"] += n_req
            self._last_connected = list(requesters)
            self._last_event = ("connect", n_req, free_before)
        else:
            reward += -self.BROWNOUT_ALPHA * (n_req - free_before)  # R_brownout
            self._diag["brownouts"] += 1
            self._diag["brownout_excess"] += n_req - free_before
            self._last_event = ("brownout", n_req, free_before)

        # 4. Battery update + satisfaction / dry-out. Charging drones charge;
        #    everyone else unsatisfied (docked-waiting OR still navigating) drains.
        newly_satisfied = 0
        for i in range(self.n_agents):
            if self.satisfied[i]:
                continue
            if self.is_charging[i]:
                self.battery[i] = min(1.0, self.battery[i] + self.charge_rate)
                self._diag["charge_steps"] += 1
                if self.battery[i] >= self.target_level:
                    self.satisfied[i]   = True
                    self.is_charging[i] = False
                    newly_satisfied += 1
                    self._last_satisfied.append(i)
            else:
                self.battery[i] -= self.drain_rate
                if self.battery[i] <= 0.0:
                    self.battery[i] = 0.0
                    dryout = True
        reward += self.REWARD_SATISFY * newly_satisfied

        # 5. Terminal checks.
        terminated = False
        if dryout:
            reward += self.REWARD_DRYOUT
            terminated = True
        elif np.all(self.satisfied):
            reward += self.REWARD_SUCCESS_ALL
            terminated = True

        # Terminal outcomes outrank a per-step port event in the render badge.
        if dryout:
            self._last_event = ("dryout", 0, 0)
        elif terminated:
            self._last_event = ("all_satisfied", 0, 0)
        elif newly_satisfied and self._last_event is None:
            self._last_event = ("satisfy", newly_satisfied, 0)

        reward += self.REWARD_STEP

        # 6. Wind — blow idle docked drones off the dock (they must MOVE back).
        if self.wind_prob > 0.0 and not terminated:
            for i in range(self.n_agents):
                if self.satisfied[i] or self.is_charging[i]:
                    continue
                if self._on_charger(i) and self._rng.random() < self.wind_prob:
                    self._blow_off_dock(i)

        if self._trails is not None:
            for i in range(self.n_agents):
                self._trails[i].append(tuple(self.pos[i]))
                if len(self._trails[i]) > max(1, self.render_trail):
                    self._trails[i].pop(0)

        self._t += 1
        truncated = self._t >= self._time_limit

        # Episode-level summaries, recomputed each step so the terminal step (the
        # only one a runner reads) always carries the final values.
        self._diag["mesh_size_sum"]  += int(sum(self._comm_bits()))
        self._diag["n_satisfied"]     = int(self.satisfied.sum())
        self._diag["dryout"]          = int(dryout)
        self._diag["all_satisfied"]   = int(bool(np.all(self.satisfied)))

        if self.render_flag:
            import time as _time
            self.render()
            _time.sleep(0.2)

        return self.get_obs(), float(reward), terminated, truncated, dict(self._diag)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def get_obs(self):
        return tuple(self.get_obs_agent(i) for i in range(self.n_agents))

    def get_obs_agent(self, agent_id):
        if self.battery is None:
            return np.zeros(self.obs_dim, dtype=np.float32)
        on_charger  = 1.0 if self._on_charger(agent_id) else 0.0
        battery     = float(np.clip(self.battery[agent_id], 0.0, 1.0))
        is_charging = 1.0 if self.is_charging[agent_id] else 0.0
        connected   = self._comm_bits()[agent_id]
        satisfied   = 1.0 if self.satisfied[agent_id] else 0.0
        return np.array(
            [on_charger, battery, is_charging, connected, satisfied],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # State (privileged — centralized critic only)
    # ------------------------------------------------------------------

    def get_state(self):
        if self.battery is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        comm_bits = np.asarray(self._comm_bits(), dtype=np.float32)
        pos = np.asarray(self.pos, dtype=np.float32)
        dist = np.array([self._dist_to_station(i) for i in range(self.n_agents)],
                        dtype=np.float32)
        state = np.concatenate([
            pos[:, 0] / (self.G - 1),
            pos[:, 1] / (self.G - 1),
            np.clip(self.battery, 0.0, 1.0).astype(np.float32),
            self.is_charging.astype(np.float32),
            self.satisfied.astype(np.float32),
            dist / self._dist_norm,
            np.array([self._free_ports() / self.capacity,
                      comm_bits.sum() / self.n_agents,
                      self.satisfied.mean(),
                      self._t / self._time_limit], dtype=np.float32),
            # LAST N entries: the comm bits. Contract with the mat_comm variants,
            # which slice them straight off the tail. Do not append after this.
            comm_bits,
        ])
        return state.astype(np.float32)

    # ------------------------------------------------------------------
    # Action availability
    # ------------------------------------------------------------------

    def get_avail_actions(self):
        return [self._avail_for(i) for i in range(self.n_agents)]

    def _avail_for(self, i):
        avail = [0] * self.n_actions
        avail[self.WAIT] = 1  # always legal
        if self.satisfied[i]:
            return avail
        if self.is_charging[i]:
            avail[self.DISCONNECT] = 1
        elif self._on_charger(i):
            avail[self.CONNECT] = 1
        else:
            avail[self.MOVE] = 1
        return avail

    # ------------------------------------------------------------------
    # Comm mask (per-agent bits) & env info
    # ------------------------------------------------------------------

    def get_comm_mask(self):
        """Per-agent connectivity bits, shape (N, 1).

        Same representation as aerial_ordering.py. Downstream this becomes
        bits (x) bits: one clique of everyone in the station's mesh, plus a
        singleton for every drone that is out of range or already satisfied.
        """
        return np.array([[b] for b in self._comm_bits()], dtype=np.float32)

    def get_env_info(self):
        return {
            "n_agents":        self.n_agents,
            "n_actions":       self.n_actions,
            "obs_shape":       self.obs_dim,
            "state_shape":     self.state_dim,
            "episode_limit":   self._time_limit,
            "comm_mask_shape": self.comm_mask_shape,
        }

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

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
    def _ring_offset(n, k, radius):
        if n <= 1:
            return 0.0, 0.0
        ang = 2.0 * np.pi * k / n + np.pi / 2.0
        return radius * np.cos(ang), radius * np.sin(ang)

    def render(self):
        """Figure-quality view: the field on the left, telemetry on the right.

        What the frame is built to show, because these are the facts that decide
        the episode and none of them survive a plain positions plot:

          * the STATION with one bay per port, so "2 ports, 3 drones docked" is
            countable rather than asserted;
          * the radio disk as the Manhattan ball the env actually uses, with a
            hub link to every drone currently inside it — that set IS the comm
            clique, and a drone still en route is visibly alone;
          * the SPAWN RINGS each cohort starts on, which is what staggers the
            arrival waves that keep re-posing the exact-C choice;
          * per-drone battery against the target level, in the telemetry column,
            so a drone that is quietly draining toward a dry-out is visible
            before it dies;
          * the step's port resolution as a badge — BROWNOUT (over-asked and
            everyone was rejected), WASTED PORT (nobody dared ask), CONNECTED,
            SATISFIED, DRY-OUT.

        Configure via attributes (also accepted as env_args): render_theme,
        render_save_dir, render_dpi, render_pause, render_prefix, render_note,
        render_trail.
        """
        dr = self._render_kit()

        fig = getattr(self, "_fig", None)
        if fig is None:
            fig, ax, axp, th = dr.stage_split(
                figsize=(11.2, 8.6),
                theme_name=self.render_theme,
                save_dir=self.render_save_dir,
                window_title="Charging Rendezvous",
            )
            self._fig, self._ax, self._axp = fig, ax, axp
        else:
            ax, axp = self._ax, self._axp
            th = dr.reuse(fig, ax, self.render_theme, panel=axp)
        fig.texts.clear()
        fig.patches.clear()

        G = self.G
        dr.field(ax, 0, G - 1, 0, G - 1, th, grid_step=1, major_every=5, pad=1.05,
                 terrain=self.render_terrain,
                 terrain_seed=self.render_terrain_seed,
                 terrain_contrast=self.render_terrain_contrast)

        sx, sy    = self.station
        comm_bits = self._comm_bits()
        free      = self._free_ports()
        mesh      = int(sum(comm_bits))
        n_sat     = int(self.satisfied.sum())
        acts      = self._last_actions

        # --- cohort spawn rings (faint reference only; named in the subtitle)
        cohort_colors = [dr.agent_color(th, 4 * c) for c in range(len(self.cohort_sizes))]
        for c, d in enumerate(self.cohort_distances):
            dr.comm_disk(ax, sx, sy, d, th, color=cohort_colors[c], alpha=0.0,
                         lw=0.9, manhattan=True, zorder=1.0, edge_alpha=0.22)

        # --- radio disk + station ----------------------------------------
        dr.comm_disk(ax, sx, sy, self.comm_radius, th, alpha=0.13, manhattan=True,
                     zorder=1.3)
        port_states = ["busy"] * (self.capacity - free) + ["free"] * free
        dr.station(ax, sx, sy, size=1.05, n_ports=self.capacity,
                   port_states=port_states, th=th, label=None)

        # --- hub links to everyone in the mesh ---------------------------
        for i in range(self.n_agents):
            if comm_bits[i] > 0.5:
                dr.comm_link(ax, (sx, sy), tuple(self.pos[i]), th, alpha=0.7, lw=1.4)

        # --- trails -------------------------------------------------------
        if self._trails is not None and self.render_trail > 1:
            for i in range(self.n_agents):
                if not self.satisfied[i]:
                    dr.trail(ax, self._trails[i],
                             cohort_colors[int(self.cohort_of[i])], th, alpha=0.60)

        # --- drones -------------------------------------------------------
        from collections import defaultdict

        by_cell = defaultdict(list)
        departed = []
        for i in range(self.n_agents):
            if self.satisfied[i]:
                departed.append(i)      # drawn in the "departed" strip instead
            else:
                by_cell[tuple(self.pos[i])].append(i)

        for cell, ids in by_cell.items():
            n_here = len(ids)
            at_station = (cell == (sx, sy))
            if at_station:
                # Docked drones all sit on the station cell. Fan them around the
                # pad — always, even a lone one — so the port bays stay visible
                # and the queue for the two ports stays countable.
                ring, glyph = 1.15, 0.24
            elif n_here == 1:
                ring, glyph = 0.0, 0.30
            else:
                ring, glyph = 0.34, 0.25
            for k, i in enumerate(ids):
                if at_station:
                    ang = 2.0 * np.pi * k / n_here + np.pi / 2.0
                    dx, dy = ring * np.cos(ang), ring * np.sin(ang)
                else:
                    dx, dy = self._ring_offset(n_here, k, ring)
                x, y = cell[0] + dx, cell[1] + dy
                charging  = bool(self.is_charging[i])
                requested = i in self._last_requesters
                rejected  = (requested and not charging
                             and self._last_event is not None
                             and self._last_event[0] == "brownout")
                halo = (th["ok"] if charging else
                        th["bad"] if rejected else
                        th["warn"] if requested else None)
                dr.drone(
                    ax, x, y, size=glyph,
                    color=cohort_colors[int(self.cohort_of[i])], th=th,
                    label=f"D{i + 1}", halo=halo, halo_alpha=0.30, zorder=7,
                )
                # battery as a small pill under the drone: an arc gauge is
                # unreadable at this glyph size, and the pill matches the
                # telemetry column so the two are read the same way.
                dr.battery_bar(ax, x - 0.26, y - glyph * 2.05, 0.52, 0.145,
                               float(self.battery[i]), th,
                               target=self.target_level, zorder=8)
                if charging:
                    dr.bolt(ax, x, y + glyph * 1.55, size=0.22, th=th)
                    # power line into the pad
                    ax.plot([x, sx], [y, sy], color=th["ok"], lw=1.9, alpha=0.85,
                            solid_capstyle="round", zorder=5.4)
                elif requested:
                    ax.text(x, y + glyph * 2.0,
                            "REJECTED" if rejected else "CONNECT",
                            ha="center", va="bottom",
                            color=th["bad"] if rejected else th["warn"],
                            fontsize=6.4, fontweight="bold", zorder=8,
                            path_effects=dr.stroke(th["field_top"], 2.4))
                elif acts is not None and acts[i] == self.MOVE:
                    # heading arrow toward the station: shows the navigation
                    # prologue without needing a caption
                    d = max(1e-6, abs(sx - x) + abs(sy - y))
                    ax.annotate(
                        "", xy=(x + (sx - x) / d * 0.55, y + (sy - y) / d * 0.55),
                        xytext=(x + (sx - x) / d * 0.30, y + (sy - y) / d * 0.30),
                        arrowprops=dict(arrowstyle="-|>", color=th["muted"],
                                        lw=1.2, alpha=0.75), zorder=6,
                    )

        # --- departed (satisfied) strip above the field -------------------
        if departed:
            ax.text(-0.55, G - 1 + 0.72, "DEPARTED", ha="left", va="center",
                    color=th["ok"], fontsize=6.8, fontweight="bold", zorder=6)
            for k, i in enumerate(departed):
                x = 1.15 + 0.95 * k
                dr.drone(ax, x, G - 1 + 0.72, size=0.20,
                         color=cohort_colors[int(self.cohort_of[i])], th=th,
                         label=f"D{i + 1}", dimmed=True, shadow=False, zorder=6)
                ax.text(x, G - 1 + 0.36, "✓", ha="center", va="center",
                        color=th["ok"], fontsize=8, fontweight="bold", zorder=6)

        dr.compass(ax, -0.30, G - 2.2, th)

        # --- telemetry column ---------------------------------------------
        dr.panel(axp, th, title=f"drone telemetry · target {self.target_level:.0%}")
        # port bays
        axp.text(0.055, 0.918, "STATION PORTS", ha="left", va="center",
                 color=th["muted"], fontsize=6.8, fontweight="bold")
        charging_ids = [i for i in range(self.n_agents) if self.is_charging[i]]
        for p in range(self.capacity):
            px = 0.055 + p * 0.47
            occupied = p < len(charging_ids)
            from matplotlib.patches import FancyBboxPatch

            axp.add_patch(FancyBboxPatch(
                (px, 0.845), 0.42, 0.055,
                boxstyle="round,pad=0.0,rounding_size=0.012",
                facecolor=th["ok"] if occupied else th["bg"],
                edgecolor=th["panel_edge"], lw=1.0, alpha=0.9, zorder=2,
            ))
            axp.text(px + 0.21, 0.872,
                     f"⚡ D{charging_ids[p] + 1}" if occupied else "free",
                     ha="center", va="center",
                     color="#FFFFFF" if occupied else th["muted"],
                     fontsize=7.6, fontweight="bold", zorder=3)

        row_y = 0.775
        for i in range(self.n_agents):
            if self.satisfied[i]:
                status, scolor, note = "SATISFIED", th["ok"], "left the mesh"
            elif self.is_charging[i]:
                status, scolor, note = "CHARGING", th["ok"], "holds a port"
            elif self._on_charger(i):
                if i in self._last_requesters:
                    status, scolor = ("REJECTED", th["bad"]) if (
                        self._last_event and self._last_event[0] == "brownout"
                    ) else ("CONNECTING", th["warn"])
                    note = "asked for a port"
                else:
                    status, scolor, note = "WAITING", th["warn"], "docked, deferring"
            else:
                status, scolor = "EN ROUTE", th["muted"]
                note = f"{self._dist_to_station(i)} cells out · cohort {self.cohort_of[i]}"
            dr.panel_row(
                axp, row_y, f"D{i + 1}",
                cohort_colors[int(self.cohort_of[i])],
                float(np.clip(self.battery[i], 0.0, 1.0)), status, scolor, th,
                target=self.target_level, note=note,
                dimmed=bool(self.satisfied[i]),
            )
            row_y -= 0.083

        axp.text(0.055, 0.085, "battery pill: fill = charge, tick = target level",
                 ha="left", va="center", color=th["muted"], fontsize=6.4)
        axp.text(0.055, 0.050,
                 f"drain {self.drain_rate:.2f}/step · charge {self.charge_rate:.2f}/step",
                 ha="left", va="center", color=th["muted"], fontsize=6.4)
        axp.text(0.055, 0.015,
                 "identical batteries ⇒ no observable sort key",
                 ha="left", va="center", color=th["comm"], fontsize=6.4,
                 fontweight="bold")

        # --- HUD ------------------------------------------------------------
        dr.header(
            fig, ax, "Charging Rendezvous",
            f"{self.n_agents} interchangeable drones · one station, {self.capacity} "
            f"ports · cohorts {list(self.cohort_sizes)} at rings "
            f"{list(self.cohort_distances)} · exactly {self.capacity} may connect "
            f"per wave",
            th,
        )
        docked_idle = sum(
            1 for i in range(self.n_agents)
            if not self.satisfied[i] and not self.is_charging[i] and self._on_charger(i)
        )
        low = float(np.min([self.battery[i] for i in range(self.n_agents)
                            if not self.satisfied[i]], initial=1.0))
        dr.stat_row(fig, [
            ("step", f"{self._t}/{self._time_limit}"),
            ("ports free", f"{free}/{self.capacity}",
             th["bad"] if (free > 0 and docked_idle > 0) else None),
            ("in mesh", f"{mesh}/{self.n_agents}", th["comm"] if mesh else None),
            ("docked waiting", str(docked_idle),
             th["warn"] if docked_idle > free else None),
            ("lowest battery", f"{low * 100:.0f}%", dr.battery_color(th, low)),
            ("satisfied", f"{n_sat}/{self.n_agents}",
             th["ok"] if n_sat else None),
        ], th)

        ev = self._last_event
        if ev is not None:
            kind, n1, n2 = ev
            if kind == "brownout":
                dr.status_pill(fig, f"BROWNOUT   {n1} asked / {n2} free — all rejected",
                               th["bad"], th)
            elif kind == "wasted":
                dr.status_pill(fig, f"WASTED PORT   {n1} idle, {n2} docked, nobody asked",
                               th["warn"], th)
            elif kind == "connect":
                dr.status_pill(fig, f"CONNECTED   {n1} admitted / {n2} free",
                               th["ok"], th)
            elif kind == "satisfy":
                dr.status_pill(fig, f"SATISFIED   +{n1} drone(s) at target",
                               th["ok"], th)
            elif kind == "dryout":
                dr.status_pill(fig, "DRY-OUT — a drone hit 0%", th["bad"], th)
            elif kind == "all_satisfied":
                dr.status_pill(fig, "ALL DRONES SATISFIED", th["ok"], th)

        dr.legend_row(fig, [
            (th["comm"], "station radio mesh"),
            (th["ok"], "charging on a port"),
            (th["warn"], "requesting a port"),
            (th["bad"], "rejected by brownout"),
        ], th)
        dr.timeline(fig, self._t, self._time_limit, th)
        dr.footer(fig, "Charging Rendezvous · MAC benchmark", th,
                  right=self.render_note)

        path = dr.finish(
            fig,
            save_dir=self.render_save_dir,
            name=f"{self.render_prefix}_ep{max(0, self._episode_idx):02d}_t{self._frame_idx:03d}",
            dpi=self.render_dpi,
            pause=self.render_pause,
        )
        self._frame_idx += 1
        if path:
            self._frame_paths.append(path)
        return path

    def render_text(self):
        """The original console render, kept for headless logging."""
        comm_bits = self._comm_bits()
        lines = [
            f"t={self._t}  satisfied={int(self.satisfied.sum())}/{self.n_agents}  "
            f"free_ports={self._free_ports()}/{self.capacity}  "
            f"mesh={int(sum(comm_bits))}/{self.n_agents}"
        ]
        for i in range(self.n_agents):
            st = "S" if self.satisfied[i] else ("C" if self.is_charging[i] else
                                                ("@" if self._on_charger(i) else "·"))
            comm = "" if comm_bits[i] > 0.5 else "x"
            lines.append(
                f"  d{i}[c{self.cohort_of[i]}] pos={tuple(self.pos[i])} "
                f"dist={self._dist_to_station(i)} batt={self.battery[i]:.2f} {st}{comm}"
            )
        print("\n".join(lines))

    def close(self):
        fig = getattr(self, "_fig", None)
        if fig is not None:
            import matplotlib.pyplot as plt

            plt.close(fig)
            self._fig = None
        return None


# ======================================================================
# __main__ verification block
# ======================================================================
if __name__ == "__main__":
    import sys

    PASS = "\033[92mPASS\033[0m"
    FAIL = "\033[91mFAIL\033[0m"
    errors = []

    def check(cond, msg):
        if cond:
            print(f"  {PASS}: {msg}")
        else:
            print(f"  {FAIL}: {msg}")
            errors.append(msg)

    # ------------------------------------------------------------------
    # 1. Structure & shapes
    # ------------------------------------------------------------------
    print("\n=== 1. Structure & shapes ===")
    e = ChargingRendezvousEnv(seed=0)
    e.reset(seed=0)
    obs = e.get_obs()
    check(e.obs_dim == 5, "obs_dim == 5")
    check(e.n_actions == 4, "n_actions == 4")
    check(all(o.shape == (5,) for o in obs), "all obs shape (5,)")
    check(e.get_state().shape == (e.state_dim,), f"state shape ({e.state_dim},)")
    check(e.state_dim == 7 * e.n_agents + 4, "state_dim == 7N + 4")
    cm = e.get_comm_mask()
    check(cm.shape == (e.n_agents, 1), "comm_mask shape (N, 1) — per-agent bits")
    check(e.get_env_info()["comm_mask_shape"] == 1, "env_info comm_mask_shape == 1")
    check(set(np.unique(cm)) <= {0.0, 1.0}, "comm_mask is binary")
    # Spawn rings are at distance 3 and 6, both outside comm_radius=2.
    check(float(cm.sum()) == 0.0, "at reset everyone is out of mesh (all singletons)")

    # ------------------------------------------------------------------
    # 2. Per-agent battery funds the trip -> everyone ARRIVES tied
    # ------------------------------------------------------------------
    print("\n=== 2. init_battery_i = base + distance_i * drain (funds the trip) ===")
    ep = ChargingRendezvousEnv(init_battery=0.06, battery_sigma=0.0, seed=1)
    ep.reset(seed=1)
    ok_formula = all(
        abs(ep.battery[i] - (0.06 + ep.distance_of[i] * ep.drain_rate)) < 1e-9
        for i in range(ep.n_agents)
    )
    check(ok_formula, "each drone's start = base + distance*drain")
    # farther cohort starts with strictly more charge
    near = [i for i in range(ep.n_agents) if ep.cohort_of[i] == 0]
    far  = [i for i in range(ep.n_agents) if ep.cohort_of[i] == 1]
    check(ep.battery[far[0]] > ep.battery[near[0]], "farther cohort dispatched with more battery")

    # ------------------------------------------------------------------
    # 3. Staggered arrival: near cohort docks before far cohort; nobody dies
    # ------------------------------------------------------------------
    print("\n=== 3. Staggered arrival timing (all-MOVE) ===")
    e3 = ChargingRendezvousEnv(init_battery=0.06, cohort_sizes=(3, 3),
                                   cohort_distances=(3, 6), seed=2)
    e3.reset(seed=2)
    near = [i for i in range(6) if e3.cohort_of[i] == 0]  # distance 3
    far  = [i for i in range(6) if e3.cohort_of[i] == 1]  # distance 6
    dock_step   = {i: None for i in range(6)}
    arrival_bat = {i: None for i in range(6)}
    for t in range(1, 7):
        e3.step([e3.MOVE] * 6)
        for i in range(6):
            if dock_step[i] is None and e3._on_charger(i):
                dock_step[i]   = t
                arrival_bat[i] = float(e3.battery[i])  # battery the moment it docks
    check(all(dock_step[i] == 3 for i in near), "near cohort (dist 3) all dock at step 3")
    check(all(dock_step[i] == 6 for i in far), "far cohort (dist 6) all dock at step 6")
    check(all(arrival_bat[i] > 0.0 for i in range(6)),
          "nobody dries out en route (battery funds the trip)")
    check(all(abs(arrival_bat[i] - 0.06) < 1e-6 for i in range(6)),
          "every cohort ARRIVES at the same base battery (0.06) despite different distances")

    # ------------------------------------------------------------------
    # 4. Exact within-cohort tie at the dock (sigma=0) & obs symmetry
    # ------------------------------------------------------------------
    print("\n=== 4. Exact within-cohort tie at the dock ===")
    e4 = ChargingRendezvousEnv(init_battery=0.06, battery_sigma=0.0,
                                   cohort_sizes=(3, 3), cohort_distances=(3, 6), seed=3)
    e4.reset(seed=3)
    near = [i for i in range(6) if e4.cohort_of[i] == 0]
    for _ in range(3):
        e4.step([e4.MOVE] * 6)  # near cohort now docked
    check(all(e4._on_charger(i) for i in near), "near cohort docked after 3 MOVE steps")
    check(np.allclose(e4.battery[near], e4.battery[near[0]]),
          "co-cohort docked drones exactly tied in battery")
    check(np.allclose(e4.get_obs_agent(near[0]), e4.get_obs_agent(near[1])),
          "co-cohort docked drones have identical observations (no within-cohort sort key)")
    # 3 docked tied drones, capacity 2 -> the exact-C contested selection
    check(len(near) > e4.capacity, "neediest cohort larger than port count (contested exact-C)")

    # jitter breaks the exact tie into near-ties
    e4j = ChargingRendezvousEnv(init_battery=0.06, battery_sigma=0.02,
                                    cohort_sizes=(3, 3), cohort_distances=(3, 6), seed=3)
    e4j.reset(seed=3)
    nearj = [i for i in range(6) if e4j.cohort_of[i] == 0]
    check(not np.allclose(e4j.battery[nearj], e4j.battery[nearj[0]]),
          "battery_sigma>0 turns the exact tie into near-ties (ablation knob)")

    # ------------------------------------------------------------------
    # 4b. Navigation shaping rewards progress toward the dock
    # ------------------------------------------------------------------
    print("\n=== 4b. Navigation shaping (potential-based) ===")
    es = ChargingRendezvousEnv(init_battery=0.10, shaping_weight=0.1,
                                   cohort_sizes=(3, 3), cohort_distances=(3, 6), seed=8)
    es.reset(seed=8)
    off = [i for i in range(6) if not es._on_charger(i)][0]
    d0 = es._dist_to_station(off)
    _, r_move, _, _, _ = es.step([es.MOVE] * 6)   # everyone steps one cell in
    # each of the 6 unsatisfied drones moved 1 cell closer => +0.1 each = +0.6,
    # net of the -0.1 step cost => +0.5 on this step
    check(r_move > 0.0, f"all-MOVE toward station yields POSITIVE shaped reward ({r_move:.2f})")
    check(es._dist_to_station(off) == d0 - 1, "MOVE reduced distance by one cell")
    es0 = ChargingRendezvousEnv(init_battery=0.10, shaping_weight=0.0, seed=8)
    es0.reset(seed=8)
    _, r_move0, _, _, _ = es0.step([es0.MOVE] * 6)
    check(r_move0 < 0.0, f"with shaping_weight=0 the same step is a net loss ({r_move0:.2f}) "
                         f"(unshaped plateau)")
    # waiting off-dock earns no shaping (no progress), so MOVE is strictly preferred
    es.reset(seed=8)
    _, r_wait, _, _, _ = es.step([es.WAIT] * 6)
    check(r_wait < r_move, "WAIT (no progress) is strictly worse than MOVE under shaping")

    # ------------------------------------------------------------------
    # 5. Comm bits: ONE clique (the station mesh) + singletons
    # ------------------------------------------------------------------
    print("\n=== 5. Comm structure: one station-hub clique + singletons ===")
    # e4 is 3 MOVE steps in: near cohort docked (dist 0), far cohort at dist 3.
    far4 = [i for i in range(6) if e4.cohort_of[i] == 1]
    bits = e4.get_comm_mask()[:, 0]
    check(all(bits[i] == 1.0 for i in near), "docked cohort is in the mesh (bit 1)")
    check(all(bits[i] == 0.0 for i in far4),
          "cohort still en route (dist 3 > comm_radius 2) is isolated (bit 0)")
    check(float(bits.sum()) == len(near),
          f"exactly one group of {len(near)}, everyone else a singleton")

    # The state's LAST N entries must BE the comm bits (MAT contract).
    st = e4.get_state()
    check(np.allclose(st[-e4.n_agents:], bits), "state tail == comm bits")
    check(abs(float(st[-e4.n_agents - 3]) - bits.sum() / e4.n_agents) < 1e-6,
          "state n_in_comm field == mesh size / N")

    # Membership rules, checked directly.
    e5 = ChargingRendezvousEnv(init_battery=0.10, seed=3)
    e5.reset(seed=3)
    for i in range(6):
        e5.pos[i] = [4, 4]                       # everyone co-located at the station
    check(all(b == 1.0 for b in e5._comm_bits()), "co-located drones form one 6-clique")
    e5.satisfied[0] = True
    bits5 = e5._comm_bits()
    check(bits5[0] == 0.0, "a SATISFIED drone drops out of the mesh")
    check(all(bits5[i] == 1.0 for i in range(1, 6)), "the remaining 5 stay in the clique")
    e5.satisfied[0] = False
    for i in range(1, 6):
        e5.pos[i] = [0, 0]                       # far outside comm_radius
    check(all(b == 0.0 for b in e5._comm_bits()),
          "a LONE drone in range has no peer -> bit 0, decodes as a singleton")

    # Mesh is strictly wider than the dock: an approaching drone hears first.
    e5.pos[0] = [4, 4]
    e5.pos[1] = [4, 6]                           # dist 2: in mesh, NOT docked
    bits5 = e5._comm_bits()
    check(bits5[1] == 1.0 and not e5._on_charger(1),
          "a drone on final approach (dist 2) is in the mesh but not docked")
    check(e5.get_obs_agent(1)[3] == 1.0 and e5.get_obs_agent(1)[0] == 0.0,
          "obs reflects it: in_comm=1 while on_charger=0")

    # ------------------------------------------------------------------
    # 6. Brownout rejects ALL (order independent)
    # ------------------------------------------------------------------
    print("\n=== 6. Brownout resolution (reject-all, order-independent) ===")
    def dock_cohort(env, members):
        for _ in range(max(env.distance_of[i] for i in members)):
            env.step([env.MOVE] * env.n_agents)
    e6 = ChargingRendezvousEnv(init_battery=0.10, cohort_sizes=(3, 3),
                                   cohort_distances=(3, 6), seed=4)
    e6.reset(seed=4)
    near = [i for i in range(6) if e6.cohort_of[i] == 0]
    dock_cohort(e6, near)
    acts = [e6.WAIT] * 6
    for i in near:
        acts[i] = e6.CONNECT  # 3 docked tied drones request, 2 ports
    e6.step(acts)
    check(sum(int(e6.is_charging[i]) for i in near) == 0,
          "overload (3 req, 2 ports) -> NOBODY charges (reject-all)")

    # ------------------------------------------------------------------
    # 7. No same-step release-and-claim swap
    # ------------------------------------------------------------------
    print("\n=== 7. DISCONNECT + CONNECT same step -> no swap ===")
    e7 = ChargingRendezvousEnv(init_battery=0.10, cohort_sizes=(3, 3),
                                   cohort_distances=(3, 6), seed=5)
    e7.reset(seed=5)
    near = [i for i in range(6) if e7.cohort_of[i] == 0]
    dock_cohort(e7, near)
    acts = [e7.WAIT] * 6
    acts[near[0]] = e7.CONNECT
    acts[near[1]] = e7.CONNECT  # 2 charge (exact fill)
    e7.step(acts)
    check(e7.is_charging[near[0]] and e7.is_charging[near[1]],
          "exact-fill (2 req, 2 ports) -> both admitted")
    charger, waiter = near[0], near[2]
    acts = [e7.WAIT] * 6
    acts[charger] = e7.DISCONNECT
    acts[waiter]  = e7.CONNECT   # tries to claim the just-vacated port, same step
    e7.step(acts)
    check(not e7.is_charging[waiter],
          "waiter CANNOT claim a port vacated by DISCONNECT in the same step")
    acts = [e7.WAIT] * 6
    acts[waiter] = e7.CONNECT
    e7.step(acts)
    check(bool(e7.is_charging[waiter]), "the vacated port becomes claimable the NEXT step")

    # ------------------------------------------------------------------
    # 8. Satisfaction frees a port
    # ------------------------------------------------------------------
    print("\n=== 8. Satisfaction leaves & frees a port ===")
    e8 = ChargingRendezvousEnv(init_battery=0.75, target_level=0.80,
                                   charge_rate=0.10, cohort_sizes=(3, 3),
                                   cohort_distances=(3, 6), seed=6)
    e8.reset(seed=6)
    near = [i for i in range(6) if e8.cohort_of[i] == 0]
    dock_cohort(e8, near)  # arrive at ~0.75 - drained a touch; top of range
    acts = [e8.WAIT] * 6
    acts[near[0]] = e8.CONNECT
    acts[near[1]] = e8.CONNECT
    # charge them across the target
    for _ in range(3):
        e8.step(acts)
        acts = [e8.WAIT] * 6
        acts[near[0]] = e8.WAIT
        acts[near[1]] = e8.WAIT
    check(bool(e8.satisfied[near[0]]) and bool(e8.satisfied[near[1]]),
          "drones crossing target become satisfied")
    check(e8._free_ports() == 2, "ports freed after satisfied drones leave")

    # ------------------------------------------------------------------
    # 8b. Per-episode diagnostics in info
    # ------------------------------------------------------------------
    print("\n=== 8b. Episode diagnostics (info) ===")
    ed = ChargingRendezvousEnv(init_battery=0.10, cohort_sizes=(3, 3),
                                   cohort_distances=(3, 6), seed=9)
    _, info0 = ed.reset(seed=9)
    check(info0["brownouts"] == 0 and info0["connects"] == 0,
          "reset zeroes the counters")
    near = [i for i in range(6) if ed.cohort_of[i] == 0]
    _, _, _, _, info = ed.step([ed.MOVE] * 6)
    check(info["docked_wait_steps"] == 0,
          "no docked_wait_steps while everyone is still en route")
    for _ in range(2):
        _, _, _, _, info = ed.step([ed.MOVE] * 6)  # step 3 lands the near cohort
    # MOVE resolves before the CONNECT phase, so the arrival step itself already
    # registers 3 docked idle drones -- and charges the wasted-port penalty, even
    # though CONNECT was masked for them at decision time (they were off-dock).
    check(info["docked_wait_steps"] == 3, "arrival step counts 3 docked idle drones")
    check(info["wasted_port_steps"] == 1 and info["wasted_ports"] == 2,
          "arrival step wastes both ports (CONNECT was still masked)")
    acts = [ed.WAIT] * 6
    for i in near:
        acts[i] = ed.CONNECT                       # 3 requests, 2 ports
    _, _, _, _, info = ed.step(acts)
    check(info["brownouts"] == 1 and info["brownout_excess"] == 1,
          "brownout counted with excess 1 (3 req, 2 ports)")
    check(info["connects"] == 0, "a rejected wave admits nobody")
    check(info["docked_wait_steps"] == 6, "docked_wait_steps accumulates (3 + 3)")
    _, _, _, _, info = ed.step([ed.WAIT] * 6)      # free ports, nobody asks
    check(info["wasted_port_steps"] == 2 and info["wasted_ports"] == 4,
          "a second wasted-port step counted, both ports idle again")
    acts = [ed.WAIT] * 6
    acts[near[0]] = ed.CONNECT
    acts[near[1]] = ed.CONNECT
    _, _, _, _, info = ed.step(acts)
    check(info["connects"] == 2, "exact-fill counted as 2 connects")
    check(info["charge_steps"] == 2, "2 drone-steps of charging")
    check(info["brownouts"] == 1, "counters are cumulative, not per-step")
    check(info["mesh_size_sum"] > 0, "mesh size accumulated for a per-step mean")

    # ------------------------------------------------------------------
    # 9. Calibration baselines: oracle > backoff > degenerate
    # ------------------------------------------------------------------
    print("\n=== 9. Calibration baselines ===")

    def run_policy(policy, n_ep=40, seed0=100):
        rets, done, dry = [], 0, 0
        for ep in range(n_ep):
            env = ChargingRendezvousEnv(seed=seed0 + ep)
            env.reset(seed=seed0 + ep)
            R, term, trunc = 0.0, False, False
            while not (term or trunc):
                _, r, term, trunc, _ = env.step(policy(env))
                R += r
            rets.append(R)
            if term and np.all(env.satisfied):
                done += 1
            elif term:
                dry += 1
        return float(np.mean(rets)), done / n_ep, dry / n_ep

    def oracle(env, margin_steps=3):
        acts = [env.WAIT] * env.n_agents
        danger = env.drain_rate * margin_steps
        # navigate everyone in first
        for i in range(env.n_agents):
            if not env.satisfied[i] and not env.is_charging[i] and not env._on_charger(i):
                acts[i] = env.MOVE
        free = env._free_ports()
        docked_idle = sorted(
            [i for i in range(env.n_agents)
             if not env.satisfied[i] and not env.is_charging[i] and env._on_charger(i)],
            key=lambda i: env.battery[i],
        )
        for i in docked_idle[:max(0, free)]:
            acts[i] = env.CONNECT
        # preemptive swap: disconnect a high-battery charger for a critical waiter
        remaining = [i for i in docked_idle[max(0, free):] if env.battery[i] <= danger]
        chargers_desc = sorted(
            [i for i in range(env.n_agents) if env.is_charging[i]],
            key=lambda i: env.battery[i], reverse=True,
        )
        for i in remaining:
            if chargers_desc:
                c = chargers_desc.pop(0)
                if acts[c] == env.WAIT:
                    acts[c] = env.DISCONNECT
        return acts

    def backoff(env):
        acts = [env.WAIT] * env.n_agents
        for i in range(env.n_agents):
            if env.satisfied[i] or env.is_charging[i]:
                continue
            if not env._on_charger(i):
                acts[i] = env.MOVE
            elif env._rng.random() < 0.66:
                acts[i] = env.CONNECT
        return acts

    def degenerate(env):
        return [env.WAIT] * env.n_agents

    r_o, d_o, dry_o = run_policy(oracle)
    r_b, d_b, dry_b = run_policy(backoff)
    r_d, d_d, dry_d = run_policy(degenerate)
    print(f"  oracle     : return={r_o:8.2f}  complete={d_o:.0%}  dryout={dry_o:.0%}")
    print(f"  backoff    : return={r_b:8.2f}  complete={d_b:.0%}  dryout={dry_b:.0%}")
    print(f"  degenerate : return={r_d:8.2f}  complete={d_d:.0%}  dryout={dry_d:.0%}")
    check(r_o > r_b and r_o > r_d, "oracle return strictly highest")
    check(d_o > 0.9, "oracle completes almost every episode (env is feasible)")
    check(dry_d > 0.9, "always-WAIT never even arrives -> dries out almost every episode")
    check(d_b < d_o, "randomized backoff completes strictly less than oracle "
                     "(coordination is required, not luck)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED ({len(errors)} errors):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED ✔")
    print("=" * 60)
