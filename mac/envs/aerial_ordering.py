"""Aerial Ordering: find a target, learn its hidden K, lift it with exactly K.

One sensor drone (agent 0) and `n_workers` worker drones (agents 1..n_workers)
fly on a `grid_size` x `grid_size` grid. Each target has a hidden requirement
K ~ U[k_min, k_max]: exactly K workers must PICKUP in the same step. Only the
sensor can read K, and only while standing on the target (sense_radius=0); a
worker sees K only while it is within `comm_radius` (Manhattan) of the sensor
and the sensor is reading it. Wind drifts each worker one random cell with
probability `wind_prob` per step, so the team has to re-assemble as it goes.
Pickup Consensus is this task with the grid, sensor, comm gate and wind removed.

Observation, obs_mode="simple" (5 floats per agent)
    [on_target, K/k_max or -1 if unknown, is_loaded (-1 for the sensor),
     free workers connected to the sensor / n_workers (0 if not connected),
     fraction of targets remaining]
State (centralised critic)
    every agent's position, target position, K/k_max, connected workers,
    target_lifted, targets remaining, per-agent comm bits.
Actions, simple mode
    0 MOVE   one step toward the active target (the path is not learned)
    1 PICKUP masked unless the agent is a free worker that currently knows K
    2 WAIT
    (obs_mode="extended" uses positions and directional moves instead.)
Reward (shared)
    every step               -0.3 far from the target, down to -0.1 on it
    exactly K lift together  +10, and +10 more once every target is lifted
    a failed lift            -20 and the episode ends: premature (fewer free
                             workers assembled than K), overshoot, undershoot,
                             or a picker not on the target
Info (per-episode counters)
    targets_lifted (the success rate when n_targets=1), correct_pickups,
    overshoot_pickups, undershoot_pickups, premature_pickups, offtarget_pickups,
    assembled_steps, pickup_steps.

Paper configuration (mac/presets.py): 5 workers, 1 target, K in {2,3,4}, 10x10
grid, comm_radius=0, wind_prob=0.1, time_limit=100. Rule-based levels there:
fly in and lift with exactly K +18.78 (solve rate 1.0); fly in and wait on the
target forever -10.36.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AerialOrderingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # ---- simple mode action ids ----
    MOVE   = 0
    PICKUP = 1
    WAIT   = 2

    # ---- extended mode action ids ----
    STAY   = 0
    NORTH  = 1
    SOUTH  = 2
    EAST   = 3
    WEST   = 4
    HANDLE = 5

    _DELTAS = {1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)}

    # Reward constants (shared across modes)
    REWARD_CORRECT_PICKUP     =  10.0
    REWARD_OVERSHOOT          = -20.0
    REWARD_UNDERSHOOT         = -20.0
    REWARD_PREMATURE          = -20.0
    # Time cost scales with how far the team is from the active target: FAR at
    # maximum distance, NEAR while standing on it. There are deliberately no wait
    # bonuses — this single term supplies BOTH time pressure (it is always
    # negative, so idling anywhere is never free) and navigation pressure (the
    # bleed slows as the team closes in). That ordering is what makes
    # "solve" > "wait on the target" > "roam around".
    REWARD_STEP_FAR           =  -0.3
    REWARD_STEP_NEAR          =  -0.1
    REWARD_EPISODE_COMPLETION =  10.0

    def __init__(
        self,
        obs_mode="simple",
        grid_size=8,
        n_workers=3,
        n_targets=2,
        k_min=1,
        k_max=2,
        comm_radius=1,
        sense_radius=0,
        wind_prob=0.1,
        time_limit=100,
        seed=None,
        render_flag=False,
        **kwargs,
    ):
        super().__init__()

        if obs_mode not in ("simple", "extended"):
            raise ValueError(f"obs_mode must be 'simple' or 'extended', got {obs_mode!r}")

        self.obs_mode     = obs_mode
        self.G            = int(grid_size)
        self.n_targets    = int(n_targets)
        self.k_min        = int(k_min)
        self.k_max        = int(k_max)
        self.comm_radius  = int(comm_radius)
        self.sense_radius = int(sense_radius)
        self.wind_prob    = float(wind_prob)
        self._time_limit  = int(time_limit)
        self.render_flag  = bool(render_flag)
        self._fig         = None
        self._rng         = np.random.default_rng(seed)

        # --- rendering options (see render()) -------------------------------
        # Read out of **kwargs so they can be set from an env_args block, and
        # they are plain attributes so a driver script (scripts/render_policy.py)
        # can flip them on an already-constructed env.
        self.render_theme    = kwargs.get("render_theme", "paper")
        self.render_save_dir = kwargs.get("render_save_dir", None)
        self.render_dpi      = int(kwargs.get("render_dpi", 150))
        self.render_pause    = float(kwargs.get("render_pause", 0.35))
        self.render_prefix   = kwargs.get("render_prefix", "aerial_ordering")
        # Ground: procedural dust/scrub terrain. `render_terrain=False`
        # falls back to the flat background, and the contrast knob dials
        # the texture down if it competes with the foreground in print.
        self.render_terrain = bool(kwargs.get("render_terrain", True))
        self.render_terrain_contrast = float(
            kwargs.get("render_terrain_contrast", 0.72))
        # Ground texture is procedural; the seed picks which patch of field
        # this benchmark sits on, so the figures do not share a texture.
        self.render_terrain_seed = int(kwargs.get("render_terrain_seed", 7))
        self.render_note     = kwargs.get("render_note", "")
        self.render_trail    = int(kwargs.get("render_trail", 6))
        self._frame_paths    = []

        self.n_sensor  = 1
        self.n_workers = int(n_workers)
        self.n_agents  = self.n_sensor + self.n_workers  # agent 0 = sensor, rest = workers

        # Every target needs at least k_min workers, and a worker is spent after one
        # lift, so the whole episode is only solvable when there are enough to go
        # round. Without this the K sampler silently emits sum(K) > n_workers and
        # later targets become impossible.
        if self.n_targets * self.k_min > self.n_workers:
            raise ValueError(
                f"infeasible: n_targets({self.n_targets}) * k_min({self.k_min}) = "
                f"{self.n_targets * self.k_min} > n_workers({self.n_workers}). "
                f"Lower n_targets/k_min or raise n_workers."
            )
        if self.n_agents > self.G * self.G or self.n_targets > self.G * self.G:
            raise ValueError(
                f"grid {self.G}x{self.G} is too small to place {self.n_agents} agents "
                f"and {self.n_targets} targets at distinct cells."
            )

        if obs_mode == "simple":
            self.obs_dim   = 5
            self.n_actions = 3
        else:
            # sensor: 8 real fields + 2 zero padding; workers: 10 fields
            self.obs_dim   = 10
            self.n_actions = 6

        # state_dim: 2*4 positions + 2 tgt xy + K + N_comm + lifted + remaining + 4 comm bits
        self.state_dim = 2 * self.n_agents + 2 + 1 + 1 + 1 + 1 + self.n_agents  # = 18

        self.action_space = spaces.Tuple(
            tuple(spaces.Discrete(self.n_actions) for _ in range(self.n_agents))
        )
        self.observation_space = spaces.Tuple(
            tuple(
                spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
                for _ in range(self.n_agents)
            )
        )

        # Episode state
        self.pos                = None   # list of [x, y] per agent
        self.target_pos         = None   # list of [x, y] per target
        self.K                  = None   # list of int, hidden per target
        self.loaded             = None   # [bool]*n_workers — True once worker participated in lift
        self.targets_lifted     = 0
        self.current_target_idx = 0
        self._t                 = 0

        # Render-only bookkeeping. Never read by get_obs/get_state/step logic —
        # it exists so a frame can show WHAT JUST HAPPENED (who committed, and
        # how the attempt resolved), which a positions-only picture cannot.
        self._last_actions = None
        self._last_pickers = []     # workers that committed PICKUP last step
        self._last_event   = None   # ("correct"|"overshoot"|"undershoot"|
                                    #  "offtarget"|"premature", n_pickup, K)
        self._trails       = None
        self._episode_idx  = -1
        self._frame_idx    = 0

        print(
            f"AerialOrderingEnv (Aerial Ordering): "
            f"obs_mode={self.obs_mode!r}  grid={self.G}x{self.G}  "
            f"n_workers={self.n_workers}  n_targets={self.n_targets}  "
            f"k=[{self.k_min},{self.k_max}]  "
            f"comm_radius={self.comm_radius}  sense_radius={self.sense_radius}  "
            f"wind_prob={self.wind_prob}  "
            f"obs_dim={self.obs_dim}  state_dim={self.state_dim}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _comm_bits(self):
        """[sensor_bit, w1_bit, w2_bit, w3_bit].
        Worker bit = 1 iff Manhattan(worker, sensor) <= comm_radius.
        Sensor bit = 1 iff at least one worker is in range.
        """
        sensor_pos = self.pos[0]
        worker_bits = [
            1.0 if self._manhattan(self.pos[wi], sensor_pos) <= self.comm_radius else 0.0
            for wi in range(1, self.n_agents)
        ]
        sensor_bit = 1.0 if any(b > 0.5 for b in worker_bits) else 0.0
        return [sensor_bit] + worker_bits

    def _n_workers_in_comm(self, comm_bits=None):
        if comm_bits is None:
            comm_bits = self._comm_bits()
        return sum(comm_bits[1:])

    def _n_free_workers_in_comm(self, comm_bits=None):
        """Number of UNLOADED workers within comm range of the sensor.

        Only free workers can participate in a lift, so loaded workers loitering
        near the sensor must not count toward the readiness gate.
        """
        if comm_bits is None:
            comm_bits = self._comm_bits()
        return sum(
            1.0 for wi in range(1, self.n_agents)
            if comm_bits[wi] > 0.5 and not self.loaded[wi - 1]
        )

    def _current_target(self):
        if self.current_target_idx < self.n_targets:
            return self.target_pos[self.current_target_idx]
        return None

    def _current_K(self):
        if self.current_target_idx < self.n_targets:
            return self.K[self.current_target_idx]
        return self.k_max

    def _sensor_on_target(self, target_pos):
        return self._manhattan(self.pos[0], target_pos) <= self.sense_radius

    def _move_toward(self, agent_idx, target):
        """Low-level controller: one step toward target. x wins ties. No-op if on target."""
        x, y = self.pos[agent_idx]
        tx, ty = target[0], target[1]
        dx, dy = tx - x, ty - y
        if dx == 0 and dy == 0:
            return
        if abs(dx) >= abs(dy):
            nx, ny = x + (1 if dx > 0 else -1), y
        else:
            nx, ny = x, y + (1 if dy > 0 else -1)
        if 0 <= nx < self.G and 0 <= ny < self.G:
            self.pos[agent_idx] = [nx, ny]

    def _step_penalty(self):
        """Per-step time cost, scaled by the team's distance to the active target.

        Linear from REWARD_STEP_FAR at maximum Manhattan distance to
        REWARD_STEP_NEAR when standing on the target. Averaged over the sensor
        plus every FREE worker — loaded workers are spent, so their position no
        longer matters to the task.
        """
        target_pos = self._current_target()
        if target_pos is None:
            return self.REWARD_STEP_NEAR
        who = [0] + [wi for wi in range(1, self.n_agents) if not self.loaded[wi - 1]]
        d = float(np.mean([self._manhattan(self.pos[i], target_pos) for i in who]))
        frac = min(d / (2 * (self.G - 1)), 1.0)
        return self.REWARD_STEP_NEAR + (self.REWARD_STEP_FAR - self.REWARD_STEP_NEAR) * frac

    def _knows_K(self, ai):
        """True iff worker `ai` can currently SEE K.

        Exactly the condition that makes observation field 1 (target_requirement)
        a real value instead of -1.0: the worker is within comm_radius of the
        sensor AND the sensor is standing on the active target.

        Used to gate the PICKUP action (see _avail_for). The gate deliberately
        uses only information the worker already has in its OWN observation --
        its connectivity to the sensor and the broadcast K.

        It must NOT be widened to `N_comm_free >= K`. As of 2026-08-19 the worker
        DOES observe that count directly (obs field 3, see the "Readiness
        broadcast" note in the module docstring), so the old rationale -- that the
        count lives only on the sensor -- no longer applies. The reason now is
        different: premature has to stay REACHABLE. If the mask enforced
        readiness, "do not lift before the team is assembled" would be satisfied
        by construction and the timing half of the task would stop being learned
        at all. The policy must read field 3 and decide, not be prevented.
        """
        target_pos = self._current_target()
        if target_pos is None:
            return False
        return (
            self._comm_bits()[ai] > 0.5
            and self._sensor_on_target(target_pos)
        )

    def _reset_diagnostics(self):
        """Per-episode counters returned in `info` on every step.

        The parallel runner reads `info` only at episode termination and SUMS each
        scalar key across episodes, then logs it as `<key>_mean` (total divided by
        n_episodes). So these must be cumulative over the episode, not per-step.

        The pair that matters for diagnosing exploration failure is
        `assembled_steps` vs `correct_pickups`:
          - assembled_steps == 0  -> the team never even gathers on the target, so
            the bottleneck is navigation, not cardinality selection.
          - assembled_steps > 0 but correct_pickups == 0 -> the team reaches the
            state where a lift is possible but never samples exactly K PICKUPs.
            That is a genuine exploration failure in the joint action.
        """
        self._diag = {
            "assembled_steps":    0,   # steps where N_comm_free >= K (lift possible)
            "pickup_steps":       0,   # steps where >=1 free worker attempted PICKUP
            "correct_pickups":    0,   # exactly K, all on target -> +10
            "overshoot_pickups":  0,   # n_pickup > K
            "undershoot_pickups": 0,   # 1..K-1 while assembled (terminates)
            "offtarget_pickups":  0,   # exactly K but not all on the target cell
            "premature_pickups":  0,   # any PICKUP while N_comm_free < K
            "targets_lifted":     0,
        }

    def _apply_drift(self, agent_idx):
        """Random drift in a valid direction with probability wind_prob."""
        if self._rng.random() >= self.wind_prob:
            return
        x, y = self.pos[agent_idx]
        candidates = []
        if y + 1 < self.G: candidates.append((0,  1))
        if y - 1 >= 0:     candidates.append((0, -1))
        if x + 1 < self.G: candidates.append((1,  0))
        if x - 1 >= 0:     candidates.append((-1, 0))
        if candidates:
            dx, dy = candidates[self._rng.integers(len(candidates))]
            self.pos[agent_idx] = [x + dx, y + dy]

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t                 = 0
        self.targets_lifted     = 0
        self.current_target_idx = 0
        self.loaded             = [False] * self.n_workers
        self._reset_diagnostics()

        # Sample K with feasibility constraint: sum(K) <= n_workers
        self.K = []
        free = self.n_workers
        for i in range(self.n_targets):
            remaining_targets = self.n_targets - i
            k_upper = min(self.k_max, free - remaining_targets + 1)
            k_upper = max(k_upper, self.k_min)
            k = int(self._rng.integers(self.k_min, k_upper + 1))
            self.K.append(k)
            free -= k

        # Place agents at distinct positions
        occupied = set()
        self.pos = []
        for _ in range(self.n_agents):
            while True:
                x = int(self._rng.integers(self.G))
                y = int(self._rng.integers(self.G))
                if (x, y) not in occupied:
                    occupied.add((x, y))
                    self.pos.append([x, y])
                    break

        # Place targets at distinct positions
        target_set = set()
        self.target_pos = []
        for _ in range(self.n_targets):
            while True:
                x = int(self._rng.integers(self.G))
                y = int(self._rng.integers(self.G))
                if (x, y) not in target_set:
                    target_set.add((x, y))
                    self.target_pos.append([x, y])
                    break

        self._last_actions = None
        self._last_pickers = []
        self._last_event   = None
        self._trails       = [[tuple(p)] for p in self.pos]
        self._episode_idx += 1
        self._frame_idx    = 0

        if self.render_flag:
            self.render()

        return self.get_obs(), {}

    def step(self, actions):
        a = [int(x) for x in actions]
        target_pos = self._current_target()
        self._last_actions = list(a)   # render-only; see _last_event below
        self._last_event   = None
        self._last_pickers = []

        # 1. Movement
        if self.obs_mode == "simple":
            for ai in range(self.n_agents):
                if a[ai] == self.MOVE and target_pos is not None:
                    self._move_toward(ai, target_pos)
        else:
            for ai in range(self.n_agents):
                if a[ai] in self._DELTAS:
                    dx, dy = self._DELTAS[a[ai]]
                    x, y = self.pos[ai]
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.G and 0 <= ny < self.G:
                        self.pos[ai] = [nx, ny]

        pickup_action = self.PICKUP if self.obs_mode == "simple" else self.HANDLE

        # 2. Stochastic drift — workers only, skipped for workers attempting pickup
        for wi in range(1, self.n_agents):
            if a[wi] != pickup_action:
                self._apply_drift(wi)

        # 3. Recompute comm state
        comm_bits   = self._comm_bits()
        N_comm_free = self._n_free_workers_in_comm(comm_bits)
        target_pos  = self._current_target()   # re-fetch after drift (target pos unchanged)
        K           = self._current_K()

        # 4. Step penalty — scaled by distance to the active target
        reward     = self._step_penalty()
        terminated = False

        # 5. Pickup resolution — only unloaded workers count
        if target_pos is not None:
            d = self._diag
            # Opportunity counter, incremented whether or not anyone tries to lift:
            # enough free workers are assembled that a correct lift is POSSIBLE now.
            if N_comm_free >= K:
                d["assembled_steps"] += 1

            pickers = [
                wi for wi in range(1, self.n_agents)
                if a[wi] == pickup_action and not self.loaded[wi - 1]
            ]
            n_pickup = len(pickers)
            # Kept for the renderer: after a successful lift the pickers are
            # already `loaded`, so a frame drawn post-step could not otherwise
            # tell who committed — and that is exactly the frame worth printing.
            self._last_pickers = list(pickers)

            if n_pickup > 0:
                d["pickup_steps"] += 1
                if N_comm_free < K:
                    reward    += self.REWARD_PREMATURE
                    terminated = True
                    d["premature_pickups"] += 1
                    self._last_event = ("premature", n_pickup, K)
                elif n_pickup > K:
                    reward    += self.REWARD_OVERSHOOT
                    terminated = True
                    d["overshoot_pickups"] += 1
                    self._last_event = ("overshoot", n_pickup, K)
                elif n_pickup < K:
                    reward    += self.REWARD_UNDERSHOOT
                    terminated = True
                    d["undershoot_pickups"] += 1
                    self._last_event = ("undershoot", n_pickup, K)
                else:
                    # Exact count: all pickers must be on target
                    if self.obs_mode == "simple":
                        on_tgt = all(
                            tuple(self.pos[wi]) == tuple(target_pos) for wi in pickers
                        )
                    else:
                        on_tgt = all(
                            self._manhattan(self.pos[wi], target_pos) <= 1 for wi in pickers
                        )
                    if on_tgt:
                        reward += self.REWARD_CORRECT_PICKUP
                        d["correct_pickups"] += 1
                        for wi in pickers:
                            self.loaded[wi - 1] = True
                        self.targets_lifted     += 1
                        self.current_target_idx += 1
                        self._last_event = ("correct", n_pickup, K)
                        if self.targets_lifted == self.n_targets:
                            reward    += self.REWARD_EPISODE_COMPLETION
                            terminated = True
                    else:
                        reward    += self.REWARD_UNDERSHOOT
                        terminated = True
                        d["offtarget_pickups"] += 1
                        self._last_event = ("offtarget", n_pickup, K)

        if self._trails is not None:
            for ai in range(self.n_agents):
                self._trails[ai].append(tuple(self.pos[ai]))
                if len(self._trails[ai]) > max(1, self.render_trail):
                    self._trails[ai].pop(0)

        self._t += 1
        self._diag["targets_lifted"] = self.targets_lifted
        truncated = self._t >= self._time_limit

        if self.render_flag:
            import time as _time
            self.render()
            _time.sleep(0.2)

        # dict() copy: the runner stores this reference and sums it later, so it
        # must not alias the live counters.
        return self.get_obs(), float(reward), terminated, truncated, dict(self._diag)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def get_obs(self):
        return tuple(self.get_obs_agent(ai) for ai in range(self.n_agents))

    def get_obs_agent(self, agent_id):
        if self.pos is None:
            return np.zeros(self.obs_dim, dtype=np.float32)

        target_pos = self._current_target()
        K          = self._current_K()

        if self.obs_mode == "simple":
            return self._obs_simple(agent_id, target_pos, K)
        return self._obs_extended(agent_id, target_pos, K)

    def _obs_simple(self, agent_id, target_pos, K):
        on_target = 1.0 if (
            target_pos is not None and
            tuple(self.pos[agent_id]) == tuple(target_pos)
        ) else 0.0

        sensor_on = (target_pos is not None and self._sensor_on_target(target_pos))
        comm_bits = self._comm_bits()
        targets_remaining = (self.n_targets - self.targets_lifted) / self.n_targets

        if agent_id == 0:
            # Sensor: reads K only when standing on target, else -1.0. Field 3 is
            # the number of FREE workers currently connected to the sensor — the
            # readiness signal the workers must read off the sensor via attention.
            k_field = K / self.k_max if sensor_on else -1.0
            n_free_conn = self._n_free_workers_in_comm(comm_bits) / self.n_workers
            return np.array([on_target, k_field, -1.0, n_free_conn, targets_remaining], dtype=np.float32)
        else:
            # Worker: receives K through the sensor when in communication range
            # AND the sensor is currently on the target (i.e. K is known). When
            # out of range or the sensor hasn't sensed yet, K stays hidden (-1.0).
            # Field 3 is the same readiness count the sensor reports, gated on
            # in_comm so a disconnected worker reads 0.0.
            in_comm     = comm_bits[agent_id] > 0.5
            k_field     = K / self.k_max if (in_comm and sensor_on) else -1.0
            is_loaded   = 1.0 if self.loaded[agent_id - 1] else 0.0
            n_free_conn = (
                self._n_free_workers_in_comm(comm_bits) / self.n_workers
                if in_comm
                else 0.0
            )
            return np.array([on_target, k_field, is_loaded, n_free_conn, targets_remaining], dtype=np.float32)

    def _obs_extended(self, agent_id, target_pos, K):
        comm_bits = self._comm_bits()
        N_comm    = self._n_workers_in_comm(comm_bits)

        tx = target_pos[0] / self.G if target_pos is not None else 0.0
        ty = target_pos[1] / self.G if target_pos is not None else 0.0
        target_lifted     = 1.0 if self.targets_lifted > 0 else 0.0
        targets_remaining = (self.n_targets - self.targets_lifted) / self.n_targets
        x, y = self.pos[agent_id]

        if agent_id == 0:
            sensor_on = (target_pos is not None and self._sensor_on_target(target_pos))
            k_visible = K / self.k_max if sensor_on else 0.0
            return np.array([
                x / self.G, y / self.G, tx, ty,
                k_visible, N_comm / self.n_workers,
                target_lifted, targets_remaining,
                0.0, 0.0,
            ], dtype=np.float32)
        else:
            in_comm    = comm_bits[agent_id] > 0.5
            sensor_on  = (target_pos is not None and self._sensor_on_target(target_pos))
            k_broadcast = K / self.k_max if (in_comm and sensor_on) else 0.0
            sx, sy = self.pos[0]
            return np.array([
                x / self.G, y / self.G,
                sx / self.G, sy / self.G,
                tx, ty,
                k_broadcast, N_comm / self.n_workers,
                target_lifted, targets_remaining,
            ], dtype=np.float32)

    # ------------------------------------------------------------------
    # State (privileged — centralized critic only)
    # ------------------------------------------------------------------

    def get_state(self):
        if self.pos is None:
            return np.zeros(self.state_dim, dtype=np.float32)

        comm_bits  = self._comm_bits()
        N_comm     = self._n_workers_in_comm(comm_bits)
        target_pos = self._current_target()
        K          = self._current_K()

        tx = target_pos[0] / self.G if target_pos is not None else 0.0
        ty = target_pos[1] / self.G if target_pos is not None else 0.0
        target_lifted     = 1.0 if self.targets_lifted > 0 else 0.0
        targets_remaining = (self.n_targets - self.targets_lifted) / self.n_targets

        state = []
        for ai in range(self.n_agents):
            state.extend([self.pos[ai][0] / self.G, self.pos[ai][1] / self.G])
        state.extend([tx, ty, K / self.k_max, N_comm / self.n_workers,
                      target_lifted, targets_remaining])
        state.extend(comm_bits)
        return np.array(state, dtype=np.float32)  # 18 fields

    # ------------------------------------------------------------------
    # Action availability
    # ------------------------------------------------------------------

    def get_avail_actions(self):
        return [self._avail_for(ai) for ai in range(self.n_agents)]

    def _avail_for(self, ai):
        # PICKUP/HANDLE is blocked for the sensor, for spent (loaded) workers, and
        # for any worker that cannot currently see K (see _knows_K). Premature
        # lifts stay reachable: the gate does not check that K workers are there.
        can_lift = ai != 0 and not self.loaded[ai - 1] and self._knows_K(ai)

        avail = [0] * self.n_actions
        if self.obs_mode == "simple":
            avail[self.MOVE] = 1
            avail[self.WAIT] = 1
            avail[self.PICKUP] = 1 if can_lift else 0
        else:
            avail[self.STAY]  = 1
            x, y = self.pos[ai]
            avail[self.NORTH] = 1 if y + 1 < self.G else 0
            avail[self.SOUTH] = 1 if y - 1 >= 0 else 0
            avail[self.EAST]  = 1 if x + 1 < self.G else 0
            avail[self.WEST]  = 1 if x - 1 >= 0 else 0
            avail[self.HANDLE] = 1 if can_lift else 0
        return avail

    # ------------------------------------------------------------------
    # Comm mask & env info
    # ------------------------------------------------------------------

    def get_comm_mask(self):
        bits = self._comm_bits()
        return np.array([[b] for b in bits], dtype=np.float32)

    def get_env_info(self):
        return {
            "n_agents":      self.n_agents,
            "n_actions":     self.n_actions,
            "obs_shape":     self.obs_dim,
            "state_shape":   self.state_dim,
            "episode_limit": self._time_limit,
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
    def _spread(n, k, radius=0.24):
        """Offset for the k-th of n agents sharing one cell, so they never
        overlap into an unreadable blob."""
        if n <= 1:
            return 0.0, 0.0
        ang = 2.0 * np.pi * k / n + np.pi / 2.0
        return radius * np.cos(ang), radius * np.sin(ang)

    def render(self):
        """Figure-quality top-down view of the field.

        The scene is composed to make the benchmark's actual bottleneck legible
        in a single still, which is what a paper figure has to do:

          * the CARGO CRATE carries its requirement placard — `K = 3` once the
            sensor is standing on it and reading it, `K = ?` while it is hidden,
            so "the count is not known yet" is visible rather than implied;
          * the sensor's radio footprint is drawn as the Manhattan diamond the
            env actually uses, and every worker inside it gets a dashed link
            carrying a `K` packet — that is exactly the set of workers whose
            observation field 1 is a real number instead of -1;
          * workers that committed PICKUP this step are haloed and tagged, so
            the exact-cardinality decision (how many committed vs. how many were
            required) is readable off the picture;
          * the outcome badge names the resolution — CORRECT LIFT / OVERSHOOT /
            UNDERSHOOT / PREMATURE — which is the quantity the benchmark scores.

        Configure via attributes (also accepted as env_args):
            render_theme    "paper" (print) | "night" (slides) | "blueprint"
            render_save_dir  write PNG frames here instead of opening a window
            render_dpi, render_pause, render_prefix, render_note, render_trail
        """
        dr = self._render_kit()

        fig = getattr(self, "_fig", None)
        if fig is None:
            fig, ax, th = dr.stage(
                figsize=(8.4, 9.0),
                theme_name=self.render_theme,
                save_dir=self.render_save_dir,
                window_title="Aerial Ordering",
            )
            self._fig, self._ax = fig, ax
        else:
            ax = self._ax
            th = dr.reuse(fig, ax, self.render_theme)
        fig.texts.clear()
        fig.patches.clear()

        G = self.G
        # generous padding: at the lift moment the whole team rings the crate,
        # and a tight field would clip the outermost drones.
        dr.field(ax, 0, G - 1, 0, G - 1, th, grid_step=1, major_every=5, pad=1.05,
                 terrain=self.render_terrain,
                 terrain_seed=self.render_terrain_seed,
                 terrain_contrast=self.render_terrain_contrast)

        comm_bits   = self._comm_bits()
        N_comm_free = int(self._n_free_workers_in_comm(comm_bits))
        active_idx  = self.current_target_idx
        active_cell = self._current_target()
        sensor_xy   = tuple(self.pos[0])
        sensor_on   = (active_cell is not None and self._sensor_on_target(active_cell))
        K_active    = self._current_K() if active_cell is not None else None
        acts        = self._last_actions

        # --- radio footprint of the sensor (Manhattan ball, as in _comm_bits)
        # No caption here: at the decisive moment the whole team rings this
        # spot, and the radius is already stated in the subtitle.
        dr.comm_disk(
            ax, sensor_xy[0], sensor_xy[1], self.comm_radius, th,
            manhattan=self.comm_radius > 0,
        )

        # --- cargo crates ------------------------------------------------
        just_lifted = (active_idx - 1) if (self._last_event
                                          and self._last_event[0] == "correct") else None
        for ti, (tx, ty) in enumerate(self.target_pos):
            if ti == just_lifted:
                state, label, sub = "lifting", None, None
            elif ti < active_idx:
                state, label, sub = "done", None, f"T{ti} lifted"
            elif ti == active_idx:
                # The active crate carries no caption — drones ring this cell —
                # so "hidden vs broadcast" is stated in the HUD instead.
                state = "active"
                label = f"K={self.K[ti]}" if sensor_on else "K=?"
                sub = None
            else:
                state, label, sub = "pending", "K=?", f"T{ti} queued"
            dr.crate(ax, tx, ty, size=0.86, th=th, label=label, sub=sub, state=state)

        # --- sensing ping: concentric arcs while the sensor reads K ------
        if sensor_on and active_cell is not None:
            from matplotlib.patches import Circle

            for i, r in enumerate((0.55, 0.80, 1.05)):
                ax.add_patch(Circle(tuple(active_cell), r, facecolor="none",
                                    edgecolor=th["sensor"], lw=1.4,
                                    alpha=0.45 - 0.12 * i, linestyle=(0, (3, 3)),
                                    zorder=2.2))

        # --- comm links sensor <-> workers in range ----------------------
        for wi in range(1, self.n_agents):
            if comm_bits[wi] <= 0.5:
                continue
            p, q = sensor_xy, tuple(self.pos[wi])
            if p == q:
                continue
            dr.comm_link(ax, p, q, th, alpha=0.8 if sensor_on else 0.45,
                         lw=1.6 if sensor_on else 1.1)
            if sensor_on:
                mx, my = (p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0
                ax.text(mx, my, f"K={K_active}", ha="center", va="center",
                        color=th["comm"], fontsize=6.4, fontweight="bold",
                        zorder=4.5, path_effects=dr.stroke(th["field_top"], 2.6))

        # --- trails ------------------------------------------------------
        if self._trails is not None and self.render_trail > 1:
            for ai in range(self.n_agents):
                c = th["sensor"] if ai == 0 else dr.agent_color(th, ai - 1)
                dr.trail(ax, self._trails[ai], c, th, alpha=0.60)

        # --- agents ------------------------------------------------------
        from collections import defaultdict

        by_cell = defaultdict(list)
        for ai in range(self.n_agents):
            by_cell[tuple(self.pos[ai])].append(ai)

        pickup_action = self.PICKUP if self.obs_mode == "simple" else self.HANDLE
        crate_cells = {tuple(c) for c in self.target_pos}
        n_committed = 0
        for cell, ais in by_cell.items():
            n_here = len(ais)
            # A cell holding a crate gets a WIDE ring so the crate — and its K
            # placard — stays visible in the middle of the drones lifting it;
            # that arrangement is the money shot of this benchmark.
            on_crate = cell in crate_cells
            if n_here == 1:
                ring, glyph = 0.0, 0.33
            elif on_crate:
                ring, glyph = 0.95, 0.24
            else:
                ring, glyph = 0.36, 0.26
            for k, ai in enumerate(ais):
                dx, dy = self._spread(n_here, k, radius=ring)
                x, y = cell[0] + dx, cell[1] + dy
                if ai == 0:
                    dr.drone(ax, x, y, size=glyph, color=th["sensor"], th=th,
                             label="S", halo=th["sensor"] if sensor_on else None,
                             halo_alpha=0.20, zorder=7)
                    continue
                loaded = self.loaded[ai - 1]
                committing = ai in self._last_pickers
                n_committed += int(committing)
                lift_color = (th["ok"] if (self._last_event
                                           and self._last_event[0] == "correct")
                              else th["warn"])
                if committing and on_crate:
                    # taut lift tether to the crate
                    ax.plot([x, cell[0]], [y, cell[1]], color=lift_color, lw=1.8,
                            alpha=0.9, solid_capstyle="round", zorder=5.5)
                dr.drone(
                    ax, x, y, size=glyph, color=dr.agent_color(th, ai - 1), th=th,
                    label=f"W{ai}",
                    # a drone that just committed stays bright for this one
                    # frame, then fades once it is spent
                    dimmed=loaded and not committing,
                    carrying=loaded and not on_crate,
                    halo=lift_color if committing else None, halo_alpha=0.32,
                    zorder=7,
                )
                if committing:
                    ax.text(x, y + glyph * 1.85, "LIFT", ha="center", va="bottom",
                            color=lift_color, fontsize=6.6, fontweight="bold",
                            zorder=8, path_effects=dr.stroke(th["field_top"], 2.4))
                elif comm_bits[ai] > 0.5 and not loaded:
                    # compact dot instead of a "linked" caption: several drones
                    # share a cell at the decisive moment and captions collide.
                    dr.badge_dot(ax, x + glyph * 1.05, y + glyph * 1.05,
                                 th["comm"], th)

        # --- readiness banner over the active crate ----------------------
        if active_cell is not None and K_active is not None:
            ready = N_comm_free >= K_active
            msg = (f"ASSEMBLED  {N_comm_free}/{K_active} — exactly {K_active} must lift"
                   if ready else
                   f"gathering  {N_comm_free}/{K_active} free workers linked")
            ax.text(
                (G - 1) / 2.0, G - 1 + 0.42, msg, ha="center", va="center",
                color="#FFFFFF" if ready else th["muted"], fontsize=8.2,
                fontweight="bold", zorder=9,
                bbox=dict(boxstyle="round,pad=0.36",
                          facecolor=th["ok"] if ready else th["panel"],
                          edgecolor="none" if ready else th["panel_edge"],
                          alpha=0.95),
            )

        if self.wind_prob > 0:
            ax.text(G - 1 + 0.40, -0.44, f"↺ wind p={self.wind_prob:g}",
                    ha="right", va="center", color=th["muted"], fontsize=7.0,
                    fontweight="bold", zorder=6)
        dr.compass(ax, -0.28, G - 1.15, th)

        # --- HUD ---------------------------------------------------------
        dr.header(
            fig, ax, "Aerial Ordering",
            f"{self.n_workers} interchangeable carriers + 1 sensor · grid {G}×{G} · "
            f"radio r={self.comm_radius} · K∈[{self.k_min},{self.k_max}] · exactly K "
            f"must lift in the same step",
            th,
        )
        dr.stat_row(fig, [
            ("step", f"{self._t}/{self._time_limit}"),
            ("required K" if (sensor_on or K_active is None) else "required K (hidden)",
             str(K_active) if K_active is not None else "—",
             th["ink"] if sensor_on else th["muted"]),
            ("linked free", f"{N_comm_free}/{self.n_workers}",
             th["ok"] if (K_active and N_comm_free >= K_active) else None),
            ("committed", str(n_committed),
             th["warn"] if n_committed else None),
            ("lifted", f"{self.targets_lifted}/{self.n_targets}"),
        ], th)

        badge = {
            "correct":   ("CORRECT LIFT", th["ok"]),
            "overshoot": ("OVERSHOOT", th["bad"]),
            "undershoot": ("UNDERSHOOT", th["bad"]),
            "offtarget": ("OFF-TARGET LIFT", th["bad"]),
            "premature": ("PREMATURE", th["warn"]),
        }
        if self._last_event is not None:
            kind, n_pick, K_ev = self._last_event
            text, color = badge[kind]
            dr.status_pill(fig, f"{text}   {n_pick} committed / K={K_ev}", color, th)

        dr.legend_row(fig, [
            (th["sensor"], "sensor (reads K on the crate)"),
            (dr.agent_color(th, 0), "carrier drones"),
            (th["comm"], "radio link carrying K"),
            (th["warn"], "committed to lift"),
        ], th)
        dr.timeline(fig, self._t, self._time_limit, th)
        dr.footer(fig, "Aerial Ordering · MAC benchmark", th,
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
    # 1. Simple mode — obs structure
    # ------------------------------------------------------------------
    print("\n=== 1. Simple mode obs structure ===")
    e = AerialOrderingEnv(obs_mode="simple", wind_prob=0.0, comm_radius=2, seed=0)
    e.reset(seed=0)
    # Place sensor on target so K is visible
    tgt = e.target_pos[0]
    e.pos[0] = list(tgt)   # sensor on target
    e.pos[1] = list(tgt)   # worker1 on target → in comm range, sees K
    e.pos[2] = [(tgt[0] + 5) % e.G, (tgt[1] + 5) % e.G]  # worker2 far → out of comm

    s_obs  = e.get_obs_agent(0)
    w1_obs = e.get_obs_agent(1)
    w2_obs = e.get_obs_agent(2)

    print(f"  Sensor  obs: on_target={s_obs[0]:.0f}  target_req={s_obs[1]:.3f}  is_loaded={s_obs[2]:.0f}  n_free_conn={s_obs[3]:.3f}  targets_rem={s_obs[4]:.3f}")
    print(f"  Worker1 obs: on_target={w1_obs[0]:.0f}  target_req={w1_obs[1]:.3f}  is_loaded={w1_obs[2]:.0f}  n_free_conn={w1_obs[3]:.3f}  targets_rem={w1_obs[4]:.3f}")
    print(f"  Worker2 obs: on_target={w2_obs[0]:.0f}  target_req={w2_obs[1]:.3f}  is_loaded={w2_obs[2]:.0f}  n_free_conn={w2_obs[3]:.3f}  targets_rem={w2_obs[4]:.3f}")

    in_range = e._manhattan(e.pos[2], e.pos[0]) <= e.comm_radius
    check(e.obs_dim == 5, "simple obs_dim == 5")
    check(e.n_actions == 3, "simple n_actions == 3")
    check(s_obs[0] == 1.0, "sensor on_target == 1 when on target")
    check(s_obs[1] > 0.0, "sensor target_req > 0 when on target (K visible)")
    check(s_obs[2] == -1.0, "sensor is_loaded == -1")
    check(abs(s_obs[3] - 1.0 / 3.0) < 1e-6, "sensor n_free_conn == 1/3 (only W1 in comm, free)")
    check(w1_obs[1] > 0.0, "worker1 target_req > 0 (in comm range, K received from sensor)")
    check(abs(w1_obs[1] - s_obs[1]) < 1e-6, "worker1 target_req matches sensor's K/k_max")
    check(w1_obs[2] == 0.0, "worker1 is_loaded == 0 at start")
    # Field 3 is the readiness COUNT as of 2026-08-19, not a connectivity bit. A
    # connected worker reads exactly what the sensor reads; a disconnected one
    # reads 0.0, so the zero/non-zero connectivity meaning is preserved.
    check(abs(w1_obs[3] - s_obs[3]) < 1e-6,
          "worker1 n_free_conn matches the sensor's readiness count (in comm range)")
    check(w1_obs[3] > 0.0, "worker1 n_free_conn > 0 (connected free worker counts itself)")
    check(not in_range and w2_obs[1] == -1.0, "worker2 target_req == -1 (out of comm range)")
    check(w2_obs[0] == 0.0, "worker2 on_target == 0 when not on target")
    check(w2_obs[3] == 0.0, "worker2 n_free_conn == 0 (out of comm range, learns nothing)")
    check(s_obs.shape == (5,), "sensor obs shape (5,)")
    check(w1_obs.shape == (5,), "worker obs shape (5,)")

    # ------------------------------------------------------------------
    # 2. K feasibility constraint
    # ------------------------------------------------------------------
    print("\n=== 2. K feasibility constraint (sum <= n_workers) ===")
    for trial in range(100):
        e2 = AerialOrderingEnv(obs_mode="simple", wind_prob=0.0, seed=trial)
        e2.reset(seed=trial)
        total_K = sum(e2.K)
        if total_K > e2.n_workers:
            check(False, f"trial {trial}: sum(K)={total_K} > n_workers={e2.n_workers}")
            break
    else:
        check(True, f"sum(K) <= n_workers={e2.n_workers} across 100 episodes")
    print(f"  Example K values from last trial: {e2.K} (sum={sum(e2.K)})")

    # ------------------------------------------------------------------
    # 3. MOVE low-level controller
    # ------------------------------------------------------------------
    print("\n=== 3. MOVE low-level controller ===")
    e = AerialOrderingEnv(obs_mode="simple", wind_prob=0.0, seed=5)
    e.reset(seed=5)
    e.pos[0] = [0, 0]
    tgt = e.target_pos[0]
    e.pos[1] = [0, 0]  # worker far from target
    before = list(e.pos[1])
    e.step([e.MOVE, e.MOVE, e.WAIT, e.WAIT])
    after = list(e.pos[1])
    moved = before != after
    check(moved, f"MOVE advances worker toward target: {before} -> {after}")
    # Sensor on target → no-op
    e.pos[0] = list(tgt)
    before_s = list(e.pos[0])
    e.step([e.MOVE, e.WAIT, e.WAIT, e.WAIT])
    check(list(e.pos[0]) == before_s, "MOVE on target cell is no-op for sensor")

    # ------------------------------------------------------------------
    # 4. is_loaded mechanic
    # ------------------------------------------------------------------
    print("\n=== 4. is_loaded mechanic ===")
    e = AerialOrderingEnv(obs_mode="simple", wind_prob=0.0, k_min=1, k_max=1,
                             comm_radius=8, seed=0)
    e.reset(seed=0)
    # Force K=1, put sensor + worker1 on target, all in comm range
    e.K = [1, 1]
    tgt = e.target_pos[0]
    e.pos[0] = list(tgt)   # sensor on target
    e.pos[1] = list(tgt)   # worker1 on target
    e.pos[2] = [1, 0]
    e.pos[3] = [2, 0]

    check(e.loaded == [False, False, False], "all unloaded at reset")
    # Step penalty is position-dependent now; capture it before the step. Nobody
    # moves here (WAIT/PICKUP, wind off), so this is the value step() will use.
    sp = e._step_penalty()
    obs, r, term, trunc, _ = e.step([e.WAIT, e.PICKUP, e.WAIT, e.WAIT])
    check(e.loaded[0] == True, "worker1 is_loaded=True after successful pickup")
    check(e.loaded[1] == False, "worker2 still unloaded")
    check(obs[1][2] == 1.0, "worker1 obs[2] (is_loaded) == 1.0 after pickup")
    check(obs[2][2] == 0.0, "worker2 obs[2] (is_loaded) == 0.0")
    check(abs(r - (sp + e.REWARD_CORRECT_PICKUP)) < 0.01,
          f"correct pickup reward after K=1 lift: {r:.2f}")

    # Loaded worker PICKUP is masked. The sensor has to be moved onto the NEXT
    # target first: PICKUP is now gated on the worker being able to see K, and the
    # active target advanced when target 0 was lifted.
    tgt2 = e.target_pos[1]
    e.pos[0] = list(tgt2)   # sensor on the new active target
    e.pos[1] = list(tgt2)   # worker1 — loaded, so still masked
    e.pos[2] = list(tgt2)   # worker2 — unloaded and can see K
    avail = e.get_avail_actions()
    check(avail[1][e.PICKUP] == 0, "loaded worker1 PICKUP masked")
    check(avail[2][e.PICKUP] == 1, "unloaded worker2 PICKUP available (knows K)")
    check(avail[0][e.PICKUP] == 0, "sensor PICKUP always masked")

    # Loaded worker cannot count toward next pickup
    tgt2 = e.target_pos[1]
    e.pos[0] = list(tgt2)   # sensor moves to target2
    e.pos[1] = list(tgt2)   # worker1 (loaded) tries PICKUP — should not count
    e.pos[2] = list(tgt2)   # worker2 (unloaded) tries PICKUP
    e.pos[3] = [0, 0]
    # K[1]=1, so only 1 unloaded PICKUP needed. worker2 does it, worker1 is ignored.
    obs, r, term, trunc, _ = e.step([e.WAIT, e.PICKUP, e.PICKUP, e.WAIT])
    check(e.targets_lifted == 2, "both targets lifted (loaded worker PICKUP ignored)")
    check(term, "episode terminates after all targets lifted")

    # ------------------------------------------------------------------
    # 5. Simple mode reward logic
    # ------------------------------------------------------------------
    print("\n=== 5. Simple mode reward cases ===")

    def simple_env(K_val, sensor_pos, worker_positions, comm_r=8):
        # k_min stays 1 so the feasibility guard passes (n_targets*k_min <= n_workers);
        # the actual requirement is forced by overwriting e.K below.
        e = AerialOrderingEnv(obs_mode="simple", wind_prob=0.0,
                                 k_min=1, k_max=K_val, comm_radius=comm_r, seed=42)
        e.reset(seed=42)
        e.K = [K_val, K_val]
        e.target_pos = [[4, 4], [0, 0]]
        e.pos[0] = list(sensor_pos)
        for i, p in enumerate(worker_positions):
            e.pos[i + 1] = list(p)
        e.loaded = [False] * e.n_workers
        return e

    TGT = [4, 4]

    # Correct K=2: both pickers ON target
    e = simple_env(2, TGT, [TGT, TGT, [0, 0]])
    sp = e._step_penalty()
    _, r, _, _, _ = e.step([e.WAIT, e.PICKUP, e.PICKUP, e.WAIT])
    check(abs(r - (sp + e.REWARD_CORRECT_PICKUP)) < 0.01,
          f"correct K=2, both on target → {r:.2f}")

    # Overshoot: K=2, 3 workers PICKUP. Terminates — caps the episode at one
    # pickup attempt, same as premature/undershoot/offtarget.
    e = simple_env(2, TGT, [TGT, TGT, TGT])
    sp = e._step_penalty()
    _, r, term, _, _ = e.step([e.WAIT, e.PICKUP, e.PICKUP, e.PICKUP])
    check(abs(r - (sp + e.REWARD_OVERSHOOT)) < 0.01,
          f"overshoot K=2, 3 PICKUPs → {r:.2f}")
    check(term, "overshoot terminates (caps episode at one pickup attempt)")

    # Undershoot: K=2, 1 PICKUP
    e = simple_env(2, TGT, [TGT, TGT, [0, 0]])
    sp = e._step_penalty()
    _, r, _, _, _ = e.step([e.WAIT, e.PICKUP, e.WAIT, e.WAIT])
    check(abs(r - (sp + e.REWARD_UNDERSHOOT)) < 0.01,
          f"undershoot K=2, 1 PICKUP → {r:.2f}")

    # Exact K=2 but one NOT on target
    e = simple_env(2, TGT, [TGT, [3, 4], [0, 0]])  # worker2 off target
    sp = e._step_penalty()
    _, r, _, _, _ = e.step([e.WAIT, e.PICKUP, e.PICKUP, e.WAIT])
    check(abs(r - (sp + e.REWARD_UNDERSHOOT)) < 0.01,
          f"K=2 PICKUPs but one off target → {r:.2f}")

    # Premature: N_comm=0 < K=2. Terminates as of 2026-08-18 (see module docstring),
    # so its cost is capped at one -10 per episode, symmetric with undershoot.
    e = simple_env(2, TGT, [[7, 7], [7, 6], [6, 7]], comm_r=2)
    sp = e._step_penalty()
    _, r, term, _, _ = e.step([e.WAIT, e.PICKUP, e.WAIT, e.WAIT])
    check(abs(r - (sp + e.REWARD_PREMATURE)) < 0.01,
          f"premature N_comm<K → {r:.2f}")
    check(term, "premature TERMINATES the episode (cost capped at one -10)")

    # No pickups: reward is now the bare distance-scaled step penalty (no wait bonus)
    e = simple_env(2, TGT, [[7, 7], [7, 6], [6, 7]], comm_r=2)
    sp = e._step_penalty()
    _, r, _, _, _ = e.step([e.WAIT, e.WAIT, e.WAIT, e.WAIT])
    check(abs(r - sp) < 0.01, f"no pickups → bare step penalty {r:.2f}")

    # ------------------------------------------------------------------
    # 5b. Distance-scaled step penalty
    # ------------------------------------------------------------------
    print("\n=== 5b. Distance-scaled step penalty ===")
    e = simple_env(2, TGT, [TGT, TGT, TGT])          # everyone ON the target
    check(abs(e._step_penalty() - e.REWARD_STEP_NEAR) < 1e-6,
          f"all on target → REWARD_STEP_NEAR ({e._step_penalty():.3f})")
    e = simple_env(2, [0, 0], [[0, 0], [0, 0], [0, 0]])
    e.target_pos = [[7, 7], [0, 0]]                  # everyone at max distance
    check(abs(e._step_penalty() - e.REWARD_STEP_FAR) < 1e-6,
          f"all at max distance → REWARD_STEP_FAR ({e._step_penalty():.3f})")
    # target is [4,4]: [3,3] is 2 away, [7,7] is 6 away
    near = simple_env(2, [3, 3], [[3, 3], [3, 3], [3, 3]])._step_penalty()
    far  = simple_env(2, [7, 7], [[7, 7], [7, 7], [7, 7]])._step_penalty()
    check(near > far, f"closer to target is less costly ({near:.3f} > {far:.3f})")

    # ------------------------------------------------------------------
    # 5c. PICKUP gating — masked unless the worker can see K
    # ------------------------------------------------------------------
    print("\n=== 5c. PICKUP gated on knowing K ===")

    def gated_env(sensor_pos, worker_positions, comm_r=2):
        e = AerialOrderingEnv(obs_mode="simple", wind_prob=0.0, k_min=1, k_max=2,
                              comm_radius=comm_r, seed=1)
        e.reset(seed=1)
        e.target_pos = [[4, 4], [0, 0]]
        e.pos[0] = list(sensor_pos)
        for i, p in enumerate(worker_positions):
            e.pos[i + 1] = list(p)
        e.loaded = [False] * e.n_workers
        return e

    TG = [4, 4]
    # Sensor ON target, worker co-located -> in comm AND K visible -> allowed.
    e = gated_env(TG, [TG, [9, 9], [9, 8]])
    check(e.get_avail_actions()[1][e.PICKUP] == 1,
          "worker in comm + sensor on target -> PICKUP available")
    check(e.get_obs_agent(1)[1] > 0.0,
          "  ...and that worker's obs really does expose K (gate matches obs)")

    # Sensor OFF target: worker is still in comm, but K is unknown -> masked.
    e = gated_env([1, 1], [[1, 1], [9, 9], [9, 8]])
    check(e.get_avail_actions()[1][e.PICKUP] == 0,
          "worker in comm but sensor OFF target -> PICKUP masked")
    check(e.get_obs_agent(1)[1] == -1.0, "  ...and its obs shows K hidden (-1.0)")

    # Sensor on target but worker out of comm range -> masked.
    e = gated_env(TG, [[9, 9], [9, 8], [8, 9]])
    check(e.get_avail_actions()[1][e.PICKUP] == 0,
          "worker OUT of comm range -> PICKUP masked")

    # MOVE/WAIT are never masked, so an agent always has a legal action.
    for e in (gated_env(TG, [TG, [9, 9], [9, 8]]),
              gated_env([1, 1], [[9, 9], [9, 8], [8, 9]])):
        ok = all(sum(a) > 0 for a in e.get_avail_actions())
        check(ok, "every agent always has >=1 available action")

    # The gate must keep premature REACHABLE: a lone connected worker can still
    # choose PICKUP (and be punished for it) when K > 1. Enforcing readiness in
    # the mask would remove the timing decision instead of teaching it.
    e = gated_env(TG, [TG, [9, 9], [9, 8]])
    e.K = [2, 2]
    check(e._n_free_workers_in_comm() < e._current_K(),
          "setup: only one free worker connected, K=2")
    check(e.get_avail_actions()[1][e.PICKUP] == 1,
          "premature is still REACHABLE — the mask does not enforce readiness")

    # ------------------------------------------------------------------
    # 5d. Readiness broadcast — the state that pays +30 and the state that
    #     pays -10 must be DISTINGUISHABLE from a worker's own observation.
    #     This is the regression test for the 2026-08-19 field-3 change.
    # ------------------------------------------------------------------
    print("\n=== 5d. Readiness broadcast (worker can see n_free_connected) ===")

    def readiness_env(worker_positions, comm_r=3):
        e = AerialOrderingEnv(obs_mode="simple", wind_prob=0.0, k_min=2, k_max=2,
                              n_workers=3, n_targets=1, comm_radius=comm_r, seed=2)
        e.reset(seed=2)
        e.target_pos = [[5, 5]]
        e.current_target_idx = 0
        e.K = [2]
        e.pos[0] = [5, 5]                      # sensor ON the target
        for i, p in enumerate(worker_positions):
            e.pos[i + 1] = list(p)
        e.loaded = [False] * e.n_workers
        return e

    TT = [5, 5]
    # A: only worker1 assembled -> N_comm_free=1 < K -> lifting here is PREMATURE
    eA = readiness_env([TT, [0, 0], [9, 9]])
    # B: workers 1+2 assembled  -> N_comm_free=2 == K -> lifting here scores +30
    eB = readiness_env([TT, TT, [9, 9]])
    wA, wB = eA.get_obs_agent(1), eB.get_obs_agent(1)
    sA, sB = eA.get_obs_agent(0), eB.get_obs_agent(0)

    check(eA._n_free_workers_in_comm() == 1 and eB._n_free_workers_in_comm() == 2,
          "setup: N_comm_free is 1 (premature) vs 2 (solvable)")
    check(not np.allclose(wA, wB),
          f"worker1 can DISTINGUISH the two states: {wA} != {wB}")
    check(abs(wA[3] - 1 / 3) < 1e-6 and abs(wB[3] - 2 / 3) < 1e-6,
          f"worker1 field 3 tracks the count: {wA[3]:.3f} -> {wB[3]:.3f}")
    check(abs(wA[3] - sA[3]) < 1e-6 and abs(wB[3] - sB[3]) < 1e-6,
          "worker field 3 equals the sensor's field 3 in both states")
    check(eA.get_avail_actions()[1][eA.PICKUP] == 1,
          "PICKUP still available in the premature state (must stay reachable)")
    # Disconnected workers must learn nothing from the broadcast.
    check(eA.get_obs_agent(2)[3] == 0.0 and eA.get_obs_agent(3)[3] == 0.0,
          "out-of-range workers read 0.0 (field keeps its zero/non-zero meaning)")

    # ------------------------------------------------------------------
    # 6. Worker symmetry in simple mode
    # ------------------------------------------------------------------
    print("\n=== 6. Worker symmetry in simple mode ===")
    e = AerialOrderingEnv(obs_mode="simple", wind_prob=0.0, seed=7)
    e.reset(seed=7)
    tgt = e.target_pos[0]
    e.pos[0] = list(tgt)
    e.pos[1] = list(tgt)   # worker1 on target
    e.pos[2] = list(tgt)   # worker2 on target — same cell
    e.pos[3] = [0, 0]
    w1 = e.get_obs_agent(1)
    w2 = e.get_obs_agent(2)
    check(np.allclose(w1, w2),
          f"co-located unloaded workers → identical obs {w1} == {w2}")

    # ------------------------------------------------------------------
    # 7. Extended mode still works
    # ------------------------------------------------------------------
    print("\n=== 7. Extended mode sanity ===")
    e = AerialOrderingEnv(obs_mode="extended", wind_prob=0.0, seed=0)
    e.reset(seed=0)
    check(e.obs_dim == 10, "extended obs_dim == 10")
    check(e.n_actions == 6, "extended n_actions == 6")
    obs = e.get_obs()
    check(all(o.shape == (10,) for o in obs), "all extended obs shape (10,)")
    check(e.get_state().shape == (18,), "state shape (18,)")
    check(e.get_comm_mask().shape == (4, 1), "comm_mask shape (4,1)")

    # ------------------------------------------------------------------
    # 8. get_env_info mode-specific
    # ------------------------------------------------------------------
    print("\n=== 8. get_env_info ===")
    for mode, exp_obs, exp_act in [("simple", 5, 3), ("extended", 10, 6)]:
        info = AerialOrderingEnv(obs_mode=mode).get_env_info()
        check(info["obs_shape"] == exp_obs, f"{mode} obs_shape=={exp_obs}")
        check(info["n_actions"] == exp_act, f"{mode} n_actions=={exp_act}")
        check(info["state_shape"] == 18, f"{mode} state_shape==18")

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