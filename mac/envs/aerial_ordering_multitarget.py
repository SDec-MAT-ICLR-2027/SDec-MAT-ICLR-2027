"""Concurrent multi-target Aerial Ordering -- EVALUATION ONLY.

Used by scripts/generalization/aerial_multitarget.py.

`AerialOrderingEnv` runs one sensor and one active target at a time: targets are
lifted in sequence, and the whole swarm serves whichever is current. This
subclass runs T sensors, T targets and N workers CONCURRENTLY in one shared
arena, with every worker permanently assigned to one target at reset. It exists
to ask a question the base env cannot pose: does a policy trained on a single
five-worker swarm still form the right sub-swarm and count to K_t when several
independent teams occupy the same space and the same communication medium?

Nothing here is meant for training. There is no curriculum, no reward shaping
and no attempt to make the task learnable end-to-end; the reward constants,
movement controller, drift model and pickup semantics are the parent's,
unchanged, so a frozen single-target policy can be dropped in as-is.

What is deliberately NOT segregated
-----------------------------------
The communication graph. `comm_adjacency()` is built from physical proximity
alone: a worker is linked to EVERY sensor within `comm_radius`, its own or not,
and the connected components that follow are whatever the geometry produces. If
a worker crossing the arena passes through another sub-swarm, the two merge into
one component, and the policy sees the foreign agents' tokens in its encoder
attention and in its autoregressive decode chain. Filtering that is the
mechanism under test, so the environment must not do it.

What IS abstracted per sub-swarm
--------------------------------
The observation, and only the observation. A worker sees its ASSIGNED target
and nothing else -- its own target's K, its own sub-swarm's readiness count, its
own distance-zero flag -- so the vector keeps the five fields and the value
ranges the policy trained on, and its MOVE action drives it to its assigned
target. Without this the test would confound "can it filter cross-talk in
attention" with "can it read an observation it has never seen".

    Worker w assigned to target t:
        [ on_target_t, K_t/k_max if w can see K_t else -1.0, is_loaded,
          n_free_connected_t / n_t, targets_remaining_t ]
    Sensor t:
        [ on_target_t, K_t/k_max if on target else -1.0, -1.0,
          n_free_connected_t / n_t, targets_remaining_t ]

`n_free_connected_t` counts only free workers ASSIGNED to t that are within
`comm_radius` of sensor t -- a foreign worker standing on the same cell does not
inflate it. The denominator is the sub-swarm's own size n_t, which is what makes
the field mean the same thing it meant in training (with n_t equal to the
trained worker count the encoding is bit-identical). Field 4 is 1.0 while the
worker's own target is unresolved and 0.0 after, matching the single-target
env's targets_remaining.

Per-target resolution
---------------------
In the parent, ANY bad pickup ends the episode. With T independent crates that
would throw away the other targets' outcomes, so here a pickup resolves only the
target it was aimed at: that target is marked lifted or failed, its workers stop
counting, and the remaining sub-swarms carry on. The episode ends when every
target is resolved or the time limit hits. Rewards are the parent's constants,
summed over whatever resolved this step.

Assignment guarantee
--------------------
The sub-swarms are drawn FIRST and each crate is then sized to the team that got
it: K_t ~ U[k_min, n_t - 1], so target t's k_max IS its own sub-swarm's size
minus one. That is the relation the training config uses (5 workers, k_max 4),
and the observation depends on it -- field 1 is K_t / k_max_t and field 3 is
n_free_connected_t / n_t, so both are fractions of the same sub-swarm and stay
comparable. One global k_max shared across sub-swarms of different sizes would
break that and the policy would fail for an encoding reason rather than a
coordination one.

n_t >= K_t + 1 follows for free from the cap: every sub-swarm keeps at least one
spare, so "everybody commits" is never trivially correct.
`assign_mode="balanced"` (default) splits the swarm evenly and randomises WHICH
workers go where; `"random"` gives every sub-swarm k_min + 1 workers and
scatters the surplus, producing lopsided teams. The constructor rejects
n_workers < n_targets * (k_min + 1), below which some sub-swarm admits no valid
K at all.
"""

import numpy as np

from .aerial_ordering import AerialOrderingEnv


class AerialOrderingMultiTargetEvalEnv(AerialOrderingEnv):
    """T concurrent targets, T sensors, N workers, one shared arena."""

    # Agent layout: 0..T-1 are sensors (sensor t owns target t), T..T+N-1 are
    # workers. The policy has no agent ids, so nothing depends on this ordering
    # beyond the decode order inside a component.

    def __init__(self, n_targets=2, n_workers=10, assign_mode="balanced", **kwargs):
        kwargs.setdefault("obs_mode", "simple")
        if kwargs["obs_mode"] != "simple":
            raise NotImplementedError(
                "multi-target eval is implemented for obs_mode='simple' only; "
                "the extended mode's observation carries a single sensor "
                "position and would need its own per-target abstraction.")
        if assign_mode not in ("balanced", "random"):
            raise ValueError(f"assign_mode must be balanced|random, got {assign_mode!r}")

        # Build through the parent with ONE target so its own feasibility check
        # (n_targets * k_min <= n_workers) does not fire on the concurrent
        # layout, where the constraint is per sub-swarm rather than sequential.
        super().__init__(n_targets=1, n_workers=n_workers, **kwargs)

        self.n_targets = int(n_targets)
        self.assign_mode = assign_mode
        self.n_sensor = self.n_targets
        self.n_agents = self.n_sensor + self.n_workers

        if self.n_workers < self.n_targets * (self.k_min + 1):
            raise ValueError(
                f"n_workers({self.n_workers}) < n_targets({self.n_targets}) * "
                f"(k_min({self.k_min}) + 1): some sub-swarm would be too small "
                f"to hold k_min lifters plus a spare. Raise n_workers or lower "
                f"k_min.")
        if self.n_agents + self.n_targets > self.G * self.G:
            raise ValueError(
                f"grid {self.G}x{self.G} cannot hold {self.n_agents} agents and "
                f"{self.n_targets} targets at distinct cells.")

        # Positions of every agent, then per target [tx, ty, K/k_max, resolved],
        # then the per-agent comm bits. Never read by the policy (mat_comm
        # forbids encode_state and this evaluation passes the group mask
        # explicitly); kept coherent for logging and for a centralised critic.
        self.state_dim = 2 * self.n_agents + 4 * self.n_targets + self.n_agents

        self.assigned = None       # [n_agents] target index served by each agent
        self.lifted = None         # [n_targets] bool
        self.failed = None         # [n_targets] bool
        self.sub_swarm = None      # target -> array of worker agent indices
        self.k_max_t = None        # per target: its sub-swarm size - 1

        print(
            f"AerialOrderingMultiTargetEvalEnv (concurrent, EVAL ONLY): "
            f"grid={self.G}x{self.G}  n_targets={self.n_targets}  "
            f"n_workers={self.n_workers}  assign={self.assign_mode}  "
            f"k=[{self.k_min},{self.k_max}]  comm_radius={self.comm_radius}  "
            f"wind_prob={self.wind_prob}  obs_dim={self.obs_dim}"
        )

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def is_sensor(self, ai):
        return ai < self.n_sensor

    def worker_slot(self, ai):
        """Index into self.loaded for a worker agent."""
        return ai - self.n_sensor

    def _sensor_of(self, ai):
        """Agent index of the sensor serving the same target as `ai`."""
        return self.assigned[ai]

    def _target_of(self, ai):
        return self.assigned[ai]

    def _resolved(self, t):
        return self.lifted[t] or self.failed[t]

    # The parent's single-target accessors have no meaning here, and inheriting
    # them silently would produce a plausible-looking wrong answer.
    def _current_target(self):
        raise NotImplementedError("multi-target env: use self.target_pos[t]")

    def _current_K(self):
        raise NotImplementedError("multi-target env: use self.K[t]")

    def _comm_bits(self):
        raise NotImplementedError("multi-target env: use comm_adjacency()")

    # ------------------------------------------------------------------
    # Communication — physical, never segregated by assignment
    # ------------------------------------------------------------------

    def _connected_to_sensor(self, ai, t):
        """True iff agent `ai` is within comm_radius of sensor t."""
        return self._manhattan(self.pos[ai], self.pos[t]) <= self.comm_radius

    def comm_adjacency(self):
        """[n_agents, n_agents] float adjacency of the shared medium.

        One clique per sensor -- {sensor t} together with every worker within
        comm_radius of it, REGARDLESS of assignment. That is exactly the graph
        the single-sensor env produces (its per-agent bits are the outer product
        of one such clique), applied once per hub. Cliques that share a worker
        merge under the transitive closure the policy's `build_group_mask`
        applies, which is how a passing worker drags two sub-swarms into one
        decode chain.
        """
        adj = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        for t in range(self.n_targets):
            members = [t] + [ai for ai in range(self.n_sensor, self.n_agents)
                             if self._connected_to_sensor(ai, t)]
            if len(members) > 1:
                idx = np.array(members)
                adj[np.ix_(idx, idx)] = 1.0
        np.fill_diagonal(adj, 1.0)
        return adj

    def _n_free_connected(self, t):
        """Free workers ASSIGNED to t and within comm range of sensor t.

        Assignment-scoped on purpose: this is the readiness number the
        sub-swarm's own observation reports, and letting a foreign worker
        inflate it would leak the other sub-swarm into the observation. The
        interference this env exists to measure lives in the attention graph
        (`comm_adjacency`), not here.
        """
        return sum(
            1 for ai in self.sub_swarm[t]
            if not self.loaded[self.worker_slot(ai)] and self._connected_to_sensor(ai, t)
        )

    def foreign_neighbours(self, ai):
        """Agents within comm_radius of `ai` that serve a DIFFERENT target.

        The raw material for the cross-interference metric: physical proximity,
        independent of whether it reached the policy.
        """
        mine = self.assigned[ai]
        return [aj for aj in range(self.n_agents)
                if aj != ai and self.assigned[aj] != mine
                and self._manhattan(self.pos[ai], self.pos[aj]) <= self.comm_radius]

    def _knows_K(self, ai):
        """True iff agent `ai` can currently see its OWN target's K.

        Same condition as the parent -- in comm range of the sensor, sensor on
        the target -- but scoped to the agent's assigned sensor, so standing
        next to a foreign sensor teaches a worker nothing.
        """
        t = self._target_of(ai)
        if self._resolved(t):
            return False
        return (self._connected_to_sensor(ai, t)
                and self._manhattan(self.pos[t], self.target_pos[t]) <= self.sense_radius)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _assign_workers(self):
        """Split the swarm, then size each crate to the sub-swarm that got it.

        Order matters. The sub-swarm sizes are drawn FIRST and K_t is drawn
        second, from U[k_min, n_t - 1] -- so each target's k_max is its own
        sub-swarm's size minus one, which is the relation the training config
        uses (5 workers, k_max 4) and the relation the observation depends on:
        field 1 is K_t / k_max_t and field 3 is n_free_connected_t / n_t, so
        both are fractions of the same sub-swarm and remain comparable. A single
        global k_max shared by sub-swarms of different sizes would break that,
        and the policy would fail for an encoding reason rather than a
        coordination one.

        n_t >= K_t + 1 follows for free, since K_t is capped at n_t - 1: every
        sub-swarm keeps at least one spare, so "everybody commits" is never
        trivially correct.
        """
        workers = np.array(range(self.n_sensor, self.n_agents))
        perm = self._rng.permutation(workers)
        floor = self.k_min + 1          # smallest sub-swarm that admits a valid K

        if self.assign_mode == "balanced":
            # Even split; WHICH worker goes where is still random. With an
            # evenly divisible swarm every sub-swarm is the size the policy
            # trained on, so the observation is bit-identical to training.
            groups = [np.asarray(g, dtype=int)
                      for g in np.array_split(perm, self.n_targets)]
        else:
            # Uneven split: every sub-swarm gets the floor, then the surplus is
            # scattered at random. Sizes come out lopsided (e.g. 3 and 7).
            sizes = [floor] * self.n_targets
            for _ in range(self.n_workers - floor * self.n_targets):
                sizes[int(self._rng.integers(self.n_targets))] += 1
            groups, cursor = [], 0
            for n_t in sizes:
                groups.append(np.asarray(perm[cursor:cursor + n_t], dtype=int))
                cursor += n_t

        self.sub_swarm = groups
        self.assigned = np.zeros(self.n_agents, dtype=int)
        for t in range(self.n_targets):
            self.assigned[t] = t                      # sensor t serves target t
            for ai in self.sub_swarm[t]:
                self.assigned[ai] = t

        # Per-sub-swarm k_max, then K_t. Both are recorded: the observation
        # normalises K_t by this target's own k_max_t, never by a global one.
        self.k_max_t = [len(self.sub_swarm[t]) - 1 for t in range(self.n_targets)]
        self.K = [int(self._rng.integers(self.k_min, self.k_max_t[t] + 1))
                  for t in range(self.n_targets)]

        for t in range(self.n_targets):
            n_t = len(self.sub_swarm[t])
            if n_t < floor:
                raise AssertionError(
                    f"sub-swarm {t} has {n_t} workers, below k_min + 1 = {floor}")
            if n_t < self.K[t] + 1:
                raise AssertionError(
                    f"assignment violated n_t >= K_t + 1 "
                    f"(target {t}: {n_t} < {self.K[t] + 1})")

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t = 0
        self.targets_lifted = 0
        self.loaded = [False] * self.n_workers
        self.lifted = [False] * self.n_targets
        self.failed = [False] * self.n_targets
        self._reset_diagnostics()

        # K is drawn per target inside _assign_workers, AFTER the sub-swarm
        # sizes are known, because each target's k_max is its own sub-swarm's
        # size minus one. Concurrent targets each have their own dedicated
        # workers, so the parent's sequential sum(K) <= n_workers budget does
        # not apply.
        occupied = set()
        self.pos = []
        for _ in range(self.n_agents):
            while True:
                x, y = int(self._rng.integers(self.G)), int(self._rng.integers(self.G))
                if (x, y) not in occupied:
                    occupied.add((x, y))
                    self.pos.append([x, y])
                    break

        target_set = set()
        self.target_pos = []
        for _ in range(self.n_targets):
            while True:
                x, y = int(self._rng.integers(self.G)), int(self._rng.integers(self.G))
                if (x, y) not in target_set:
                    target_set.add((x, y))
                    self.target_pos.append([x, y])
                    break

        self._assign_workers()
        return self.get_obs(), {}

    def _reset_diagnostics(self):
        T = getattr(self, "n_targets", 1)
        self._diag = {
            "assembled_steps": 0, "pickup_steps": 0, "correct_pickups": 0,
            "overshoot_pickups": 0, "undershoot_pickups": 0,
            "offtarget_pickups": 0, "premature_pickups": 0, "targets_lifted": 0,
            # Per-target views, so a failure can be attributed to one sub-swarm.
            "per_target_lifted": [0] * T,
            "per_target_assembled_steps": [0] * T,
            "per_target_outcome": ["unresolved"] * T,
        }

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _step_penalty(self):
        """Parent's distance-scaled time cost, averaged over unresolved teams.

        Each unresolved target contributes its own sensor plus its own free
        workers; resolved sub-swarms are spent and stop counting, exactly as
        loaded workers do in the parent.
        """
        dists = []
        for t in range(self.n_targets):
            if self._resolved(t):
                continue
            who = [t] + [ai for ai in self.sub_swarm[t]
                         if not self.loaded[self.worker_slot(ai)]]
            dists += [self._manhattan(self.pos[ai], self.target_pos[t]) for ai in who]
        if not dists:
            return self.REWARD_STEP_NEAR
        frac = min(float(np.mean(dists)) / (2 * (self.G - 1)), 1.0)
        return self.REWARD_STEP_NEAR + (self.REWARD_STEP_FAR - self.REWARD_STEP_NEAR) * frac

    def step(self, actions):
        a = [int(x) for x in actions]

        # 1. Movement — every agent toward ITS OWN target, via the parent's
        #    low-level controller. This is the whole of "MOVE drives the agent
        #    to the assigned target".
        for ai in range(self.n_agents):
            if a[ai] == self.MOVE:
                t = self._target_of(ai)
                if not self._resolved(t):
                    self._move_toward(ai, self.target_pos[t])

        # 2. Drift — workers only, skipped for workers attempting a pickup.
        for ai in range(self.n_sensor, self.n_agents):
            if a[ai] != self.PICKUP:
                self._apply_drift(ai)

        reward = self._step_penalty()
        d = self._diag

        # 3. Per-target pickup resolution. A pickup resolves only the target it
        #    was aimed at; the other sub-swarms keep going.
        for t in range(self.n_targets):
            if self._resolved(t):
                continue
            K = self.K[t]
            n_free_conn = self._n_free_connected(t)
            if n_free_conn >= K:
                d["assembled_steps"] += 1
                d["per_target_assembled_steps"][t] += 1

            pickers = [ai for ai in self.sub_swarm[t]
                       if a[ai] == self.PICKUP and not self.loaded[self.worker_slot(ai)]]
            if not pickers:
                continue

            d["pickup_steps"] += 1
            n_pickup = len(pickers)
            if n_free_conn < K:
                reward += self.REWARD_PREMATURE
                d["premature_pickups"] += 1
                outcome = "premature"
            elif n_pickup > K:
                reward += self.REWARD_OVERSHOOT
                d["overshoot_pickups"] += 1
                outcome = "overshoot"
            elif n_pickup < K:
                reward += self.REWARD_UNDERSHOOT
                d["undershoot_pickups"] += 1
                outcome = "undershoot"
            elif all(tuple(self.pos[ai]) == tuple(self.target_pos[t]) for ai in pickers):
                reward += self.REWARD_CORRECT_PICKUP
                d["correct_pickups"] += 1
                for ai in pickers:
                    self.loaded[self.worker_slot(ai)] = True
                self.lifted[t] = True
                self.targets_lifted += 1
                outcome = "lifted"
            else:
                reward += self.REWARD_UNDERSHOOT
                d["offtarget_pickups"] += 1
                outcome = "offtarget"

            d["per_target_outcome"][t] = outcome
            if outcome != "lifted":
                self.failed[t] = True

        if self.targets_lifted == self.n_targets:
            reward += self.REWARD_EPISODE_COMPLETION

        self._t += 1
        d["targets_lifted"] = self.targets_lifted
        d["per_target_lifted"] = [int(v) for v in self.lifted]

        terminated = all(self._resolved(t) for t in range(self.n_targets))
        truncated = (not terminated) and self._t >= self._time_limit

        info = dict(d)
        info["per_target_lifted"] = list(d["per_target_lifted"])
        info["per_target_assembled_steps"] = list(d["per_target_assembled_steps"])
        info["per_target_outcome"] = list(d["per_target_outcome"])
        info["K"] = list(self.K)
        info["sub_swarm_sizes"] = [len(g) for g in self.sub_swarm]
        info["simultaneous_success"] = float(self.targets_lifted == self.n_targets)
        return self.get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Observations — each agent sees only its own target
    # ------------------------------------------------------------------

    def get_obs_agent(self, agent_id):
        if self.pos is None:
            return np.zeros(self.obs_dim, dtype=np.float32)

        t = self._target_of(agent_id)
        K = self.K[t]
        # This target's own k_max: its sub-swarm size minus one. NOT a global
        # constant -- see _assign_workers on why the two must agree.
        k_max_t = max(self.k_max_t[t], 1)
        tgt = self.target_pos[t]
        n_t = max(len(self.sub_swarm[t]), 1)

        on_target = 1.0 if tuple(self.pos[agent_id]) == tuple(tgt) else 0.0
        sensor_on = self._manhattan(self.pos[t], tgt) <= self.sense_radius
        # A resolved target is finished business: its team reads "no target
        # remaining", the same value the single-target env shows after its lift.
        targets_remaining = 0.0 if self._resolved(t) else 1.0
        n_free_conn = self._n_free_connected(t) / n_t

        if self.is_sensor(agent_id):
            k_field = K / k_max_t if (sensor_on and not self._resolved(t)) else -1.0
            return np.array([on_target, k_field, -1.0, n_free_conn, targets_remaining],
                            dtype=np.float32)

        in_comm = self._connected_to_sensor(agent_id, t)
        k_field = (K / k_max_t
                   if (in_comm and sensor_on and not self._resolved(t)) else -1.0)
        is_loaded = 1.0 if self.loaded[self.worker_slot(agent_id)] else 0.0
        return np.array(
            [on_target, k_field, is_loaded, n_free_conn if in_comm else 0.0,
             targets_remaining], dtype=np.float32)

    # ------------------------------------------------------------------
    # State (logging only — the policy never reads it)
    # ------------------------------------------------------------------

    def get_state(self):
        if self.pos is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        state = []
        for ai in range(self.n_agents):
            state += [self.pos[ai][0] / self.G, self.pos[ai][1] / self.G]
        for t in range(self.n_targets):
            state += [self.target_pos[t][0] / self.G, self.target_pos[t][1] / self.G,
                      self.K[t] / max(self.k_max_t[t], 1),
                      1.0 if self._resolved(t) else 0.0]
        adj = self.comm_adjacency()
        state += [float(adj[ai].sum() > 1.0) for ai in range(self.n_agents)]
        return np.array(state, dtype=np.float32)

    # ------------------------------------------------------------------
    # Action availability
    # ------------------------------------------------------------------

    def _avail_for(self, ai):
        can_lift = (
            not self.is_sensor(ai)
            and not self.loaded[self.worker_slot(ai)]
            and self._knows_K(ai)
        )
        avail = [0] * self.n_actions
        avail[self.MOVE] = 1
        avail[self.WAIT] = 1
        avail[self.PICKUP] = 1 if can_lift else 0
        return avail

    def get_env_info(self):
        info = super().get_env_info()
        info["n_agents"] = self.n_agents
        info["state_shape"] = self.state_dim
        info["n_targets"] = self.n_targets
        return info

    def render(self):
        raise NotImplementedError(
            "the parent renderer draws one sensor and one active target; a "
            "multi-target view is not implemented.")

