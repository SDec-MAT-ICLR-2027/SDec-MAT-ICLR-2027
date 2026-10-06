"""
    Joint-capture navigation, built on the InforMARL graph-MPE base.

    Differences from `navigation_graph.py`:
      • Targets are SHARED between agents, not one private goal per agent.
      • A target is captured only when `k_req` agents are simultaneously
        inside `capture_radius` AND all emit the CAPTURE action on the same
        step. Fewer than `k_req` capturers is an undershoot and is taxed.

    The reward is deliberately split so that everything an individual agent
    can climb on its own is dense, and the only thing that requires
    correlated action is unshaped:

        navigation shaping : -dist to nearest active target  (dense, individual)
        assembly shaping   : +co-location progress to k_req   (dense, individual)
        capture            : +capture_rew, all-or-nothing     (JOINT, unshaped)
        undershoot         : +fail_rew for 1..k_req-1 capturers (JOINT, unshaped)

    That split is what removes the need for a failure-penalty curriculum: an
    agent can learn to approach and to gather from a standing gradient, while
    the simultaneous trigger stays a genuine joint decision that a product
    policy cannot express.

    `k_req` is the swept coordination knob. At k_req=1 the task is fully
    decomposable and decentralized execution should win or tie; as k_req grows
    the joint trigger becomes progressively unreachable without correlation.
"""
from typing import Optional, Tuple, List
import argparse
import numpy as np
from numpy import ndarray as arr
from scipy import sparse
import os, sys

sys.path.append(os.path.abspath(os.getcwd()))

from multiagent.core import World, Agent, Landmark, Entity
from multiagent.scenario import BaseScenario

entity_mapping = {"agent": 0, "landmark": 1, "obstacle": 2}

# index of the CAPTURE action within the extra (non-movement) actions
CAPTURE = 1


class Scenario(BaseScenario):
    # read by `GraphMPEEnv`; appends CAPTURE after the 5 movement actions
    n_extra_actions = 1

    def make_world(self, args: argparse.Namespace) -> World:
        """
        Parameters in args (in addition to the `navigation_graph.py` ones)
        ––––––––––––––––––
        • num_targets: int
            Number of shared targets. Unlike `navigation_graph.py` this is
            independent of `num_agents`.
        • k_req: int
            Agents that must capture simultaneously for a target to fall.
            THE coordination knob; sweep it to trace the DE/SDE boundary.
        • capture_radius: float
            Distance within which an agent may capture a target.
        • capture_rew: float
            Team reward when a target is successfully captured.
        • fail_rew: float
            Team reward (negative) when 1..k_req-1 agents capture a target.
        • assembly_rew: float
            Per-step dense shaping for being co-located at an active target.
        • dist_rew_scale: float
            Scale on the dense -distance-to-nearest-active-target shaping.
        • obs_ready_count: bool
            If True, append the number of agents currently in range of the
            agent's nearest active target to its local observation. This is
            the information control: it hands a decentralized baseline the
            count without handing it the ability to correlate actions.
        """
        # pull params from args
        self.world_size = args.world_size
        self.num_agents = args.num_agents
        self.num_scripted_agents = getattr(args, "num_scripted_agents", 0)
        self.num_obstacles = args.num_obstacles
        self.collaborative = args.collaborative
        self.max_speed = args.max_speed
        self.collision_rew = args.collision_rew
        self.use_dones = args.use_dones
        self.episode_length = args.episode_length

        # joint-capture params
        self.num_targets = int(getattr(args, "num_targets", args.num_agents))
        self.k_req = int(getattr(args, "k_req", 2))
        # Exact-k capture (E4). None = the original rule (>= k_req captures).
        # Otherwise more than k_max in-range capturers is an OVERKILL: no
        # capture and fail_rew, so a larger group must split into firers and
        # holders (anti-coordination, SMAC's overkill avoidance).
        k_max = getattr(args, "k_max", None)
        self.k_max = None if k_max is None else int(k_max)
        if self.k_max is not None and self.k_max < self.k_req:
            raise ValueError("k_max (%d) < k_req (%d)" % (self.k_max, self.k_req))
        self.capture_radius = float(getattr(args, "capture_radius", 0.3))
        self.capture_rew = float(getattr(args, "capture_rew", 10.0))
        self.fail_rew = float(getattr(args, "fail_rew", -1.0))
        self.assembly_rew = float(getattr(args, "assembly_rew", 0.05))
        self.dist_rew_scale = float(getattr(args, "dist_rew_scale", 1.0))
        # Per-step TEAM cost, charged every step until the episode ends. With
        # shaping telescoping and off-target CAPTURE being a free no-op, doing
        # nothing otherwise costs exactly zero, so "never fire" is a free
        # absorbing strategy (measured: DE control reaches READY 19-26x/ep and
        # fires 0.000-0.014 of the time, 0.00 captures). Pricing time makes the
        # only escape from the bleed clearing the targets. Aerial Ordering
        # (-0.1/step), Pickup Consensus (-0.3/step) and Team Partition
        # (step_cost 0.1) all charge for time; this env was the only one that
        # did not. 0.0 = off, which is every run made before this flag.
        self.step_cost = float(getattr(args, "step_cost", 0.0))
        # Potential-based shaping (Ng et al. 1999). MUST stay on for any real
        # run. With the naive form (potential_shaping=0) the navigation term
        # `-dist_to_nearest_ACTIVE_target` is paid as a per-step RATE, so
        # capturing the target you are standing on makes your nearest active
        # target jump >= min_target_sep away and your per-step reward collapse
        # for the rest of the episode. Measured on the 6a/3t/k3 cell, that made
        # capturing worth -37 versus camping: the reward punished task
        # completion, and both MAPPO and CT-SDE correctly learned never to
        # capture. See `probe_camp_vs_capture.py`.
        #
        # As a potential, F = gamma*Phi(s') - Phi(s) telescopes to a bounded
        # path-independent constant, so the one-time drop on capture is repaid
        # while walking to the next target, and the optimal policy of the
        # underlying MDP (+capture_rew per target, fail_rew per undershoot) is
        # provably unchanged.
        self.potential_shaping = bool(int(getattr(args, "potential_shaping", 1)))
        self.shaping_gamma = float(getattr(args, "shaping_gamma", 0.99))
        # Radius used to FORM joint-decision groups. Strictly >= capture_radius:
        # coupling starts during the approach, not only once agents are already
        # inside the trigger zone. With group_radius == capture_radius the
        # cliques are nearly always singletons early in training and the
        # semi-decentralized decoder has nothing to decode over.
        # Capture itself always uses capture_radius, never this.
        self.group_radius = float(
            getattr(args, "group_radius", 2.0 * self.capture_radius)
        )
        if self.group_radius < self.capture_radius:
            raise ValueError(
                f"group_radius ({self.group_radius}) < capture_radius "
                f"({self.capture_radius}): agents could be required to capture "
                f"together without ever being grouped"
            )
        self.obs_ready_count = bool(getattr(args, "obs_ready_count", False))
        # "local"  : own state + nearest active target only (no neighbour state)
        # "shared" : own state + every other agent's relative position + every
        #            target's relative position and active flag. This is the
        #            observation-sharing setting: a decentralized baseline sees
        #            everything CT-SDE sees, so any remaining gap is about how
        #            agents ACT, not about what they know.
        # "radius" : partial observability. An agent perceives other agents and
        #            targets only within `sense_radius`; hidden slots read zero
        #            with a visible=0 flag. Fixed slot layout so MAT tokens keep
        #            a constant size.
        self.obs_mode = str(getattr(args, "obs_mode", "local"))
        if self.obs_mode not in ("local", "shared", "radius"):
            raise ValueError(
                f"obs_mode must be 'local', 'shared' or 'radius', "
                f"got {self.obs_mode!r}"
            )
        self.sense_radius = float(getattr(args, "sense_radius", 1.0))
        # CENSUS-vs-TALLY experiment (both default to the previous behaviour):
        #   obs_radius   : how far an agent SEES OTHER AGENTS. Separate from
        #                  sense_radius, which still gates target visibility and
        #                  still builds the comm graph. Setting it below
        #                  2*capture_radius means two agents in one capture
        #                  group need not see each other, so no agent can take
        #                  a census of its own group.
        #   obs_abs_pos  : include the agent's ABSOLUTE own position in its
        #                  observation. With it removed every observation is
        #                  expressed in the agent's own frame, so pooling a
        #                  neighbour's observation vector no longer restores the
        #                  census that obs_radius removed.
        _or = getattr(args, "obs_radius", None)
        self.obs_radius = (self.sense_radius if _or is None or float(_or) <= 0.0
                           else float(_or))
        self.obs_abs_pos = bool(int(getattr(args, "obs_abs_pos", 1) or 0))
        # Remove target-id artefacts (both default off, so earlier runs are
        # unchanged):
        #   shuffle_targets: sequential rejection placement makes target 0
        #     central and later ids peripheral (large min_target_sep); a random
        #     permutation of the placed positions over ids removes that.
        #   target_order="distance": radius obs lists targets nearest-first
        #     (live visible, then visible captured, then hidden) instead of by
        #     id -- same local information, id-free layout.
        self.shuffle_targets = bool(int(getattr(args, "shuffle_targets", 0) or 0))
        self.target_order = str(getattr(args, "target_order", "id") or "id")
        if self.target_order not in ("id", "distance"):
            raise ValueError(
                f"target_order must be 'id' or 'distance', got {self.target_order!r}")
        # How agents are connected for semi-decentralized execution.
        #   target_clique : both within group_radius of the same live target
        #                   (uses task structure: knows what a target is)
        #   agent_radius  : within sense_radius of each other (pure distance,
        #                   no task knowledge; sensing and comm share one R)
        self.comm_mode = str(getattr(args, "comm_mode", "target_clique"))
        if self.comm_mode not in ("target_clique", "agent_radius"):
            raise ValueError(
                f"comm_mode must be 'target_clique' or 'agent_radius', "
                f"got {self.comm_mode!r}"
            )
        # Three agents inside one capture circle can be up to 2*capture_radius
        # apart. Below that, a ready group need not be connected and the joint
        # capture could be impossible to coordinate.
        if (self.comm_mode == "agent_radius"
                and self.sense_radius < 2.0 * self.capture_radius):
            raise ValueError(
                f"sense_radius ({self.sense_radius}) < 2*capture_radius "
                f"({2.0 * self.capture_radius}): a ready group could be "
                f"disconnected"
            )
        # Targets must be far enough apart that no single assembled group sits
        # inside two capture radii at once; otherwise one joint action fells
        # several targets and the capture count stops measuring coordination.
        _sep = getattr(args, "min_target_sep", None)
        self.min_target_sep = (
            float(_sep) if _sep is not None else 2.0 * self.group_radius
        )
        # kept for graph-feature parity with navigation_graph.py
        self.min_dist_thresh = float(getattr(args, "min_dist_thresh", 0.1))

        if self.k_req > self.num_agents:
            raise ValueError(
                f"k_req ({self.k_req}) > num_agents ({self.num_agents}): "
                f"no target can ever be captured"
            )

        if not hasattr(args, "max_edge_dist"):
            self.max_edge_dist = 1
            print("_" * 60)
            print(
                f"Max Edge Distance for graphs not specified. "
                f"Setting it to {self.max_edge_dist}"
            )
            print("_" * 60)
        else:
            self.max_edge_dist = args.max_edge_dist
        ####################
        world = World()
        # graph related attributes
        world.cache_dists = True
        world.graph_mode = True
        world.graph_feat_type = args.graph_feat_type
        world.world_length = args.episode_length
        world.current_time_step = 0
        world.dim_c = 2
        world.collaborative = args.collaborative

        # add agents
        global_id = 0
        world.agents = [Agent() for i in range(self.num_agents)]
        world.scripted_agents = [Agent() for _ in range(self.num_scripted_agents)]
        for i, agent in enumerate(world.agents + world.scripted_agents):
            agent.id = i
            agent.name = f"agent {i}"
            agent.collide = True
            agent.silent = True
            agent.global_id = global_id
            global_id += 1
            agent.max_speed = self.max_speed
        # add targets (shared landmarks)
        world.landmarks = [Landmark() for i in range(self.num_targets)]
        world.scripted_agents_goals = []
        for i, landmark in enumerate(world.landmarks):
            landmark.id = i
            landmark.name = f"landmark {i}"
            landmark.collide = False
            landmark.movable = False
            landmark.global_id = global_id
            global_id += 1
        # add obstacles
        world.obstacles = [Landmark() for i in range(self.num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = f"obstacle {i}"
            obstacle.collide = True
            obstacle.movable = False
            obstacle.global_id = global_id
            global_id += 1
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world: World) -> None:
        world.current_time_step = 0
        world.num_obstacle_collisions = np.zeros(self.num_agents)
        world.num_agent_collisions = np.zeros(self.num_agents)
        # per-target capture state
        world.target_active = np.ones(self.num_targets, dtype=bool)
        # episode tallies, surfaced through info_callback
        world.joint_captures = 0
        world.undershoots = 0
        world.overkills = 0
        world.capture_actions = 0
        # capture resolution cache, keyed by world.current_time_step so the
        # joint outcome is computed once per step and not once per agent
        world._resolved_step = -1
        world._team_event_rew = 0.0
        world._shaping_rew = 0.0
        # Phi(s_0); set properly at the end of reset_world once entities are
        # actually placed, since it depends on positions.
        world._prev_potential = 0.0

        #################### set colours ####################
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        for i, agent in enumerate(world.scripted_agents):
            agent.color = np.array([0.15, 0.15, 0.15])
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.85, 0.15])
        for i, obstacle in enumerate(world.obstacles):
            obstacle.color = np.array([0.25, 0.25, 0.25])
        #####################################################
        self.random_scenario(world)
        # entities are placed now, so Phi(s_0) is well defined
        world._prev_potential = self.potential(world)

    def random_scenario(self, world: World) -> None:
        """Randomly place obstacles, agents and targets without overlap."""
        for obstacle in world.obstacles:
            obstacle.state.p_pos = 0.8 * np.random.uniform(
                -self.world_size / 2, self.world_size / 2, world.dim_p
            )
            obstacle.state.p_vel = np.zeros(world.dim_p)

        # agents at random non-colliding positions
        num_agents_added = 0
        agents_added = []
        while num_agents_added < self.num_agents:
            random_pos = np.random.uniform(
                -self.world_size / 2, self.world_size / 2, world.dim_p
            )
            agent_size = world.agents[num_agents_added].size
            obs_collision = self.is_obstacle_collision(random_pos, agent_size, world)
            agent_collision = self.check_agent_collision(
                random_pos, agent_size, agents_added
            )
            if not obs_collision and not agent_collision:
                world.agents[num_agents_added].state.p_pos = random_pos
                world.agents[num_agents_added].state.p_vel = np.zeros(world.dim_p)
                world.agents[num_agents_added].state.c = np.zeros(world.dim_c)
                agents_added.append(world.agents[num_agents_added])
                num_agents_added += 1

        # targets at random positions, kept `min_target_sep` apart so that a
        # single group of agents can never be in range of two at once
        num_goals_added = 0
        attempts = 0
        sep = self.min_target_sep
        while num_goals_added < self.num_targets:
            attempts += 1
            if attempts > 2000:
                # board too crowded for this separation; relax rather than hang
                sep *= 0.9
                attempts = 0
                if sep < 1e-3:
                    raise RuntimeError(
                        f"cannot place {self.num_targets} targets in a "
                        f"{self.world_size} world; lower num_targets or "
                        f"capture_radius"
                    )
            random_pos = 0.8 * np.random.uniform(
                -self.world_size / 2, self.world_size / 2, world.dim_p
            )
            goal_size = world.landmarks[num_goals_added].size
            if self.is_obstacle_collision(random_pos, goal_size, world):
                continue
            placed = [lm.state.p_pos for lm in world.landmarks[:num_goals_added]]
            if any(
                np.sqrt(np.sum(np.square(p - random_pos))) < sep for p in placed
            ):
                continue
            world.landmarks[num_goals_added].state.p_pos = random_pos
            world.landmarks[num_goals_added].state.p_vel = np.zeros(world.dim_p)
            num_goals_added += 1
            attempts = 0

        if self.shuffle_targets and self.num_targets > 1:
            perm = np.random.permutation(self.num_targets)
            placed = [world.landmarks[t].state.p_pos.copy()
                      for t in range(self.num_targets)]
            for t in range(self.num_targets):
                world.landmarks[t].state.p_pos = placed[perm[t]]

        world.calculate_distances()
        self.update_graph(world)

    ###################################################################
    # joint capture resolution
    ###################################################################
    def _agents_within(self, world: World, target_idx: int, radius: float) -> arr:
        """Boolean mask over agents inside `radius` of a target."""
        target_pos = world.landmarks[target_idx].state.p_pos
        d = np.array(
            [
                np.sqrt(np.sum(np.square(a.state.p_pos - target_pos)))
                for a in world.agents
            ]
        )
        return d <= radius

    def _agents_in_range(self, world: World, target_idx: int) -> arr:
        """Agents close enough to CAPTURE a target."""
        return self._agents_within(world, target_idx, self.capture_radius)

    def target_clique_adjacency(self, world: World) -> arr:
        """
        N x N group adjacency for semi-decentralized execution.

        Agents i and j are grouped iff they are both inside `group_radius`
        of the SAME still-active target, i.e. iff they are converging on the
        same joint decision. This is the direct analogue of the same-prey
        catcher cliques used on maco_pursuit. Diagonal is always on, so a
        lone agent forms a singleton group and simply decodes by itself.

        Note this uses `group_radius`, not `capture_radius`: agents need to
        be coordinating on the approach, not only once they are already in
        the trigger zone.

        Proximity is this task's coupling criterion, not a general claim:
        the gate marks the states where actions are actually coupled.
        """
        n = len(world.agents)
        adj = np.eye(n, dtype=np.int64)
        for t in range(self.num_targets):
            if not world.target_active[t]:
                continue
            in_range = self._agents_within(world, t, self.group_radius)
            idx = np.nonzero(in_range)[0]
            for a in idx:
                for b in idx:
                    adj[a, b] = 1
        return adj

    def _nearest_active_target(self, agent: Agent, world: World) -> Tuple[int, float]:
        """Index of and distance to the nearest still-active target (-1 if none)."""
        best_idx, best_d = -1, np.inf
        for t in range(self.num_targets):
            if not world.target_active[t]:
                continue
            d = np.sqrt(
                np.sum(
                    np.square(agent.state.p_pos - world.landmarks[t].state.p_pos)
                )
            )
            if d < best_d:
                best_idx, best_d = t, d
        return best_idx, best_d

    def potential(self, world: World) -> float:
        """
        Phi(s), the team-level shaping potential. Higher is better, bounded.

        Two components, both of which an individual agent can climb on its own:
          - approach : -dist_rew_scale * sum_i dist(i, nearest active target)
          - assembly : +assembly_rew * co-location progress toward k_req,
                       summed over still-active targets

        Once every target is captured Phi = 0, which is its maximum, so
        clearing the board is the top of the potential rather than a cliff.
        """
        phi = 0.0
        for agent in world.agents:
            _, d = self._nearest_active_target(agent, world)
            if np.isfinite(d):
                phi -= self.dist_rew_scale * d
        for t in range(self.num_targets):
            if not world.target_active[t]:
                continue
            n_near = int(self._agents_in_range(world, t).sum())
            phi += self.assembly_rew * min(n_near, self.k_req) / self.k_req
        return float(phi)

    def _resolve_captures(self, world: World) -> None:
        """
        Resolve every target's joint capture exactly once per env step.

        Called from `reward()`; guarded on `world.current_time_step` so that
        the N per-agent reward calls in `MultiAgentGraphEnv.step` all observe
        the same joint outcome instead of re-resolving it N times.
        """
        if world._resolved_step == world.current_time_step:
            return
        world._resolved_step = world.current_time_step

        capturing = np.array(
            [getattr(a.action, "extra", 0) == CAPTURE for a in world.agents]
        )
        world.capture_actions += int(capturing.sum())

        team_rew = 0.0
        for t in range(self.num_targets):
            if not world.target_active[t]:
                continue
            in_range = self._agents_in_range(world, t)
            n_capturing = int(np.sum(in_range & capturing))
            if self.k_max is not None and n_capturing > self.k_max:
                # overkill: too many fired at once, no capture
                world.overkills += 1
                team_rew += self.fail_rew
            elif n_capturing >= self.k_req:
                world.target_active[t] = False
                world.joint_captures += 1
                team_rew += self.capture_rew
            elif n_capturing > 0:
                # undershoot: the joint action was attempted but not met
                world.undershoots += 1
                team_rew += self.fail_rew
        world._team_event_rew = team_rew

        # Shaping is evaluated on the RESULTING state s', so it must come
        # after captures have been applied above.
        if self.potential_shaping:
            phi_next = self.potential(world)
            world._shaping_rew = (
                self.shaping_gamma * phi_next - world._prev_potential
            )
            world._prev_potential = phi_next
        else:
            world._shaping_rew = 0.0

    ###################################################################
    # rewards
    ###################################################################
    def reward(self, agent: Agent, world: World) -> float:
        """
        Per-agent reward. `MultiAgentGraphEnv.step` SUMS these and then shares
        the sum, so the team-level capture outcome is divided by num_agents
        here to make the sum come out to exactly the intended team reward.
        """
        self._resolve_captures(world)

        rew = 0.0
        if self.potential_shaping:
            # --- team-level potential difference, split so the sum is exact.
            # Dense and individually climbable (approaching or joining a group
            # both raise Phi), but it cannot be farmed: standing still earns
            # zero, so camping on a target is worth nothing.
            rew += world._shaping_rew / float(self.num_agents)
        else:
            # --- legacy per-step rate. Kept only to reproduce the camping
            # pathology; see the note in make_world. Do not use for real runs.
            t_idx, t_dist = self._nearest_active_target(agent, world)
            if t_idx >= 0:
                rew -= self.dist_rew_scale * t_dist
                in_range = self._agents_in_range(world, t_idx)
                if in_range[agent.id]:
                    n_near = int(in_range.sum())
                    rew += (
                        self.assembly_rew * min(n_near, self.k_req) / self.k_req
                    )

        # --- collisions
        if agent.collide:
            for a in world.agents:
                if a.id == agent.id:
                    continue
                if self.is_collision(a, agent):
                    rew -= self.collision_rew
            if self.is_obstacle_collision(
                pos=agent.state.p_pos, entity_size=agent.size, world=world
            ):
                rew -= self.collision_rew

        # --- per-step team cost, split so the team sum is exactly step_cost
        rew -= self.step_cost / float(self.num_agents)

        # --- unshaped joint outcome, split so the team sum is exact
        rew += world._team_event_rew / float(self.num_agents)
        return rew

    ###################################################################
    # observations
    ###################################################################
    def observation(self, agent: Agent, world: World) -> arr:
        """
        Local observation.

        obs_mode="local" (dim 8, + 1 if obs_ready_count):
            [vel(2), pos(2), rel_nearest_active_target(2),
             in_capture_range(1), any_target_left(1)]
            No neighbour state at all. InforMARL gets neighbours through the
            graph; CT-SDE gets them through its group channel.

        obs_mode="shared" (dim 8 + 2*(N-1) + 3*M, + 1 if obs_ready_count):
            the above, plus every other agent's relative position and every
            target's relative position and active flag. Every arm then sees
            the same world and only the execution scheme differs.
        """
        if self.obs_mode == "radius":
            return self._radius_observation(agent, world)
        return self._vector_observation(agent, world, self.obs_mode == "shared")

    def state_observation(self, agent: Agent, world: World) -> arr:
        """Agent-centred FULL view for the centralized critic, never the
        actor: the shared-mode observation whatever obs_mode is. In shared
        mode it equals observation() exactly."""
        return self._vector_observation(agent, world, True)

    def _vector_observation(self, agent: Agent, world: World, shared: bool) -> arr:
        t_idx, t_dist = self._nearest_active_target(agent, world)
        if t_idx >= 0:
            rel_target = world.landmarks[t_idx].state.p_pos - agent.state.p_pos
            in_range = float(t_dist <= self.capture_radius)
            any_left = 1.0
        else:
            rel_target = np.zeros(world.dim_p)
            in_range = 0.0
            any_left = 0.0

        feats = [agent.state.p_vel, agent.state.p_pos, rel_target,
                 np.array([in_range, any_left])]

        if shared:
            # every other agent's relative position, in fixed id order
            for other in world.agents:
                if other.id == agent.id:
                    continue
                feats.append(other.state.p_pos - agent.state.p_pos)
            # every target's relative position plus whether it is still live
            for t in range(self.num_targets):
                feats.append(world.landmarks[t].state.p_pos - agent.state.p_pos)
                feats.append(np.array([float(world.target_active[t])]))

        if self.obs_ready_count:
            # information control: the count, without the ability to correlate
            if t_idx >= 0:
                n_near = float(np.sum(self._agents_in_range(world, t_idx)))
            else:
                n_near = 0.0
            feats.append(np.array([n_near]))

        return np.concatenate(feats)

    def _radius_observation(self, agent: Agent, world: World) -> arr:
        """
        obs_mode="radius", dim (4 if obs_abs_pos else 2) + 3*(N-1) + 4*M + 1:
            [vel(2), pos(2) -- pos omitted when obs_abs_pos=0,
             per other agent (id order):  rel_pos(2), visible(1),
             per target:                  rel_pos(2), alive(1), visible(1),
             in_capture_range(1)]
        Other AGENTS beyond obs_radius and TARGETS beyond sense_radius read as
        zeros with visible=0, and a hidden target's alive flag is also 0 (the
        agent cannot know it). obs_radius defaults to sense_radius, which
        reproduces every run made before the flag existed.
        There is no "nearest target" feature: that would leak global state.
        """
        R = self.obs_radius          # agent-to-AGENT visibility (the census)
        R_tgt = self.sense_radius    # agent-to-TARGET visibility (navigation)
        pos = agent.state.p_pos
        feats = [agent.state.p_vel, pos] if self.obs_abs_pos else [agent.state.p_vel]
        for other in world.agents:
            if other.id == agent.id:
                continue
            rel = other.state.p_pos - pos
            vis = float(np.sqrt(np.sum(np.square(rel))) <= R)
            feats.append(rel * vis)
            feats.append(np.array([vis]))
        in_cap = 0.0
        rows = []
        for t in range(self.num_targets):
            rel = world.landmarks[t].state.p_pos - pos
            d = float(np.sqrt(np.sum(np.square(rel))))
            vis = float(d <= R_tgt)
            alive = float(world.target_active[t]) * vis
            rows.append((rel * vis, alive, vis, d))
            if world.target_active[t] and d <= self.capture_radius:
                in_cap = 1.0
        if self.target_order == "distance":
            # live visible first (nearest first), then visible captured, then
            # hidden rows (all zeros, so their order carries nothing)
            rows.sort(key=lambda r: (-r[2], -r[1], r[3] if r[2] else 0.0))
        for rel_v, alive, vis, _d in rows:
            feats.append(rel_v)
            feats.append(np.array([alive, vis]))
        feats.append(np.array([in_cap]))
        return np.concatenate(feats)

    def agent_radius_adjacency(self, world: World) -> arr:
        """N x N: agents i, j connected iff within sense_radius (diagonal on)."""
        P = np.array([a.state.p_pos for a in world.agents])
        d = np.sqrt(np.sum(np.square(P[:, None, :] - P[None, :, :]), axis=-1))
        adj = (d <= self.sense_radius).astype(np.int64)
        np.fill_diagonal(adj, 1)
        return adj

    def comm_adjacency(self, world: World) -> arr:
        """The group adjacency selected by comm_mode."""
        if self.comm_mode == "agent_radius":
            return self.agent_radius_adjacency(world)
        return self.target_clique_adjacency(world)

    def get_id(self, agent: Agent) -> arr:
        return np.array([agent.global_id])

    def graph_observation(self, agent: Agent, world: World) -> Tuple[arr, arr]:
        """
        Node features and adjacency, same contract as `navigation_graph.py`
        but with one extra `active` feature per node so the GNN can tell a
        captured target from a live one.
        """
        node_obs = []
        if world.graph_feat_type == "global":
            for entity in world.entities:
                node_obs.append(self._get_entity_feat_global(entity, world))
        elif world.graph_feat_type == "relative":
            for entity in world.entities:
                node_obs.append(self._get_entity_feat_relative(agent, entity, world))

        node_obs = np.array(node_obs)
        adj = world.cached_dist_mag
        return node_obs, adj

    def update_graph(self, world: World) -> None:
        dists = world.cached_dist_mag
        connect = np.array((dists <= self.max_edge_dist) * (dists > 0)).astype(int)
        sparse_connect = sparse.csr_matrix(connect)
        sparse_connect = sparse_connect.tocoo()
        row, col = sparse_connect.row, sparse_connect.col
        world.edge_list = np.stack([row, col])
        world.edge_weight = dists[row, col]

    def _goal_and_active(self, entity: Entity, world: World) -> Tuple[arr, float]:
        """
        The `goal` slot of a node feature, plus whether the node is live.

        For an agent the goal is its nearest active target (there are no
        private goals here). For a target it is its own position, and
        `active` flags whether it has already been captured.
        """
        if "agent" in entity.name:
            t_idx, _ = self._nearest_active_target(entity, world)
            goal_pos = (
                world.landmarks[t_idx].state.p_pos
                if t_idx >= 0
                else entity.state.p_pos
            )
            return goal_pos, 1.0
        if "landmark" in entity.name:
            return entity.state.p_pos, float(world.target_active[entity.id])
        if "obstacle" in entity.name:
            return entity.state.p_pos, 1.0
        raise ValueError(f"{entity.name} not supported")

    def _get_entity_feat_global(self, entity: Entity, world: World) -> arr:
        """Returns ([vel, pos, goal_pos, entity_type, active]) in global coords."""
        pos = entity.state.p_pos
        vel = entity.state.p_vel
        goal_pos, active = self._goal_and_active(entity, world)
        entity_type = entity_mapping[self._entity_kind(entity)]
        return np.hstack([vel, pos, goal_pos, entity_type, active])

    def _get_entity_feat_relative(
        self, agent: Agent, entity: Entity, world: World
    ) -> arr:
        """Returns ([rel_vel, rel_pos, rel_goal_pos, entity_type, active])."""
        rel_pos = entity.state.p_pos - agent.state.p_pos
        rel_vel = entity.state.p_vel - agent.state.p_vel
        goal_pos, active = self._goal_and_active(entity, world)
        rel_goal_pos = goal_pos - agent.state.p_pos
        entity_type = entity_mapping[self._entity_kind(entity)]
        return np.hstack([rel_vel, rel_pos, rel_goal_pos, entity_type, active])

    @staticmethod
    def _entity_kind(entity: Entity) -> str:
        for kind in ("agent", "landmark", "obstacle"):
            if kind in entity.name:
                return kind
        raise ValueError(f"{entity.name} not supported")

    ###################################################################
    # collisions / done / info
    ###################################################################
    def is_obstacle_collision(self, pos, entity_size: float, world: World) -> bool:
        for obstacle in world.obstacles:
            delta_pos = obstacle.state.p_pos - pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            if dist < obstacle.size + entity_size:
                return True
        return False

    def check_agent_collision(self, pos, agent_size, agent_added: List) -> bool:
        for agent in agent_added:
            delta_pos = agent.state.p_pos - pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            if dist < agent.size + agent_size:
                return True
        return False

    def is_collision(self, agent1: Agent, agent2: Agent) -> bool:
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        return dist < (agent1.size + agent2.size)

    def is_landmark_collision(self, pos, size: float, landmark_list: List) -> bool:
        for landmark in landmark_list:
            delta_pos = landmark.state.p_pos - pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            if dist < size + landmark.size:
                return True
        return False

    def done(self, agent: Agent, world: World) -> bool:
        # episode ends when every target is down, or the horizon is reached
        if self.use_dones and not world.target_active.any():
            return True
        return world.current_time_step >= world.world_length

    def info_callback(self, agent: Agent, world: World) -> dict:
        if agent.collide:
            if self.is_obstacle_collision(agent.state.p_pos, agent.size, world):
                world.num_obstacle_collisions[agent.id] += 1
            for a in world.agents:
                if a is agent:
                    continue
                if self.is_collision(agent, a):
                    world.num_agent_collisions[agent.id] += 1

        _, t_dist = self._nearest_active_target(agent, world)
        return {
            # the two headline metrics: joint captures vs undershoot attempts
            "joint_captures": world.joint_captures,
            "undershoots": world.undershoots,
            "overkills": world.overkills,
            "capture_actions": world.capture_actions,
            "targets_left": int(world.target_active.sum()),
            "all_captured": bool(not world.target_active.any()),
            "Dist_to_target": float(t_dist) if np.isfinite(t_dist) else 0.0,
            "Num_agent_collisions": world.num_agent_collisions[agent.id],
            "Num_obst_collisions": world.num_obstacle_collisions[agent.id],
        }
