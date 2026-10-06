"""Pickup Consensus: the bare exact-K counting problem.

`n_agents` identical drones hover on one box that needs EXACTLY K of them to
lift it in the same step. K is drawn uniformly from [k_min, k_max] every episode
and is told to every agent. There is no navigation, no hidden information and
no communication limit -- the only question is WHICH K of the N agents commit.
It is Aerial Ordering with everything incidental stripped away (grid, sensor,
comm gate, wind), so a gap between methods can only come from how they break
the symmetry between agents.

Observation (2 floats, identical for every agent)
    [K / n_agents, n_agents / n_max]
State
    identical to the observation (there is nothing privileged to add).
Actions
    0 WAIT, 1 PICKUP -- both always available.
Reward (shared)
    exactly K PICKUPs              +10.0, episode ends (success)
    any other nonzero count        -10.0, episode ends (over/undershoot)
    no PICKUP                       episode continues
    every step                      -0.3;  truncated at time_limit
Info
    success (1.0 iff exactly K committed), overshoot, undershoot, no_attempt
    (truncated without any PICKUP), n_pickup, K.

Why it is hard for parameter sharing
    Every agent sees the same observation, so a shared policy without agent ids
    gives every agent the same commit probability p, and the number of PICKUPs
    is Binomial(N, p). P(exactly K) is maximised at p = K/N and is far below 1
    (`symmetric_ceiling`, `uniform_k_ceiling`). A method that breaks the tie at
    decode time -- SDec-MAT's autoregressive decoder -- is not bound by it.
    Appending a one-hot agent id also removes the barrier ("lift iff my index
    < K"); the paper reports CTDE baselines with and without ids for that
    reason.

Paper configuration (mac/presets.py): N=5, K in {2,3,4}, time_limit=10.
Rule-based levels there: perfect coordination +9.70 (success 1.0); every agent
committing independently with p = K/N -2.80 (success 0.375). Keep k_min >= 2:
with K = 1 a symmetric policy can exploit "nobody committed" as a free re-roll.
"""

import math
from functools import lru_cache

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@lru_cache(maxsize=None)
def symmetric_ceiling(n_agents, k, time_limit=10, reward_correct=10.0,
                      reward_wrong=-20.0, reward_step=-0.3, n_grid=4000):
    """Best value attainable by a symmetric policy that COMMITS (p > 0).

    Such a policy emits one Bernoulli(p) per agent, so per step the number of
    committers is Binomial(n_agents, p): it hits with P(K), fails terminally
    with 1 - P(0) - P(K), and re-rolls with P(0). Backward induction over
    `time_limit` steps gives the exact value, maximised over p by grid search.

    NOTE: the grid spans [1e-3, 1-1e-3] and therefore EXCLUDES p = 0. That is
    deliberate — p = 0 is the ALL-WAIT policy, whose value is exactly
    time_limit * reward_step and which is handled separately by
    `freeze_value()`. At the shipped constants ALL-WAIT is the true optimum, so
    this function returns the best *committing* alternative, i.e. the value the
    policy gives up by not freezing. Compare the two with `freeze_dominates()`.

    Returns (best_p, expected_return, success_probability).
    """
    best = (0.0, -math.inf, 0.0)
    for p in np.linspace(1e-3, 1 - 1e-3, n_grid):
        probs = [math.comb(n_agents, i) * p ** i * (1 - p) ** (n_agents - i)
                 for i in range(n_agents + 1)]
        p_hit, p_zero = probs[k], probs[0]
        p_bad = 1.0 - p_hit - p_zero
        value, success = 0.0, 0.0
        for _ in range(time_limit):  # backward induction from the horizon
            value = reward_step + p_hit * reward_correct + p_bad * reward_wrong + p_zero * value
            success = p_hit + p_zero * success
        if value > best[1]:
            best = (float(p), float(value), float(success))
    return best


class PickupConsensusEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # ---- action ids ----
    WAIT   = 0
    PICKUP = 1

    # ---- reward constants ----
    REWARD_CORRECT_PICKUP = 10.0   # exactly K committed
    REWARD_WRONG_COUNT    = -10.0  # any other nonzero count (over- or undershoot)
    REWARD_STEP           = -0.3   # time pressure; makes stalling cost 10*-0.3

    def __init__(
        self,
        n_agents=5,
        k_min=1,
        k_max=None,
        n_max=8,
        time_limit=10,
        seed=None,
        render_flag=False,
        **kwargs,
    ):
        super().__init__()

        self.n_agents = int(n_agents)
        self.k_min    = int(k_min)
        # Default k_max is n_agents-1: K == n_agents is solved by "everybody
        # commits" and would contribute free success to every method.
        self.k_max    = int(self.n_agents - 1 if k_max is None else k_max)
        self.n_max    = int(n_max)

        self._time_limit = int(time_limit)
        self.render_flag = bool(render_flag)
        self._rng        = np.random.default_rng(seed)

        # --- rendering options (see render()) -------------------------------
        # Read out of **kwargs so an env_args block can set them, and plain
        # attributes so a driver script can flip them on a live env.
        self.render_theme    = kwargs.get("render_theme", "paper")
        self.render_save_dir = kwargs.get("render_save_dir", None)
        self.render_dpi      = int(kwargs.get("render_dpi", 150))
        self.render_pause    = float(kwargs.get("render_pause", 0.35))
        self.render_prefix   = kwargs.get("render_prefix", "pickup_consensus")
        # Ground: procedural dust/scrub terrain. `render_terrain=False`
        # falls back to the flat background, and the contrast knob dials
        # the texture down if it competes with the foreground in print.
        self.render_terrain = bool(kwargs.get("render_terrain", True))
        self.render_terrain_contrast = float(
            kwargs.get("render_terrain_contrast", 0.72))
        # Ground texture is procedural; the seed picks which patch of field
        # this benchmark sits on, so the figures do not share a texture.
        self.render_terrain_seed = int(kwargs.get("render_terrain_seed", 21))
        self.render_note     = kwargs.get("render_note", "")
        self._fig            = None
        self._frame_paths    = []

        if self.k_min < 1:
            raise ValueError(f"k_min must be >= 1, got {self.k_min}.")
        if self.n_agents < self.k_min + 1:
            raise ValueError(
                f"n_agents must be >= k_min+1 ({self.k_min + 1}), so that "
                f"k_max=n_agents-1 >= k_min; got n_agents={self.n_agents}."
            )
        if self.k_max > self.n_agents - 1:
            raise ValueError(
                f"k_max must be <= n_agents-1 ({self.n_agents - 1}), got {self.k_max}. "
                f"K == n_agents is trivially solved by 'everybody commits'."
            )
        if self.k_min > self.k_max:
            raise ValueError(f"k_min({self.k_min}) > k_max({self.k_max}).")
        if self.n_agents > self.n_max:
            raise ValueError(
                f"n_agents({self.n_agents}) > n_max({self.n_max}); raise n_max so the "
                f"team_size observation field stays normalised to [0, 1]."
            )

        self.obs_dim   = 2
        self.n_actions = 2
        self.state_dim = 2

        self.action_space = spaces.Tuple(
            tuple(spaces.Discrete(self.n_actions) for _ in range(self.n_agents))
        )
        self.observation_space = spaces.Tuple(
            tuple(
                spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
                for _ in range(self.n_agents)
            )
        )

        # Episode state
        self.K          = None   # this episode's required committer count
        self._t         = 0
        self._last_n    = 0      # pickups on the previous step (diagnostics only)
        self._succeeded = False

        # Render-only bookkeeping: WHICH agents committed (not just how many)
        # and how the attempt resolved. Never read by obs/state/step logic.
        self._last_actions = None
        self._last_event   = None   # "success" | "overshoot" | "undershoot"
        self._episode_idx  = -1
        self._frame_idx    = 0

        ceil_p, ceil_r, ceil_s = self.uniform_k_ceiling()
        frz = self.freeze_value()
        verdict = ("ALL-WAIT is OPTIMAL" if self.freeze_dominates()
                   else f"!! committing BEATS freezing by {ceil_r - frz:+.2f} !!")
        print(
            f"PickupConsensusEnv (Pickup Consensus): "
            f"n_agents={self.n_agents}  k=[{self.k_min},{self.k_max}]  "
            f"time_limit={self._time_limit}  "
            f"obs_dim={self.obs_dim}  n_actions={self.n_actions}  "
            f"state_dim={self.state_dim}\n"
            f"  perfect={self.REWARD_CORRECT_PICKUP + self.REWARD_STEP:+.2f}  |  "
            f"ALL-WAIT={frz:+.2f}  |  best committing p>0: "
            f"return={ceil_r:+.2f} success={ceil_s:.3f}  ->  {verdict}"
        )

    # ------------------------------------------------------------------
    # Analytic reference values
    # ------------------------------------------------------------------

    def freeze_value(self):
        """Exact value of the ALL-WAIT policy: never commit, truncate at the
        horizon. At the shipped constants this is the OPTIMAL symmetric policy,
        and therefore the line to draw on a baseline learning curve."""
        return float(self.REWARD_STEP * self._time_limit)

    def freeze_dominates(self):
        """True iff ALL-WAIT beats every commit rate p > 0, i.e. the benchmark is
        in the regime it is designed for. False means a symmetric policy has a
        genuine strategy available (most often because k_min == 1) and the
        'baselines converge to a no-op equilibrium' claim does not hold."""
        return self.uniform_k_ceiling()[1] < self.freeze_value()

    def uniform_k_ceiling(self):
        """Best COMMITTING (p > 0) symmetric policy, averaged over
        K ~ Uniform[k_min, k_max]. See symmetric_ceiling() on why p = 0 is
        excluded; use freeze_value() for the ALL-WAIT policy.

        Returns (mean_best_p, mean_return, mean_success).
        """
        rows = [
            symmetric_ceiling(self.n_agents, k, self._time_limit,
                              self.REWARD_CORRECT_PICKUP, self.REWARD_WRONG_COUNT,
                              self.REWARD_STEP)
            for k in range(self.k_min, self.k_max + 1)
        ]
        return tuple(float(np.mean([r[i] for r in rows])) for i in range(3))

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t         = 0
        self._last_n    = 0
        self._succeeded = False
        self.K          = int(self._rng.integers(self.k_min, self.k_max + 1))

        self._last_actions = None
        self._last_event   = None
        self._episode_idx += 1
        self._frame_idx    = 0

        if self.render_flag:
            self.render()

        return self.get_obs(), {}

    def step(self, actions):
        a = [int(x) for x in actions]
        n_pickup = sum(1 for x in a if x == self.PICKUP)

        self._last_actions = list(a)   # render-only
        self._last_event   = None

        reward     = self.REWARD_STEP
        terminated = False

        if n_pickup > 0:
            # Any commitment resolves the episode one way or the other. Zero
            # pickups is a non-event so that stalling costs only time, never a
            # penalty spike (see module docstring on why K>=2 keeps this safe).
            if n_pickup == self.K:
                reward         += self.REWARD_CORRECT_PICKUP
                self._succeeded = True
                self._last_event = "success"
            else:
                reward += self.REWARD_WRONG_COUNT
                self._last_event = ("overshoot" if n_pickup > self.K
                                    else "undershoot")
            terminated = True

        self._t      += 1
        self._last_n  = n_pickup
        truncated     = (not terminated) and self._t >= self._time_limit

        info = {
            "n_pickup":   n_pickup,
            "K":          self.K,
            "success":    float(self._succeeded),
            "overshoot":  float(n_pickup > self.K),
            "undershoot": float(0 < n_pickup < self.K),
            "no_attempt": float(truncated),
        }

        if self.render_flag:
            self.render()

        return self.get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Observations / state
    # ------------------------------------------------------------------

    def get_obs(self):
        return tuple(self.get_obs_agent(ai) for ai in range(self.n_agents))

    def get_obs_agent(self, agent_id):
        if self.K is None:
            return np.zeros(self.obs_dim, dtype=np.float32)
        # Every agent gets the SAME vector — that is the entire point. Adding
        # any per-agent field here (index, noise, last action) breaks the
        # benchmark by handing a shared policy a symmetry-breaking channel.
        return np.array(
            [self.K / self.n_agents, self.n_agents / self.n_max],
            dtype=np.float32,
        )

    def get_state(self):
        if self.K is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        return np.array(
            [self.K / self.n_agents, self.n_agents / self.n_max],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Action availability / comm
    # ------------------------------------------------------------------

    def get_avail_actions(self):
        return [self._avail_for(ai) for ai in range(self.n_agents)]

    def _avail_for(self, ai):
        # Both actions are always legal for everyone; no masking anywhere, since
        # masking is itself a form of asymmetry.
        return [1] * self.n_actions

    def get_comm_mask(self):
        # Fully connected: there is no range or topology in this env.
        return np.ones((self.n_agents, 1), dtype=np.float32)

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

    def render(self):
        """Figure-quality view of the consensus moment.

        This benchmark has no geometry — the whole task is "exactly K of N
        identical drones must commit" — so the scene is composed to show the
        two facts that decide the episode:

          * every drone hangs from the SAME crate on a slack tether and carries
            the SAME observation chip, which is drawn explicitly: the agents are
            bit-identical, so no drone can tell itself apart from the others.
            That is the reason a symmetric shared-parameter policy is stuck at
            the binomial ceiling, and the picture should say so;
          * a drone that commits pulls its tether TAUT and rises. Counting taut
            lines against the crate's `EXACTLY K` placard is the entire task,
            and the verdict badge names the outcome.

        Configure via attributes (also accepted as env_args): render_theme,
        render_save_dir, render_dpi, render_pause, render_prefix, render_note.
        """
        import numpy as _np

        dr = self._render_kit()

        fig = getattr(self, "_fig", None)
        if fig is None:
            fig, ax, th = dr.stage(
                figsize=(9.0, 8.2),
                theme_name=self.render_theme,
                save_dir=self.render_save_dir,
                window_title="Pickup Consensus",
            )
            self._fig, self._ax = fig, ax
        else:
            ax = self._ax
            th = dr.reuse(fig, ax, self.render_theme)
        fig.texts.clear()
        fig.patches.clear()

        W, H = 10.0, 8.0
        # Elevation view (drones hover ABOVE the crate), so the ground only
        # occupies the bottom of the panel and the rest is sky.
        dr.sky_field(ax, 0, W, 0, H, th, horizon=2.45 - 2.15 / 2.0, pad=0.25,
                     terrain=self.render_terrain,
                     terrain_seed=self.render_terrain_seed,
                     terrain_contrast=self.render_terrain_contrast)

        K = self.K if self.K is not None else self.k_min
        acts = self._last_actions
        committed = [i for i in range(self.n_agents)
                     if acts is not None and acts[i] == self.PICKUP]
        n_commit = len(committed)

        event = self._last_event
        verdict_color = {
            "success":    th["ok"],
            "overshoot":  th["bad"],
            "undershoot": th["bad"],
        }.get(event, th["muted"])

        # --- crate, resting on the horizon drawn by sky_field --------------
        box_x, box_y, box_s = W / 2.0, 2.45, 2.15

        crate_state = "lifting" if event == "success" else "active"
        dr.crate(ax, box_x, box_y, size=box_s, th=th,
                 label=None if event == "success" else f"EXACTLY {K}",
                 sub=None, state=crate_state, zorder=3)
        if event != "success":
            ax.text(box_x, box_y - box_s * 0.30, "must lift together",
                    ha="center", va="center", color="#FFFFFF", fontsize=7.6,
                    alpha=0.85, zorder=3.6,
                    path_effects=dr.stroke(th["crate_edge"], 2.0))

        # --- the drones, fanned above the crate ---------------------------
        R = 3.25
        angles = (_np.linspace(155.0, 25.0, self.n_agents) if self.n_agents > 1
                  else _np.array([90.0]))
        hook = (box_x, box_y + box_s * 0.52)
        drone_xy = []
        for i, ang in enumerate(angles):
            rad = _np.radians(ang)
            x = box_x + R * _np.cos(rad)
            y = box_y + box_s * 0.30 + R * _np.sin(rad) * 0.80
            lifting = i in committed
            if lifting:
                y += 0.28
            drone_xy.append((x, y))
            color = dr.agent_color(th, i)
            dr.tether(ax, (x, y - 0.30), hook, taut=lifting,
                      color=verdict_color if lifting else None, th=th)
            dr.drone(ax, x, y, size=0.44, color=color, th=th, label=f"D{i + 1}",
                     halo=verdict_color if lifting else None, halo_alpha=0.30,
                     zorder=7)
            ax.text(x, y - 0.86, "PICKUP" if lifting else "wait", ha="center",
                    va="top", color=verdict_color if lifting else th["muted"],
                    fontsize=7.2, fontweight="bold" if lifting else "normal",
                    zorder=8, path_effects=dr.stroke(th["field_top"], 2.4))

        # --- "everybody sees the same thing" band -------------------------
        band_y = H - 0.62
        ax.plot([1.0, W - 1.0], [band_y, band_y], color=th["comm"], lw=1.2,
                alpha=0.55, linestyle=(0, (5, 4)), zorder=2)
        for (x, y) in drone_xy:
            # the same 2 floats dropping into every drone
            ax.plot([x, x], [band_y, y + 0.62], color=th["comm"], lw=0.9,
                    alpha=0.30, linestyle=(0, (2, 3)), zorder=2)
            dr.badge_dot(ax, x, y + 0.62, th["comm"], th, r=0.055, zorder=2.4)
        ax.text(W / 2.0, band_y + 0.18,
                f"identical broadcast  obs = [K/N, N/n_max] = "
                f"[{K / self.n_agents:.2f}, {self.n_agents / self.n_max:.2f}]"
                f"   —   no drone can tell itself apart",
                ha="center", va="bottom", color=th["comm"], fontsize=8.0,
                fontweight="bold", zorder=3)

        # --- the count, stated where the eye lands ------------------------
        count_text = f"{n_commit}  committed   /   {K}  required"
        if acts is None:
            count_text = f"waiting for the joint action   /   {K}  required"
        ax.text(W / 2.0, 0.40, count_text, ha="center", va="center",
                color="#FFFFFF" if event else th["ink"], fontsize=11.0,
                fontweight="bold", zorder=9,
                bbox=dict(boxstyle="round,pad=0.42",
                          facecolor=verdict_color if event else th["panel"],
                          edgecolor="none" if event else th["panel_edge"],
                          alpha=0.95))

        # --- HUD ----------------------------------------------------------
        dr.header(
            fig, ax, "Pickup Consensus",
            f"{self.n_agents} identical carriers on one crate · K∈[{self.k_min},"
            f"{self.k_max}] resampled each episode · no positions, no roles, no "
            f"private information",
            th,
        )
        dr.stat_row(fig, [
            ("step", f"{self._t}/{self._time_limit}"),
            ("required K", str(K)),
            ("committed", str(n_commit), verdict_color if event else None),
            ("outcome", (event or ("stalled" if self._t >= self._time_limit
                                   else "in progress")).upper(),
             verdict_color),
        ], th)

        if event == "success":
            dr.status_pill(fig, f"CORRECT LIFT   {n_commit} = K", th["ok"], th)
        elif event == "overshoot":
            dr.status_pill(fig, f"OVERSHOOT   {n_commit} > K={K}", th["bad"], th)
        elif event == "undershoot":
            dr.status_pill(fig, f"UNDERSHOOT   {n_commit} < K={K}", th["bad"], th)
        elif self._t >= self._time_limit:
            dr.status_pill(fig, "NO ATTEMPT — all-WAIT freeze", th["muted"], th)

        dr.legend_row(fig, [
            (th["muted"], "slack tether = WAIT"),
            (th["ok"], "taut tether = PICKUP committed"),
            (th["comm"], "identical observation broadcast to every drone"),
        ], th)
        dr.timeline(fig, self._t, self._time_limit, th)
        dr.footer(fig, "Pickup Consensus · MAC benchmark", th,
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
        """The original one-line console render, kept for headless logging."""
        state = "SUCCESS" if self._succeeded else (
            "in progress" if self._t < self._time_limit else "TIMEOUT"
        )
        print(
            f"[PickupConsensus] t={self._t:2d}/{self._time_limit}  "
            f"N={self.n_agents}  K={self.K}  "
            f"last_pickups={self._last_n}  {state}"
        )

    def close(self):
        fig = getattr(self, "_fig", None)
        if fig is not None:
            import matplotlib.pyplot as plt

            plt.close(fig)
            self._fig = None
        return None
