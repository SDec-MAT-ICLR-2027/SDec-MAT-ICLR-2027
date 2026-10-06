"""
MAT vec-env wrapper over the InforMARL joint-capture navigation scenario.

The underlying scenario lives in the InforMARL tree
(`multiagent/custom_scenarios/navigation_joint_graph.py`) and is imported,
not reimplemented, so the CT-SDE arm, the MAPPO arm and the InforMARL GNN
arm all run against byte-identical dynamics and rewards.

Arms
––––
  mappo           independent actors, no joint action. Pair with
                  `--obs_mode shared` for the observation-sharing
                  decentralized baseline.
  mat_comm_soft   CT-SDE in its plain form: group-masked encoder/decoder,
                  soft_group_bias critic, ordinary learned START token per
                  group. No group-token GO head, no obs-skip MLP, no
                  head-agent token prediction.

Groups are target cliques: agents inside `capture_radius` of the same live
target. That is the state where their actions are actually coupled.
"""
import os
import sys

import numpy as np
from gym.spaces import Discrete

# arm names accepted by this wrapper
MAPPO_ALGOS = ("mappo", "mat_dec")
# `mat_comm_soft` is the arm name the user passes; train_joint_nav.py
# normalizes it to `mat_comm` (the network dispatch key in
# transformer_policy.py) with encoder_critic_attn forced to soft_group_bias.
# Both spellings are accepted here so the wrapper works either side of that.
COMM_ALGOS = ("mat_comm_soft", "mat_comm_dec", "mat_comm")
ALLOWED_ALGOS = MAPPO_ALGOS + COMM_ALGOS

REQUIRED_ARGS = (
    "algorithm_name",
    "n_agents",
    "num_targets",
    "k_req",
    "episode_length",
)

# defaults mirrored from the scenario so the banner can print resolved values
_SCENARIO_DEFAULTS = dict(
    world_size=2.0,
    num_obstacles=0,
    collaborative=True,
    max_speed=2.0,
    collision_rew=0.0,
    use_dones=False,
    graph_feat_type="relative",
    max_edge_dist=1.0,
    min_dist_thresh=0.1,
    capture_radius=0.3,
    group_radius=0.6,
    capture_rew=10.0,
    fail_rew=-1.0,
    assembly_rew=0.05,
    dist_rew_scale=1.0,
    potential_shaping=1,
    shaping_gamma=0.99,
    obs_ready_count=False,
    obs_mode="local",
    sense_radius=1.0,
    comm_mode="target_clique",
    min_target_sep=None,  # None -> 2 * group_radius (scenario default)
    shuffle_targets=0,
    target_order="id",
    k_max=None,  # None -> original >= k_req rule; else exact-k (E4)
    step_cost=0.0,  # 0.0 -> reproduces every run made before this flag existed
    # census-vs-tally: both defaults reproduce every earlier run
    obs_radius=None,   # None -> use sense_radius (agent-to-agent sight)
    obs_abs_pos=1,     # 1 -> keep absolute own position in the observation
)


def _informarl_root():
    """Locate the vendored InforMARL tree so `multiagent` is importable."""
    env_override = os.environ.get("INFORMARL_ROOT")
    if env_override:
        return env_override
    here = os.path.dirname(os.path.abspath(__file__))
    # mat/envs/joint_nav -> mat/envs -> mat -> Multi-Agent-Transformer_ctsde -> repo
    repo = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    return os.path.join(repo, "InforMARL")


class JointNavMATEnv:
    """Vec-env API over the joint-capture navigation scenario.

    The team reward is already shared by the scenario, so it is tiled to all
    agents unchanged.
    """

    def __init__(self, args):
        missing = [k for k in REQUIRED_ARGS if not hasattr(args, k)]
        if missing:
            raise ValueError("missing required env args: %s" % missing)

        algo = str(args.algorithm_name)
        if algo not in ALLOWED_ALGOS:
            raise ValueError(
                "joint_nav wrapper allows %s, got %r" % (ALLOWED_ALGOS, algo)
            )
        self.algorithm_name = algo
        self._append_pairwise_comm = algo in COMM_ALGOS

        root = _informarl_root()
        if not os.path.isdir(root):
            raise RuntimeError(
                "InforMARL tree not found at %r; set INFORMARL_ROOT" % root
            )
        if root not in sys.path:
            sys.path.insert(0, root)
        from multiagent.MPE_env import GraphMPEEnv

        self.n_agents = int(args.n_agents)
        self.num_targets = int(args.num_targets)
        self.k_req = int(args.k_req)
        self.episode_limit = int(args.episode_length)

        scen_args = _ScenarioArgs(args, self.n_agents, _SCENARIO_DEFAULTS)
        self._env = GraphMPEEnv(scen_args)
        self._scenario = self._env.reward_callback.__self__
        self._world = self._env.world
        self.obs_mode = self._scenario.obs_mode
        self.capture_radius = self._scenario.capture_radius

        self.n_actions = int(self._env.action_space[0].n)
        obs_dim = int(self._env.observation_space[0].shape[0])

        # share_obs = every agent's FULL-view (shared-mode) observation, so
        # the centralized critic sees the global state whatever the actor
        # observes. In shared mode this is exactly the concatenated obs.
        self._state_obs_dim = (
            8 + 2 * (self.n_agents - 1) + 3 * self.num_targets
            + int(bool(self._scenario.obs_ready_count))
        )
        state_dim = self.n_agents * self._state_obs_dim
        if self._append_pairwise_comm:
            state_dim += self.n_agents * self.n_agents

        self.observation_space = [[obs_dim] for _ in range(self.n_agents)]
        self.share_observation_space = [[state_dim] for _ in range(self.n_agents)]
        self.action_space = [Discrete(self.n_actions) for _ in range(self.n_agents)]

        self._ep_step = 0
        self._ep_return = 0.0

        print(
            "[JointNavMATEnv] algo=%s n_agents=%d num_targets=%d k_req=%d "
            "capture_radius=%.2f capture_rew=%.1f fail_rew=%.1f "
            "assembly_rew=%.3f dist_rew_scale=%.2f step_cost=%.3f obs_mode=%s "
            "obs_ready_count=%s horizon=%d obs_dim=%d share_obs_dim=%d "
            "n_actions=%d append_pairwise_comm=%s comm_mode=%s "
            "(sense_radius=%.2f obs_radius=%.2f obs_abs_pos=%s)"
            % (
                self.algorithm_name,
                self.n_agents,
                self.num_targets,
                self.k_req,
                self.capture_radius,
                self._scenario.capture_rew,
                self._scenario.fail_rew,
                self._scenario.assembly_rew,
                self._scenario.dist_rew_scale,
                self._scenario.step_cost,
                self.obs_mode,
                self._scenario.obs_ready_count,
                self.episode_limit,
                obs_dim,
                state_dim,
                self.n_actions,
                self._append_pairwise_comm,
                self._scenario.comm_mode,
                self._scenario.sense_radius,
                self._scenario.obs_radius,
                self._scenario.obs_abs_pos,
            )
        )

    # ------------------------------------------------------------------
    def get_comm_mask(self):
        """N x N group adjacency (diagonal on), chosen by --comm_mode:
        target_clique (same live target) or agent_radius (within R)."""
        return self._scenario.comm_adjacency(self._world)

    def _obs(self, obs_list):
        return np.asarray(obs_list, dtype=np.float32)

    def _share_obs(self, obs):
        state = np.concatenate([
            self._scenario.state_observation(a, self._world)
            for a in self._world.agents
        ]).astype(np.float32)
        if state.size != self.n_agents * self._state_obs_dim:
            raise ValueError(
                "global state size %d != n_agents*%d"
                % (state.size, self._state_obs_dim))
        if self._append_pairwise_comm:
            mask = self.get_comm_mask().reshape(-1)
            n2 = self.n_agents * self.n_agents
            if mask.size != n2:
                raise ValueError(
                    "get_comm_mask flattened size %d != n_agents**2=%d"
                    % (mask.size, n2)
                )
            state = np.concatenate([state, mask.astype(np.float32)], axis=0)
        return np.tile(state, (self.n_agents, 1))

    def _avail(self):
        # every action is always legal in this task
        return np.ones((self.n_agents, self.n_actions), dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self):
        obs_list, _, _, _ = self._env.reset()
        self._ep_step = 0
        self._ep_return = 0.0
        obs = self._obs(obs_list)
        return obs, self._share_obs(obs), self._avail()

    def step(self, actions):
        acts = [int(a) for a in np.asarray(actions).reshape(-1)]
        if len(acts) != self.n_agents:
            raise ValueError(
                "expected %d actions, got %d" % (self.n_agents, len(acts))
            )
        onehot = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        onehot[np.arange(self.n_agents), acts] = 1.0

        obs_list, _, _, _, reward_n, _, info_n = self._env.step(list(onehot))

        self._ep_step += 1
        team_reward = float(reward_n[0][0])
        self._ep_return += team_reward

        scen_info = info_n[0]
        targets_left = int(scen_info["targets_left"])
        all_captured = bool(scen_info["all_captured"])
        # fixed-horizon episodes; clearing early just stops earning
        episode_done = bool(self._ep_step >= self.episode_limit or all_captured)
        truncated = bool(self._ep_step >= self.episode_limit and targets_left > 0)

        dones = np.array([episode_done] * self.n_agents, dtype=bool)
        rewards = np.full((self.n_agents, 1), team_reward, dtype=np.float32)

        info = [
            {
                "bad_transition": truncated,
                "episode_steps": self._ep_step,
                "episode_return": self._ep_return,
                "joint_captures": int(scen_info["joint_captures"]),
                "undershoots": int(scen_info["undershoots"]),
                "overkills": int(scen_info.get("overkills", 0)),
                "capture_actions": int(scen_info["capture_actions"]),
                "targets_left": targets_left,
                "won": all_captured,
                "episode_done": episode_done,
            }
            for _ in range(self.n_agents)
        ]

        obs = self._obs(obs_list)
        return obs, self._share_obs(obs), rewards, dones, info, self._avail()

    def seed(self, seed=None):
        if seed is not None:
            seed = int(seed)
            np.random.seed(seed)
            self._env.seed(seed)
        return seed

    def close(self):
        pass

    def save_replay(self):
        pass


class _ScenarioArgs:
    """Adapts MAT's `all_args` to the argparse.Namespace the scenario expects."""

    def __init__(self, args, n_agents, defaults):
        self.scenario_name = "navigation_joint_graph"
        self.num_agents = int(n_agents)
        self.num_scripted_agents = 0
        self.episode_length = int(args.episode_length)
        self.num_targets = int(args.num_targets)
        self.k_req = int(args.k_req)
        for key, default in defaults.items():
            setattr(self, key, getattr(args, key, default))
