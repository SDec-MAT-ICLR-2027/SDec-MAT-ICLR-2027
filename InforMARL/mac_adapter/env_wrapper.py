"""MAC env -> InforMARL graph env interface.

The env itself is created by epymarl/envs REGISTRY["gymma"], the same constructor the
EPyMARL runs (mappo / commnet / tarmac) use, so obs padding, action masks, comm
masks, time limit and seeding are identical to the baselines.

Graph given to InforMARL
------------------------
  nodes      : the N agents only. Node feature of agent j = [obs_j, entity_type=0].
  edges      : the env's live communication status, rebuilt every step from
               get_comm_mask() with the same rule CommNet/TarMAC use
               (epymarl/modules/agents/comm_groups.py::build_group_mask):
                 bits      -> i and j connected iff bit_i AND bit_j
                 adjacency -> mask[i, j] (i receives from j) used as is
               Self loops are dropped (InforMARL never connects a node to itself).
  edge value : EDGE_VALUE for every connected pair. InforMARL keeps an edge iff
               0 < adj < max_edge_dist, so max_edge_dist must exceed EDGE_VALUE.

obs_agent_id
------------
  False (default): observations exactly as the env returns them.
  True: a one-hot agent index is appended to every agent's observation, the same
        thing epymarl/controllers/comm_mac.py does for obs_agent_id=True (obs, then
        eye(N)). It therefore also reaches the node features, just as in
        CommNet/TarMAC the id enters the encoder and so the messages.
"""

import os
import sys

import numpy as np
import yaml
from gym import spaces

ADAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
INFORMARL_ROOT = os.path.dirname(ADAPTER_DIR)
REPO_ROOT = os.path.dirname(INFORMARL_ROOT)
SRC_DIR = os.path.join(REPO_ROOT, "epymarl")

EDGE_VALUE = 1.0


def ensure_paths():
    """InforMARL root first (its `utils`, `onpolicy`, `multiagent`), then the
    repo root and epymarl/ appended (for `mac` and epymarl's `envs`)."""
    if INFORMARL_ROOT not in sys.path:
        sys.path.insert(0, INFORMARL_ROOT)
    for p in (REPO_ROOT, SRC_DIR):
        if p not in sys.path:
            sys.path.append(p)


def load_env_config(name):
    path = os.path.join(SRC_DIR, "config", "envs", f"{name}.yaml")
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def parse_env_overrides(items):
    """['k_min=2', 'comm_radius=3'] -> {'k_min': 2, 'comm_radius': 3}."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--env_args expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        if k.startswith("env_args."):
            k = k[len("env_args."):]
        out[k] = yaml.safe_load(v)
    return out


def comm_to_pairwise(comm_mask, n_agents, comm_mask_type="bits", symmetric=False):
    """Same semantics as build_group_mask, without the self loop.
    Returns bool [N, N] where pair[i, j] means agent i receives from agent j."""
    cm = np.asarray(comm_mask, dtype=np.float32)
    if comm_mask_type == "adjacency":
        pair = cm.reshape(n_agents, n_agents) > 0.5
        if symmetric:
            pair = pair | pair.T
    else:
        bits = cm.reshape(n_agents) > 0.5
        pair = bits[:, None] & bits[None, :]
    pair = pair.copy()
    np.fill_diagonal(pair, False)
    return pair


def pairwise_to_adj(pair):
    """InforMARL/PyG edges run row -> col (source -> target), so a message from
    j to i needs adj[j, i]."""
    return pair.T.astype(np.float32) * EDGE_VALUE


def make_gymma_env(env_args, common_reward, reward_scalarisation, seed):
    ensure_paths()
    from envs import REGISTRY as env_REGISTRY

    kwargs = dict(env_args)
    kwargs["seed"] = seed
    kwargs["common_reward"] = common_reward
    kwargs["reward_scalarisation"] = reward_scalarisation
    return env_REGISTRY["gymma"](**kwargs)


class MACGraphEnv:
    """One MAC env with InforMARL's MultiAgentGraphEnv interface."""

    def __init__(self, env_args, common_reward=True, reward_scalarisation="sum",
                 seed=0, comm_symmetric=False, obs_agent_id=False):
        self.env = make_gymma_env(env_args, common_reward, reward_scalarisation, seed)
        self.common_reward = common_reward
        info = self.env.get_env_info()
        self.n = self.num_agents = int(info["n_agents"])
        self.env_obs_dim = int(info["obs_shape"])
        self.obs_agent_id = bool(obs_agent_id)
        self.obs_dim = self.env_obs_dim + (self.n if self.obs_agent_id else 0)
        self.n_actions = int(info["n_actions"])
        self.episode_limit = info["episode_limit"]
        self.comm_mask_type = info.get("comm_mask_type", "bits")
        self.comm_symmetric = bool(comm_symmetric)

        N, D = self.n, self.obs_dim

        def box(shape):
            return spaces.Box(low=-np.inf, high=+np.inf, shape=shape, dtype=np.float32)

        self.observation_space = [box((D,)) for _ in range(N)]
        self.share_observation_space = [box((N * D,)) for _ in range(N)]
        self.action_space = [spaces.Discrete(self.n_actions) for _ in range(N)]
        self.node_observation_space = [box((N, D + 1)) for _ in range(N)]
        self.adj_observation_space = [box((N, N)) for _ in range(N)]
        self.edge_observation_space = [box((1,)) for _ in range(N)]
        self.agent_id_observation_space = [box((1,)) for _ in range(N)]
        self.share_agent_id_observation_space = [box((N,)) for _ in range(N)]

    # ------------------------------------------------------------------
    def get_spaces(self):
        return (
            self.observation_space,
            self.share_observation_space,
            self.action_space,
            self.node_observation_space,
            self.adj_observation_space,
            self.edge_observation_space,
            self.agent_id_observation_space,
            self.share_agent_id_observation_space,
        )

    def describe(self):
        return {
            "n_agents": self.n,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "episode_limit": self.episode_limit,
            "comm_mask_type": self.comm_mask_type,
            "obs_agent_id": self.obs_agent_id,
        }

    def get_pairwise(self):
        return comm_to_pairwise(self.env.get_comm_mask(), self.n,
                                self.comm_mask_type, self.comm_symmetric)

    def get_avail_actions(self):
        return np.asarray(self.env.get_avail_actions(), dtype=np.float32)

    def _graph_obs(self):
        obs = np.asarray(self.env.get_obs(), dtype=np.float32).reshape(self.n, self.env_obs_dim)
        if self.obs_agent_id:
            obs = np.concatenate([obs, np.eye(self.n, dtype=np.float32)], axis=1)
        entity_type = np.zeros((self.n, 1), dtype=np.float32)  # every node is an agent
        node_obs = np.concatenate([obs, entity_type], axis=1)
        adj = pairwise_to_adj(self.get_pairwise())
        obs_n = [obs[i] for i in range(self.n)]
        agent_id_n = [np.array([i]) for i in range(self.n)]
        node_obs_n = [node_obs.copy() for _ in range(self.n)]
        adj_n = [adj.copy() for _ in range(self.n)]
        return obs_n, agent_id_n, node_obs_n, adj_n

    # ------------------------------------------------------------------
    def reset(self):
        self.env.reset()
        return self._graph_obs()

    def step(self, action_n):
        a = np.asarray(action_n)
        if a.ndim == 2:  # InforMARL runners send one-hot actions
            a = a.argmax(axis=-1)
        actions = [int(x) for x in a.reshape(-1)]
        _, reward, terminated, truncated, info = self.env.step(actions)
        done = bool(terminated or truncated)

        if self.common_reward or np.isscalar(reward):
            reward_n = [[float(reward)] for _ in range(self.n)]
        else:
            reward_n = [[float(r)] for r in reward]

        obs_n, agent_id_n, node_obs_n, adj_n = self._graph_obs()
        done_n = [done] * self.n
        info_n = [dict(info) for _ in range(self.n)]
        return obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n

    def seed(self, seed=None):
        self.env.seed(seed)

    def close(self):
        self.env.close()
