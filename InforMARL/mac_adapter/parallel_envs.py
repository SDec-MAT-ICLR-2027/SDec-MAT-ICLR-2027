"""Parallel copies of an MAC env (one subprocess per env), like EPyMARL's
batch_size_run parallel runner.

Same protocol as InforMARL's GraphSubprocVecEnv, plus the action masks. The masks
are read AFTER the auto-reset, so the first step of a new episode gets the new
episode's masks (InforMARL's graphworker has no mask channel at all).
"""

from multiprocessing import Pipe, Process

import numpy as np

from mac_adapter.env_wrapper import MACGraphEnv, ensure_paths


class CloudpickleWrapper:
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


def env_worker(remote, parent_remote, env_fn_wrapper):
    ensure_paths()
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, ag_id, node_ob, adj, reward, done, info = env.step(data)
            if np.all(done):
                ob, ag_id, node_ob, adj = env.reset()
            remote.send((ob, ag_id, node_ob, adj, reward, done, info,
                         env.get_avail_actions()))
        elif cmd == "reset":
            ob, ag_id, node_ob, adj = env.reset()
            remote.send((ob, ag_id, node_ob, adj, env.get_avail_actions()))
        elif cmd == "get_spaces":
            remote.send(env.get_spaces())
        elif cmd == "describe":
            remote.send(env.describe())
        elif cmd == "close":
            env.close()
            remote.close()
            break
        else:
            raise NotImplementedError(cmd)


class _SpacesMixin:
    def _set_spaces(self, sp):
        (
            self.observation_space,
            self.share_observation_space,
            self.action_space,
            self.node_observation_space,
            self.adj_observation_space,
            self.edge_observation_space,
            self.agent_id_observation_space,
            self.share_agent_id_observation_space,
        ) = sp


def _stack_step(results):
    obs, ag_ids, node_obs, adj, rews, dones, infos, avail = zip(*results)
    return (np.stack(obs), np.stack(ag_ids), np.stack(node_obs), np.stack(adj),
            np.stack(rews), np.stack(dones), infos, np.stack(avail))


def _stack_reset(results):
    obs, ag_ids, node_obs, adj, avail = zip(*results)
    return (np.stack(obs), np.stack(ag_ids), np.stack(node_obs), np.stack(adj),
            np.stack(avail))


class SubprocParallelEnvs(_SpacesMixin):
    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
        self.ps = [
            Process(target=env_worker,
                    args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(("get_spaces", None))
        self._set_spaces(self.remotes[0].recv())
        self.remotes[0].send(("describe", None))
        self.info = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        return _stack_step([remote.recv() for remote in self.remotes])

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return _stack_reset([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True


class InProcessEnvs(_SpacesMixin):
    """Single-process fallback, used when only one env is requested."""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self._set_spaces(self.envs[0].get_spaces())
        self.info = self.envs[0].describe()

    def step(self, actions):
        results = []
        for a, env in zip(actions, self.envs):
            ob, ag_id, node_ob, adj, reward, done, info = env.step(a)
            if np.all(done):
                ob, ag_id, node_ob, adj = env.reset()
            results.append((ob, ag_id, node_ob, adj, reward, done, info,
                            env.get_avail_actions()))
        return _stack_step(results)

    def reset(self):
        results = []
        for env in self.envs:
            ob, ag_id, node_ob, adj = env.reset()
            results.append((ob, ag_id, node_ob, adj, env.get_avail_actions()))
        return _stack_reset(results)

    def close(self):
        for env in self.envs:
            env.close()


def make_parallel_envs(env_args, common_reward, reward_scalarisation, seeds,
                       comm_symmetric=False, obs_agent_id=False):
    def get_env_fn(seed):
        def init_env():
            return MACGraphEnv(env_args, common_reward, reward_scalarisation,
                                 seed=seed, comm_symmetric=comm_symmetric,
                                 obs_agent_id=obs_agent_id)
        return init_env

    fns = [get_env_fn(s) for s in seeds]
    if len(fns) == 1:
        return InProcessEnvs(fns)
    return SubprocParallelEnvs(fns)
