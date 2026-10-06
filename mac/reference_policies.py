"""Scripted reference policies and the levels the paper's figures draw.

Each policy reads the env's privileged internals (positions, K, batteries) --
they are yardsticks, not baselines. `measure_levels(name)` rolls them out on
the paper configuration over a fixed set of episode seeds, so every level of a
benchmark is measured on the same episodes.

    python -m mac.reference_policies                 # all four benchmarks
    python -m mac.reference_policies team_partition

Levels drawn in the paper (upper / lower dotted line):

    pickup_consensus     Perfect (exactly K PICKUP at t=0) / PICKUP w.p. K/N
    aerial_ordering      Perfect Solution / Navigate-and-Wait
    charging_rendezvous  Perfect Solution / Navigate-and-Wait
    team_partition       Perfect Solution / Navigate-and-Wait
"""
import contextlib
import io
import sys

import numpy as np

from . import make
from .envs import (
    AerialOrderingEnv as _AO,
    ChargingRendezvousEnv as _CC,
    PickupConsensusEnv as _PC,
)


# ----------------------------------------------------------------------------
# Pickup Consensus
# ----------------------------------------------------------------------------

def pc_perfect(env, rng):
    """Exactly K agents commit, immediately."""
    return [_PC.PICKUP] * env.K + [_PC.WAIT] * (env.n_agents - env.K)


def pc_random_k_over_n(env, rng):
    """Each agent commits independently with probability K/N -- the best a
    shared policy without agent ids can do on identical observations."""
    return (rng.random(env.n_agents) < env.K / env.n_agents).astype(int).tolist()


# ----------------------------------------------------------------------------
# Aerial Ordering
# ----------------------------------------------------------------------------

def _ao_free_connected(env):
    bits = env._comm_bits()
    return [wi for wi in range(1, env.n_agents)
            if bits[wi] > 0.5 and not env.loaded[wi - 1]]


def ao_perfect(env, rng):
    """Fly in, then exactly K of the assembled workers PICKUP."""
    tgt, K = env._current_target(), env._current_K()
    if tgt is None:
        return [_AO.MOVE] * env.n_agents
    if tuple(env.pos[0]) == tuple(tgt):
        ready = [wi for wi in _ao_free_connected(env)
                 if tuple(env.pos[wi]) == tuple(tgt)]
        if len(ready) >= K:
            a = [_AO.WAIT] * env.n_agents
            for wi in ready[:K]:
                a[wi] = _AO.PICKUP
            return a
    return [_AO.MOVE] * env.n_agents


def ao_navigate_wait(env, rng):
    """Fly to the target and WAIT there, re-flying whenever wind drifts an
    agent off. Never lifts, so it only ever pays the step cost."""
    tgt = env._current_target()
    if tgt is None:
        return [_AO.WAIT] * env.n_agents
    return [_AO.WAIT if tuple(env.pos[ai]) == tuple(tgt) else _AO.MOVE
            for ai in range(env.n_agents)]


# ----------------------------------------------------------------------------
# Charging Rendezvous
# ----------------------------------------------------------------------------

def _cc_docked_idle(env):
    return [i for i in range(env.n_agents)
            if not env.satisfied[i] and not env.is_charging[i]
            and env._on_charger(i)]


def cc_perfect(env, rng, margin_steps=3):
    """Oracle: fly everyone in, give the free ports to the neediest docked
    drones, and pre-emptively DISCONNECT a well-charged drone when a docked
    one is within `margin_steps` of drying out."""
    acts = [_CC.WAIT] * env.n_agents
    danger = env.drain_rate * margin_steps
    for i in range(env.n_agents):
        if not env.satisfied[i] and not env.is_charging[i] and not env._on_charger(i):
            acts[i] = _CC.MOVE
    free = env._free_ports()
    docked_idle = sorted(_cc_docked_idle(env), key=lambda i: env.battery[i])
    for i in docked_idle[:max(0, free)]:
        acts[i] = _CC.CONNECT
    critical = [i for i in docked_idle[max(0, free):] if env.battery[i] <= danger]
    chargers_desc = sorted([i for i in range(env.n_agents) if env.is_charging[i]],
                           key=lambda i: env.battery[i], reverse=True)
    for i in critical:
        if chargers_desc:
            c = chargers_desc.pop(0)
            if acts[c] == _CC.WAIT:
                acts[c] = _CC.DISCONNECT
    return acts


def cc_navigate_wait(env, rng):
    """Fly to the dock, then WAIT there -- never request a port."""
    return [_CC.MOVE if not env.satisfied[i] and not env._on_charger(i) else _CC.WAIT
            for i in range(env.n_agents)]


def cc_do_nothing(env, rng):
    """Everyone WAITs at spawn; the team dries out."""
    return [_CC.WAIT] * env.n_agents


def cc_greedy_connect(env, rng):
    """Every docked idle drone requests a port every step, so a tied cohort
    over-subscribes the station and is rejected every time."""
    acts = [_CC.WAIT] * env.n_agents
    for i in range(env.n_agents):
        if env.satisfied[i] or env.is_charging[i]:
            continue
        acts[i] = _CC.MOVE if not env._on_charger(i) else _CC.CONNECT
    return acts


def uniform_random(env, rng):
    """Uniform over each agent's available actions."""
    return [int(rng.choice(np.flatnonzero(av))) for av in env.get_avail_actions()]


# ----------------------------------------------------------------------------
# Team Partition
# ----------------------------------------------------------------------------

def _tp_present(env):
    return [i for i in range(env.n_agents) if env._dists()[i] == 0]


def tp_perfect(env, rng):
    """Rush in; when the gate opens, split the present agents exactly to quota."""
    if not env._gate_open():
        return [env.MOVE] * env.n_agents
    acts = [env.STAY] * env.n_agents
    slots = [c for c in range(env.n_zones) for _ in range(env.quota[c])]
    for i, c in zip(_tp_present(env), slots):
        acts[i] = c + 1
    return acts


def tp_count_barrier(env, rng):
    """Best memoryless symmetric play: each present agent draws its zone from
    the quota proportions K/N."""
    if not env._gate_open():
        return [env.MOVE] * env.n_agents
    p = np.asarray(env.quota, dtype=float)
    p = p / p.sum()
    acts = [env.STAY] * env.n_agents
    for i in _tp_present(env):
        acts[i] = int(rng.choice(env.n_zones, p=p)) + 1
    return acts


def tp_navigate_wait(env, rng):
    """Assemble at the staging cell and never commit (ends as a miss)."""
    return [env.STAY if env._dists()[i] == 0 else env.MOVE
            for i in range(env.n_agents)]


# ----------------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------------

# name -> (policies, episodes, first episode seed, success key, drawn levels).
# The seeds and episode counts are the ones the paper's levels were measured
# with; every policy of a benchmark replays the same episodes.
BENCHMARKS = {
    "pickup_consensus": (
        {"Perfect": pc_perfect, "PICKUP w.p. K/N": pc_random_k_over_n},
        5000, None, "success", ("Perfect", "PICKUP w.p. K/N")),
    "aerial_ordering": (
        {"Perfect Solution": ao_perfect, "Navigate-and-Wait": ao_navigate_wait},
        2000, 100_000, "targets_lifted", ("Perfect Solution", "Navigate-and-Wait")),
    "charging_rendezvous": (
        {"Perfect Solution": cc_perfect, "Navigate-and-Wait": cc_navigate_wait,
         "Do-Nothing": cc_do_nothing, "Greedy Connect": cc_greedy_connect,
         "Random": uniform_random},
        1000, 200_000, "all_satisfied", ("Perfect Solution", "Navigate-and-Wait")),
    "team_partition": (
        {"Perfect Solution": tp_perfect, "Count Barrier": tp_count_barrier,
         "Navigate-and-Wait": tp_navigate_wait},
        1500, 300_000, "is_success", ("Perfect Solution", "Navigate-and-Wait")),
}


def _rollout(env, policy, rng, reset_seed=None, force_k=None):
    env.reset(seed=reset_seed) if reset_seed is not None else env.reset()
    if force_k is not None:
        env.K = int(force_k)
    total, done, info = 0.0, False, {}
    while not done:
        _, r, term, trunc, info = env.step(policy(env, rng))
        total += r
        done = term or trunc
    return total, info


def _measure_pickup(policy, episodes, success_key, seed=0):
    """Pickup Consensus: K pinned per run and the level averaged over K, since
    the achievable success differs by K (as the paper's figures do it)."""
    with contextlib.redirect_stdout(io.StringIO()):
        env = make("pickup_consensus", seed=seed)
    per_k = []
    for k in range(env.k_min, env.k_max + 1):
        rng = np.random.default_rng(seed + k)
        rets, succ = zip(*[(r, float(info[success_key])) for r, info in
                           (_rollout(env, policy, rng, force_k=k)
                            for _ in range(episodes))])
        per_k.append((np.mean(rets), np.mean(succ)))
    return {"return_mean": float(np.mean([v[0] for v in per_k])),
            "success": float(np.mean([v[1] for v in per_k]))}


def measure_levels(name, episodes=None, seed=0):
    """{policy name: {"return_mean", "success"}} on the paper configuration."""
    policies, n_default, first_seed, success_key, _ = BENCHMARKS[name]
    n = episodes or n_default
    out = {}
    for label, policy in policies.items():
        if first_seed is None:
            out[label] = _measure_pickup(policy, n, success_key, seed)
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            env = make(name, seed=seed)
        rng = np.random.default_rng(seed)
        rets, succ = [], []
        for ep in range(n):
            r, info = _rollout(env, policy, rng, reset_seed=first_seed + ep)
            rets.append(r)
            succ.append(float(info.get(success_key, 0.0)))
        out[label] = {"return_mean": float(np.mean(rets)),
                      "success": float(np.mean(succ))}
    return out


def drawn_levels(name):
    """The (upper, lower) policy names the paper draws for `name`."""
    return BENCHMARKS[name][4]


def main(argv):
    names = argv or list(BENCHMARKS)
    for name in names:
        levels = measure_levels(name)
        drawn = drawn_levels(name)
        print(f"\n{name}")
        for label, v in levels.items():
            mark = "  (drawn)" if label in drawn else ""
            print(f"  {label:20s} return {v['return_mean']:+8.2f}   "
                  f"success {v['success']:.3f}{mark}")


if __name__ == "__main__":
    main(sys.argv[1:])
