"""MAC -- the Multi-Agent Counting benchmark.

Four cooperative tasks in which a team of interchangeable agents must get
EXACTLY K of its members to act, with K revealed only at run time:

  pickup_consensus     the bare counting problem: K of N identical agents must
                       PICKUP in the same step (no navigation, no hidden state)
  aerial_ordering      a sensor finds a target and reads its hidden K; exactly K
                       workers must lift it together
  charging_rendezvous  drones arriving in waves must ration a station with fewer
                       ports than drones before their batteries run dry
  team_partition       a team regroups, then splits across zones to match a
                       public quota exactly

Pickup Consensus is the controlled special case of Aerial Ordering; the other
three form the MAC benchmark proper.

    import mac
    env = mac.make("aerial_ordering", seed=0)          # paper configuration
    env = mac.make("aerial_ordering", wind_prob=0.0)   # with an override
"""
from .envs import (
    ENV_CLASSES,
    GYM_IDS,
    AerialOrderingEnv,
    ChargingRendezvousEnv,
    PickupConsensusEnv,
    TeamPartitionEnv,
)
from .presets import PAPER, paper_kwargs

ENV_NAMES = tuple(ENV_CLASSES)


def make(name, seed=None, **overrides):
    """Build `name` with the paper configuration, plus `overrides`."""
    kwargs = paper_kwargs(name, **overrides)
    if seed is not None:
        kwargs["seed"] = seed
    return ENV_CLASSES[name](**kwargs)


__all__ = [
    "make",
    "ENV_NAMES",
    "ENV_CLASSES",
    "GYM_IDS",
    "PAPER",
    "paper_kwargs",
    "PickupConsensusEnv",
    "AerialOrderingEnv",
    "ChargingRendezvousEnv",
    "TeamPartitionEnv",
]
