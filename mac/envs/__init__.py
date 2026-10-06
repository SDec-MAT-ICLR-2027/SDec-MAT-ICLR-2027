"""The MAC environments and their gymnasium registrations.

Each env speaks two interfaces at once:
  * gymnasium:  reset() / step(joint_action) -> (obs, reward, terminated,
                truncated, info), with a shared team reward;
  * SMAC-style: get_obs(), get_state(), get_avail_actions(), get_env_info(),
                which is what the MAT, EPyMARL and InforMARL adapters read.
"""
from .aerial_ordering import AerialOrderingEnv
from .aerial_ordering_multitarget import AerialOrderingMultiTargetEvalEnv
from .charging_rendezvous import ChargingRendezvousEnv
from .pickup_consensus import PickupConsensusEnv
from .team_partition import TeamPartitionEnv

# Short name -> class. The short names are what configs and CLIs use.
ENV_CLASSES = {
    "pickup_consensus": PickupConsensusEnv,
    "aerial_ordering": AerialOrderingEnv,
    "charging_rendezvous": ChargingRendezvousEnv,
    "team_partition": TeamPartitionEnv,
}

# Short name -> gymnasium id.
GYM_IDS = {
    "pickup_consensus": "pickup-consensus-v0",
    "aerial_ordering": "aerial-ordering-v0",
    "charging_rendezvous": "charging-rendezvous-v0",
    "team_partition": "team-partition-v0",
}


def _register():
    try:
        import gymnasium as gym
    except ImportError:  # pragma: no cover - gymnasium is a hard requirement
        return
    for name, gym_id in GYM_IDS.items():
        if gym_id not in gym.registry:
            gym.register(gym_id, entry_point=ENV_CLASSES[name])


_register()

__all__ = [
    "PickupConsensusEnv",
    "AerialOrderingEnv",
    "ChargingRendezvousEnv",
    "TeamPartitionEnv",
    "AerialOrderingMultiTargetEvalEnv",
    "ENV_CLASSES",
    "GYM_IDS",
]
