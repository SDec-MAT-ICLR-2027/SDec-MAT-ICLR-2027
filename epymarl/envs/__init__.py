import os
import sys
import warnings

from .multiagentenv import MultiAgentEnv
from .gymma import GymmaWrapper

# <repo>/epymarl/envs/__init__.py -> <repo>, which holds the `mac` package.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Importing `mac` registers the MAC gymnasium ids (pickup-consensus-v0,
# aerial-ordering-v0, charging-rendezvous-v0, team-partition-v0). This module
# is imported in every runner worker process, so they are registered there too.
import mac  # noqa: E402,F401

try:
    from .smaclite_wrapper import SMACliteWrapper
except ImportError:
    SMACliteWrapper = None
    warnings.warn(
        "SMAClite is not installed, so `smaclite` environment will not be available!"
    )


if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


def __check_and_prepare_smac_kwargs(kwargs):
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    assert kwargs[
        "common_reward"
    ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
    del kwargs["common_reward"]
    del kwargs["reward_scalarisation"]
    assert "map_name" in kwargs, "Please specify the map_name in the env_args"
    return kwargs


def smaclite_fn(**kwargs) -> MultiAgentEnv:
    if SMACliteWrapper is None:
        raise ImportError(
            "SMACliteWrapper is unavailable. Install smaclite to use `env=smaclite`."
        )
    kwargs = __check_and_prepare_smac_kwargs(kwargs)
    return SMACliteWrapper(**kwargs)


def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return GymmaWrapper(**kwargs)


REGISTRY = {}
REGISTRY["smaclite"] = smaclite_fn
REGISTRY["gymma"] = gymma_fn


# registering both smac and smacv2 causes a pysc2 error
# --> dynamically register the needed env
def register_smac():
    from .smac_wrapper import SMACWrapper

    def smac_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACWrapper(**kwargs)

    REGISTRY["sc2"] = smac_fn


def register_smacv2():
    from .smacv2_wrapper import SMACv2Wrapper

    def smacv2_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACv2Wrapper(**kwargs)

    REGISTRY["sc2v2"] = smacv2_fn
