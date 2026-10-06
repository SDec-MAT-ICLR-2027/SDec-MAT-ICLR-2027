"""Experiment configuration for Joint-Action Navigation.

`common.py` holds everything shared by every task and method; `tasks.py` the
four task definitions; `arms.py` the four methods. `run_experiment.py` at the
repository root composes a command line from them.

Splitting it this way keeps the invariant the comparison depends on: within a
task, every method gets the same environment, the same PPO hyperparameters and
the same three switches. Only `arms.py` varies by method, and it carries
nothing but the architecture.
"""

from .arms import ARMS
from .common import ENV, NET, PPO
from .tasks import TASKS

__all__ = ["ARMS", "ENV", "NET", "PPO", "TASKS"]
