from .q_learner import QLearner
from .ppo_learner import PPOLearner


REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["ppo_learner"] = PPOLearner
