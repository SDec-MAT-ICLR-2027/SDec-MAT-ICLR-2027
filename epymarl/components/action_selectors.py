import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class SoftPoliciesSelector():

    def __init__(self, args):
        self.args = args
        # Small uniform-exploration mix during training to prevent deterministic collapse.
        # Annealed from explore_eps_start → explore_eps_end over t_max steps.
        # test_mode always stays greedy (pure policy probabilities).
        self._explore_eps_start = float(getattr(args, "explore_eps_start", 0.05))
        self._explore_eps_end = float(getattr(args, "explore_eps_end", 0.01))
        self._t_max = int(getattr(args, "t_max", 1))

    def _explore_eps(self, t_env):
        ratio = min(max(float(t_env) / max(self._t_max, 1), 0.0), 1.0)
        return self._explore_eps_start + ratio * (self._explore_eps_end - self._explore_eps_start)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        if test_mode:
            # Pure policy — no exploration noise
            m = Categorical(agent_inputs)
            return m.sample().long()

        eps = self._explore_eps(t_env)
        # Mix: (1-eps)*pi + eps*uniform_over_available
        n_avail = avail_actions.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        uniform = avail_actions.float() / n_avail
        mixed = (1.0 - eps) * agent_inputs + eps * uniform
        m = Categorical(mixed)
        return m.sample().long()


REGISTRY["soft_policies"] = SoftPoliciesSelector