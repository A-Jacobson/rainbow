import math
import random
from utils import process_state
from torch.autograd import Variable


def select_action(q_network, state, env, epsilon):
    """
    epsilon greedy policy.
    selects action corresponding to highest predicted Q value, otherwise selects
    otherwise selects random action with epsilon probability.
    Args:
        state: current state of the environment (4 stack of image frames)
        epsilon: probability of random action (1.0 - 0.0)

    Returns:(int) action to perform
    """
    if epsilon > random.random():
        return env.action_space.sample()
    state = Variable(process_state(state), volatile=True).cuda()
    return int(q_network(state).data.max(1)[1])


def calculate_epsilon(current_step, epsilon_max=0.9, epsilon_min=0.05, decay_rate=1e-5):
    """
    calculates epsilon value given steps done and speed of decay
    """
    epsilon = epsilon_min + (epsilon_max - epsilon_min) * \
              math.exp(-decay_rate * current_step)
    return epsilon