import gym
import pytest

from atari_wrappers import wrap_deepmind
from models import DQN
from agents import DQAgent


@pytest.fixture
def agent():
    env = wrap_deepmind(gym.make('Pong-v0'), frame_stack=True)
    q_network = DQN(env.action_space.n).cuda()
    return DQAgent(q_network, env)


def test_epsilon_decay(agent):
    agent.current_step = 1000000
    eps = agent.calculate_epsilon(0.9, 0.05, 1e-4)
    assert eps == 0.05
    agent.current_step = 0
    eps = agent.calculate_epsilon(0.9, 0.05, 1e-4)
    assert eps == 0.9

