import pytest
from memory import ReplayMemory
from atari_wrappers import wrap_deepmind
import gym


@pytest.fixture
def env():
    return wrap_deepmind(gym.make('Pong-v0'), frame_stack=True)


def test_capacity(env):
    replay_memory = ReplayMemory(capacity=10)
    state = env.reset()
    action = env.action_space.sample()
    next_state, reward, info, done = env.step(action)
    for _ in range(20):
        replay_memory.add(state, action, reward, next_state, done)
    assert len(replay_memory) == 10