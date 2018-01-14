import gym
import pytest

from atari_wrappers import wrap_deepmind
from memory import ReplayMemory


@pytest.fixture
def env():
    return wrap_deepmind(gym.make('Pong-v0'), frame_stack=True)


def test_capacity(env):
    replay_memory = ReplayMemory(capacity=10)
    state = env.reset()
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    for _ in range(20):
        replay_memory.add(state, action, reward, next_state, done)
    assert len(replay_memory) == 10


def test_batch_format(env):
    replay_memory = ReplayMemory(capacity=5)
    state = env.reset()
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    for _ in range(5):
        replay_memory.add(state, action, reward, next_state, done)
    states, actions, rewards, next_states, done_mask = replay_memory.sample(5)
    assert states.size() == (5, 4, 84, 84)
    assert next_states.size() == (5, 4, 84, 84)
    assert len(actions) == 5
    assert len(rewards) == 5
    assert len(done_mask) == 5
