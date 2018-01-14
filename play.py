import argparse

import gym

from agents import DQAgent
from atari_wrappers import wrap_deepmind
from models import DQN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, type=str, help="environment to train in eg. Pong-v0")
    parser.add_argument("--checkpoint", required=True, help="path to checkpoint eg. checkpoints/dqn.pkl")
    parser.add_argument("--num_episodes", required=True, type=int, help="Number of episodes to train")

    args = parser.parse_args()

    env = wrap_deepmind(gym.make(args.env), frame_stack=True, episode_life=False)
    q_network = DQN(env.action_space.n).cuda()

    agent = DQAgent(q_network, env)
    agent.load_checkpoint(args.checkpoint)
    agent.play(args.num_episodes)
