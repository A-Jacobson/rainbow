import argparse

import gym

from agents import DQAgent, DDQAgent
from atari_wrappers import wrap_deepmind
from models import DQN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, type=str, help="environment to train in eg. Pong-v0")
    parser.add_argument("--agent_name", required=True, help="Name of agent to use")
    parser.add_argument("--agent_type", default="dqn", type=str, help='dqn, ddqn')
    parser.add_argument("--num_episodes", required=True, type=int, help="Number of episodes to train")

    args = parser.parse_args()

    env = wrap_deepmind(gym.make(args.env), frame_stack=True, episode_life=False)

    if args.agent_type == 'dqn':
        q_network = DQN(env.action_space.n).cuda()
        agent = DQAgent(q_network, env)

    elif args.agent_type == 'ddqn':
        q_network = DQN(env.action_space.n).cuda()
        target_network = DQN(env.action_space.n).cuda()
        agent = DDQAgent(q_network, target_network, env)

    agent.load_agent(args.agent_name)
    agent.play(args.num_episodes)
