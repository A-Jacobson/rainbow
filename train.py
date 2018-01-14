import argparse
import gym
from atari_wrappers import wrap_deepmind
from agents import DQAgent
from models import DQN
from memory import ReplayMemory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, type=str, help="environment to train in eg. Pong-v0")
    parser.add_argument("--num_episodes", required=True, type=int, help="Number of episodes to train")
    parser.add_argument("--exp_name", required=True, type=str, default="dqn")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint eg(checkpoints/dqn.pkl)")
    parser.add_argument("--capacity", default=1000000, type=int, help="replay memory capacity")

    args = parser.parse_args()

    env = wrap_deepmind(gym.make(args.env), frame_stack=True)
    memory = ReplayMemory(args.capacity)
    q_network = DQN(env.action_space.n).cuda()

    agent = DQAgent(q_network, memory, env, exp_name=args.exp_name)
    if args.checkpoint:
        agent.load_checkpoint(args.checkpoint)
    agent.learn(args.num_episodes)

