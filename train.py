import argparse

import gym

from agents import DQAgent, DDQAgent
from atari_wrappers import wrap_deepmind
from models import DQN
from config import AtariDefaults

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, type=str, help="training environment eg. Pong-v0")
    parser.add_argument("--num_steps", required=True, type=int, help="Number of steps to train")
    parser.add_argument("--agent_name", required=True, type=str, help="Name of agent, for logging and checkpointing")
    parser.add_argument("--agent_type", default="dqn", type=str, help='dqn, ddqn')
    parser.add_argument("--resume_training", type=bool, default=False)

    # defaults
    parser.add_argument("--memory_capacity", type=int, default=AtariDefaults.MEMORY_CAPACITY)
    parser.add_argument("--initial_memory", type=int, default=AtariDefaults.INITIAL_MEMORY)
    parser.add_argument("--learning_rate", type=float, default=AtariDefaults.LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=AtariDefaults.BATCH_SIZE)
    parser.add_argument("--max_epsilon", type=float, default=AtariDefaults.EPSILON_MAX)
    parser.add_argument("--min_epsilon", type=float, default=AtariDefaults.EPSILON_MIN)
    parser.add_argument("--decay_rate", type=float, default=AtariDefaults.DECAY_RATE)
    parser.add_argument("--checkpoint_interval", type=int, default=AtariDefaults.CHECKPOINT_INTERVAL)

    args = parser.parse_args()

    env = wrap_deepmind(gym.make(args.env), frame_stack=True)

    if args.agent_type == 'dqn':
        q_network = DQN(env.action_space.n).cuda()
        agent = DQAgent(q_network, env, name=args.agent_name)

    elif args.agent_type == 'ddqn':
        q_network = DQN(env.action_space.n).cuda()
        target_network = DQN(env.action_space.n).cuda()
        agent = DDQAgent(q_network, target_network, env, name=args.agent_name)

    if args.resume_training:
        agent.load_agent(args.agent_name)

    agent.learn(num_steps=args.num_steps, capacity=args.memory_capacity,
                initial_memory=args.initial_memory, lr=args.learning_rate,
                batch_size=args.batch_size, epsilon_max=args.max_epsilon,
                epsilon_min=args.min_epsilon, decay_rate=args.decay_rate,
                checkpoint_interval=args.checkpoint_interval)


