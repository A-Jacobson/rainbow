import math
import os
import random

import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim import Adam
from tqdm import tqdm

from memory import ReplayMemory
from utils import process_state


class DQAgent:
    """
    DeepQ Agent without bells and whistles. Uses single Q network and replay memory to interact with environment.
    """

    def __init__(self, q_network, environment, exp_name='dqn',
                 checkpoint_dir='checkpoints'):
        """
        Args:
            q_network:
            replay_memory:
            environment:
            save_path:
        """
        self.q_network = q_network
        self.replay_memory = None
        self.environment = environment

        # book keeping
        self.current_step = 0
        self.current_episode = 0
        self.save_path = os.path.join(checkpoint_dir, exp_name + '.pkl')
        self.logdir = os.path.join('runs', exp_name)

    def calculate_epsilon(self, exploration_start, exploration_end, last_exploration_frame):
        """
        calculates epsilon value given steps done and speed of decay
        """
        epsilon = exploration_end + (exploration_start - exploration_end) * \
                  math.exp(-1. * self.current_step / last_exploration_frame)
        return epsilon

    def select_action(self, state, epsilon):
        """
        epsilon greedy policy.
        selects action corresponding to maximum predicted Q value, otherwise selects
        otherwise selects random action with epsilon probability.
        Args:
            state: current state of the environment (4 stack of image frames)
            epsilon: probability of random action (1.0 - 0.0)

        Returns: action
        """
        if epsilon > random.random():
            return self.environment.action_space.sample()
        state = Variable(process_state(state), volatile=True).cuda()
        return int(self.q_network(state).data.max(1)[1])

    def learn(self, num_episodes, batch_size=32, capacity=1000000, lr=1e-4,
              exploration_start=0.9, exploration_end=0.05,
              last_exploration_frame=1000000, render=False,
              checkpoint_interval=20):

        self.replay_memory = ReplayMemory(capacity)
        cudnn.benchmark = True

        if len(self.replay_memory) < 50000:
            print('populating replay memory..')
            self.prime_replay_memory()

        writer = SummaryWriter(self.logdir)
        optimizer = Adam(self.q_network.parameters(), lr=lr)
        criterion = nn.SmoothL1Loss()

        for episode in tqdm(range(num_episodes)):
            state = self.environment.reset()
            total_reward = 0
            while True:
                if render:
                    self.environment.render()

                epsilon = self.calculate_epsilon(exploration_start, exploration_end, last_exploration_frame)
                action = self.select_action(state, epsilon)  # selection an action
                next_state, reward, done, info = self.environment.step(action)  # carry out action/observe reward

                total_reward += reward
                writer.add_scalar('epsilon', epsilon, self.current_step)
                self.current_step += 1

                # store experience s, a, r, s', done in replay memory
                self.replay_memory.add(state, action, reward, next_state, done)
                # optimize !
                self.optimize_q_network(batch_size, optimizer, criterion)
                state = next_state  # move to next state

                if done:
                    writer.add_scalar('reward', total_reward, self.current_episode)
                    self.current_episode += 1
                    if (episode + 1) % checkpoint_interval == 0:
                        self.save_checkpoint()
                    break

        self.environment.close()

    def optimize_q_network(self, batch_size, optimizer, criterion):
        """
        Samples batch of transitions form replay_memory
        calculates q_values for current states and actions q_values = Q(s, a)
        calculates targets = rewards + (discount * maxQ(s2, a))
        optimizes loss between q_values and targets
        Args:
            batch_size:
            optimizer:
            criterion:
        """
        states, actions, rewards, next_states, done_mask = self.replay_memory.sample(batch_size)

        # prepare batch
        states = Variable(states).cuda()
        next_states = Variable(next_states).cuda()
        rewards = Variable(rewards).cuda()
        done_mask = Variable(done_mask).cuda()

        q_values = self.q_network(states)[range(len(actions)), actions]  # select only Q values for actions we took

        # find next Q values and set Q values for done states to 0
        next_q_values = self.q_network(next_states).max(dim=1)[0].detach() * done_mask
        # calculate targets = rewards + (gamma * next_Q_values)
        targets = rewards + (0.99 * next_q_values)

        loss = criterion(q_values, targets)
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    def play(self, num_episodes, epsilon=0.05, render=True):
        for _ in tqdm(range(num_episodes)):
            total_reward = 0
            state = self.environment.reset()
            while True:
                if render:
                    self.environment.render()
                action = self.select_action(state, epsilon)  # selection an action
                next_state, reward, done, info = self.environment.step(action)  # carry out action/observe reward
                total_reward += reward
                state = next_state  # move to next state
                if done:
                    break
        self.environment.close()

    def prime_replay_memory(self, steps=50000):
        """
        populates replay memory with transitions generated by random actions
        """
        while len(self.replay_memory) <= steps:
            state = self.environment.reset()
            while True:
                action = self.environment.action_space.sample()
                next_state, reward, done, info = self.environment.step(action)  # carry out action/observe reward
                self.replay_memory.add(state, action, reward, next_state, done)
                state = next_state  # move to next state
                if done:
                    break

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.q_network.load_state_dict(checkpoint['checkpoints'])
        self.current_step = checkpoint['current_step']
        self.current_episode = checkpoint['current_episode']

    def save_checkpoint(self):
        checkpoint = dict(weights=self.q_network.state_dict(),
                          current_step=self.current_step,
                          current_episode=self.current_episode)
        torch.save(checkpoint, self.save_path)
