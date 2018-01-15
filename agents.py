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

    def __init__(self, q_network, environment, name='ddqn'):
        self.q_network = q_network
        self.environment = environment
        self.replay_memory = None

        # book keeping
        self.name = name
        self.current_step = 0
        self.save_path = os.path.join('checkpoints', name + '.pkl')
        self.logdir = os.path.join('runs', name)

    def calculate_epsilon(self, epsilon_max, epsilon_min, decay_rate):
        """
        calculates epsilon value given steps done and speed of decay
        """
        epsilon = epsilon_min + (epsilon_max - epsilon_min) * \
                  math.exp(-decay_rate * self.current_step)
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

    def learn(self, num_steps, batch_size=32, capacity=500000, lr=2.5e-4,
              epsilon_max=0.9, epsilon_min=0.05, decay_rate=1e-5,
              checkpoint_interval=50000, initial_memory=50000, gamma=0.99):
        cudnn.benchmark = True
        self.replay_memory = ReplayMemory(capacity)

        if len(self.replay_memory) < initial_memory:
            print('populating replay memory...')
            self.prime_replay_memory(initial_memory)

        writer = SummaryWriter(self.logdir)
        optimizer = Adam(self.q_network.parameters(), lr=lr)
        criterion = nn.SmoothL1Loss()
        steps = 0
        pbar = tqdm(total=num_steps)

        while steps <= num_steps:
            state = self.environment.reset()
            total_reward = 0
            while True:
                epsilon = self.calculate_epsilon(epsilon_max, epsilon_min, decay_rate)
                action = self.select_action(state, epsilon)  # selection an action
                next_state, reward, done, info = self.environment.step(action)  # carry out action/observe reward
                self.replay_memory.add(state, action, reward, next_state, done)

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
                targets = rewards + (gamma * next_q_values)

                loss = criterion(q_values, targets)
                optimizer.zero_grad()
                loss.backward()

                # gradient clipping
                for param in self.q_network.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

                writer.add_scalar('epsilon', epsilon, self.current_step)

                steps += 1
                total_reward += reward
                self.current_step += 1
                state = next_state  # move to next state
                if steps % checkpoint_interval == 0:
                    self.save_checkpoint()

                pbar.update()
                if done:
                    writer.add_scalar('reward', total_reward, self.current_step)
                    pbar.set_description("last episode reward: {}".format(total_reward))
                    break

        self.environment.close()

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

    def prime_replay_memory(self, steps):
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

    def load_agent(self, name):
        checkpoint_path = os.path.join('checkpoints', name + '.pkl')
        checkpoint = torch.load(checkpoint_path)
        self.q_network.load_state_dict(checkpoint['weights'])
        self.current_step = checkpoint['current_step']

    def save_checkpoint(self):
        checkpoint = dict(weights=self.q_network.state_dict(),
                          current_step=self.current_step)
        torch.save(checkpoint, self.save_path)


class DDQAgent(DQAgent):
    """
    Double DeepQ Agent with q_network and target network
    """

    def __init__(self, q_network, target_network, environment, name='ddqn'):

        self.q_network = q_network
        self.target_network = target_network
        self.replay_memory = None
        self.environment = environment

        # book keeping
        self.name = name
        self.current_step = 0
        self.save_path = os.path.join('checkpoints', name + '.pkl')
        self.logdir = os.path.join('runs', name)

    def learn(self, num_steps, batch_size=32, capacity=500000, lr=2.5e-4,
              epsilon_max=0.9, epsilon_min=0.05, decay_rate=1e-5,
              checkpoint_interval=50000, initial_memory=50000, sync_interval=1000, gamma=0.99):
        cudnn.benchmark = True
        self.replay_memory = ReplayMemory(capacity)

        if len(self.replay_memory) < initial_memory:
            print('populating replay memory...')
            self.prime_replay_memory(initial_memory)

        writer = SummaryWriter(self.logdir)
        optimizer = Adam(self.q_network.parameters(), lr=lr)
        criterion = nn.SmoothL1Loss()
        steps = 0
        pbar = tqdm(total=num_steps)

        while steps <= num_steps:
            state = self.environment.reset()
            total_reward = 0
            while True:
                epsilon = self.calculate_epsilon(epsilon_max, epsilon_min, decay_rate)
                action = self.select_action(state, epsilon)  # selection an action
                next_state, reward, done, info = self.environment.step(action)  # carry out action/observe reward
                self.replay_memory.add(state, action, reward, next_state, done)

                states, actions, rewards, next_states, done_mask = self.replay_memory.sample(batch_size)

                # prepare batch
                states = Variable(states).cuda()
                next_states = Variable(next_states).cuda()
                rewards = Variable(rewards).cuda()
                done_mask = Variable(done_mask).cuda()

                q_values = self.q_network(states)[
                    range(len(actions)), actions]  # select only Q values for actions we took

                target_actions = self.q_network(next_states).max(dim=1)[1]
                next_q_values = self.target_network(next_states)[
                                    range(len(target_actions)), target_actions].detach() * done_mask
                # calculate targets = rewards + (gamma * next_Q_values)
                targets = rewards + (gamma * next_q_values)

                loss = criterion(q_values, targets)
                optimizer.zero_grad()
                loss.backward()

                # gradient clipping
                for param in self.q_network.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

                writer.add_scalar('epsilon', epsilon, self.current_step)

                steps += 1
                total_reward += reward
                self.current_step += 1
                state = next_state  # move to next state

                if steps % sync_interval == 0:
                    dqn_params = self.q_network.state_dict()
                    self.target_network.load_state_dict(dqn_params)

                if steps % checkpoint_interval == 0:
                    self.save_checkpoint()

                pbar.update()
                if done:
                    writer.add_scalar('reward', total_reward, self.current_step)
                    pbar.set_description("last episode reward: {}".format(total_reward))
                    break

        self.environment.close()




