import random
from utils import process_state
import torch


class ReplayMemory:
    """
    samples are stored as ('state', 'action', 'next_state', 'reward', done)
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.samples = []
        self.insert_location = 0

    def add(self, state, action, reward, next_state, done):
        sample = (state, action, reward, next_state, done)
        if self.insert_location >= len(self.samples):
            self.samples.append(sample)
        else:
            self.samples[self.insert_location] = sample  # assignment is O(1) for lists
        # walk insertion point through list
        self.insert_location = (self.insert_location + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.samples))
        batch = random.sample(self.samples, batch_size)
        return self.prepare_batch(batch)

    def prepare_batch(self, batch):
        """
        Transposes and pre-processes batch of transitions into batches of torch tensors
            batch: list of transitions [[s, a, r, s2, done],
                                        [s, a, r, s2, done]]

        Returns: [s], [a], [r], [s2], [done_mask]
        """
        states, actions, rewards, next_states, done_mask = [], [], [], [], []
        for state, action, reward, next_state, done in batch:
            states.append(process_state(state))
            actions.append(action)
            rewards.append(reward)
            next_states.append(process_state(next_state))
            done_mask.append(1 - done)  # turn True values into zero for mask
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        rewards = torch.FloatTensor(rewards)
        done_mask = torch.FloatTensor(done_mask)
        return states, actions, rewards, next_states, done_mask

    def __len__(self):
        return len(self.samples)


