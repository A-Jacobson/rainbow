import torch.nn.functional as F
from torch import nn


class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, state):
        x = F.relu(self.conv1(state))  # (20, 20, 32)
        x = F.relu(self.conv2(x))  # (9, 9, 64)
        x = F.relu(self.conv3(x))  # (7, 7, 64)
        x = x.view(x.size(0), -1)  # flatten (7*7*64)
        x = F.relu(self.fc4(x))  # (512)
        q_values = self.fc5(x)  # (num_actions) q value for each action
        return q_values


class DQNbn(nn.Module):
    def __init__(self, num_actions):
        super(DQNbn, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.linear1 = nn.Linear(2592, 256)
        self.out = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.linear1(x))
        return self.out(x)
