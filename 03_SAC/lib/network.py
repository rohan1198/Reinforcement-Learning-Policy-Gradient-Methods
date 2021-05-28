import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Policy(nn.Module):
    def __init__(self, input_dims, n_actions, layer_init, fc1_dims = 300, fc2_dims = 400):
        super(Policy, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, n_actions)

        self.apply(layer_init)


    def forward(self, dims, device):
        x = torch.Tensor(dims).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.mu(x)
        sigma = torch.tanh(self.sigma(x))
        sigma = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (sigma + 1)

        return mu, sigma

    

class SoftQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, layer_init, fc1_dims = 300, fc2_dims = 400):
        super(SoftQNetwork, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(self.input_dims + self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.apply(layer_init)


    def forward(self, dims, action, device):
        x = torch.Tensor(dims).to(device)
        x = F.relu(self.fc1(torch.cat([x, action], dim = 1)))
        x = F.relu(self.fc2(x))
        x = self.q(x)

        return x
    