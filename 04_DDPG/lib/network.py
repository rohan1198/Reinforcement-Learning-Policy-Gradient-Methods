import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F





class QNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, fc1_dims = 300, fc2_dims = 400):
        super(QNetwork, self).__init__()

        self.input_shape = input_shape
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(self.input_shape[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

    
    def forward(self, dims, action, device):
        x = torch.Tensor(dims).to(device)
        x = F.relu(self.fc1(torch.cat([x, action], dim = 1)))
        x = F.relu(self.fc2(x))
        x = self.q(x)
        return x

    

class Actor(nn.Module):
    def __init__(self, input_shape, n_actions, fc1_dims = 300, fc2_dims = 400):
        super(Actor, self).__init__()

        self.input_shape = input_shape
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(*self.input_shape, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

    
    def forward(self, dims, device):
        x = torch.Tensor(dims).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.mu(x))
        return x