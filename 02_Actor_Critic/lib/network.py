import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv




class ActorCriticNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorCriticNet, self).__init__()

        self.fc_a1 = nn.Linear(input_shape[0], 512)
        self.fc_a2 = nn.Linear(512, n_actions)

        self.fc_v1 = nn.Linear(input_shape[0], 512)
        self.fc_v2 = nn.Linear(512, 1)

        self.saved_actions = []
        self.rewards = []

    
    def forward(self, dims):
        adv = F.relu(self.fc_a1(dims))
        adv = self.fc_a2(adv)

        val = F.relu(self.fc_v1(dims))
        val = self.fc_v2(val)

        return torch.softmax(adv, dim = -1), val