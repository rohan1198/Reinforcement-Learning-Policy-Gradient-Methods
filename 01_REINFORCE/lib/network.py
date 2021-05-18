import torch
import torch.nn as nn
import torch.nn.functional as F



class PolicyNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(input_shape[0], 256)
        self.drop = nn.Dropout(p = 0.4)
        self.fc2 = nn.Linear(256, n_actions)

        self.saved_log_probs = []
        self.rewards = []

    
    def forward(self, dims):
        x = F.relu(self.fc1(dims))
        x = self.drop(x)
        x = torch.softmax(self.fc2(x), dim = 1)
        return x