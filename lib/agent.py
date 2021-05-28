import torch
from torch.distributions import Normal




class Agent(object):
    def __init__(self, env, device):
        self.action_scale = torch.FloatTensor((env.action_space.high - env.action_space.low) / 2.).to(device)
        self.action_bias = torch.FloatTensor((env.action_space.high + env.action_space.low) / 2.).to(device)


    def get_action(self, state, net, device):
        mean, log_std = net.forward(state, device)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) +  1e-6)
        log_prob = log_prob.sum(1, keepdim = True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean