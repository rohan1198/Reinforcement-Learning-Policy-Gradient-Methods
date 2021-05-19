import torch
import torch.nn.functional as F


def finish_episode(net, gamma):
    R = 0
    saved_actions = net.saved_actions
    policy_loss = []
    value_loss = []
    returns = []

    for r in net.rewards[::1]:
        R = R * gamma + r
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        policy_loss.append(-log_prob * advantage)
        value_loss.append(F.smooth_l1_loss(value, torch.tensor([R])))

    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()

    return loss