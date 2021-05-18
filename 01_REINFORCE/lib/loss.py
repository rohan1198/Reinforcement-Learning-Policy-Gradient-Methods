import torch


def finish_episode(net, gamma):
    R = 0
    policy_loss = []
    returns = []

    for r in net.rewards[::1]:
        R = R * gamma + r
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)

    for log_prob, R in zip(net.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)

    loss = torch.cat(policy_loss).sum()

    return loss