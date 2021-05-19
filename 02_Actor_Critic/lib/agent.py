from collections import namedtuple
import torch
from torch.distributions import Categorical



SavedAction = namedtuple("SavedAction", field_names = ["log_prob", "value"])


def select_action(net, state):
    state = torch.from_numpy(state).float()
    probs, state_value = net(state)
    m = Categorical(probs)
    action = m.sample()
    net.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()
