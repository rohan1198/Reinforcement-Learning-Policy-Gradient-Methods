import torch
from torch.distributions import Categorical



class Agent(object):
    def __init__(self, env):
        self.env = env
        #self.reset()

    """    
    def reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0


    def play_step(self, net):
        done_reward = None

        state = torch.from_numpy(self.state).float().unsqueeze(0)
        probs = net(state)
        m = Categorical(probs)
        actions = m.sample()
        net.saved_log_probs.append(m.log_prob(actions))
        action = actions.item()

        new_state, reward, done, _ = self.env.step(action)

        self.total_reward += reward
        net.rewards.append(reward)

        self.state = new_state

        if done:
            done_reward = self.total_reward
            self.reset()
        
        return done_reward
    """

    def select_action(self, net, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = net(state)
        m = Categorical(probs)
        action = m.sample()
        net.saved_log_probs.append(m.log_prob(action))
        return action.item()
