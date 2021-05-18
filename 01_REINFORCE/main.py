import numpy as np
import gym
import torch
import torch.optim as optim
import argparse
import time

from itertools import count
from lib.network import PolicyNet
from lib.agent import Agent
from lib.loss import finish_episode
from tensorboardX import SummaryWriter

LEARNING_RATE = 0.001
STOP_BOUNDARY = 195
running_reward = 10
env_name = "CartPole-v0"


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type = float, default = 0.99, help = "Discount Factor")
    parser.add_argument("--render", action = "store_true", default = False, help = "Render the environment")
    parser.add_argument("--env", default = env_name, help = "Environment name")
    args = parser.parse_args()

    env = gym.make(args.env)

    net = PolicyNet(env.observation_space.shape, env.action_space.n)
    optimizer = optim.Adam(net.parameters(), lr= LEARNING_RATE)
    agent = Agent(env)

    writer = SummaryWriter(comment = "-REINFORCE")
    print(net)

    best_mean_reward = None

    for i in count(1):
        state = env.reset()
        ep_reward = 0
        total_rewards = []

        for t in range(10000):
            action = agent.select_action(net, state)
            state, reward, done, _ = env.step(action)

            if args.render:
                env.render()
            
            net.rewards.append(reward)
            ep_reward += reward            

            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        optimizer.zero_grad()
        loss = finish_episode(net, args.gamma)
        loss.backward()
        optimizer.step()

        del net.rewards[:]
        del net.saved_log_probs[:]

        
        print(f"episode {i} | Last reward: {ep_reward} | Average Reward: {running_reward}")
        writer.add_scalar("running_reward", running_reward, i)

        if best_mean_reward is None or best_mean_reward < running_reward:
                torch.save(net.state_dict(), args.env + "-best.dat")

                if best_mean_reward is not None:
                    print(f"Best mean reward updated {best_mean_reward:.3f} -> {running_reward:.3f} model saved!")
                
                best_mean_reward = running_reward

        if running_reward > STOP_BOUNDARY:
            print(f"Solved! Running reward is {running_reward} and runs to {t} time steps!")
            break
