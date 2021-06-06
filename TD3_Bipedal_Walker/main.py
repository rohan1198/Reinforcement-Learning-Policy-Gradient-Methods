import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import numpy as np
import gym
from gym.wrappers import Monitor
import pybullet_envs
from gym.spaces import Box
import time
import random
import os
#from lib.agent import Agent
from lib.loss import calc_loss
from lib.experience import ReplayBuffer
from lib.network import QNetwork, Actor



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Twin Delayed Deep Deterministic Policy Gradients')
    
    parser.add_argument('--exp-name', type = str, default = os.path.basename(__file__).rstrip(".py"), help = 'the name of this experiment')
    parser.add_argument('--gym-id', type= str, default = "Walker2DBulletEnv-v0", help = 'the id of the gym environment')
    parser.add_argument('--learning-rate', type = float, default = 3e-4, help = 'the learning rate of the optimizer')
    parser.add_argument('--seed', type = int, default = 2, help = 'seed of the experiment')
    parser.add_argument('--total-timesteps', type = int, default = 1000000, help = 'total timesteps of the experiments')
    parser.add_argument('--cuda', action = "store_true", default = False, help = 'if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type = lambda x:bool(strtobool(x)), default = False, nargs = '?', const = True, help = 'run the script in production mode')
    parser.add_argument('--capture-video', type = lambda x:bool(strtobool(x)), default = False, nargs = '?', const = True, help = 'weather to capture videos of the agent performances')
    parser.add_argument('--wandb-project-name', type = str, default = "TD3", help = "the wandb's project name")
    parser.add_argument('--autotune', type = lambda x:bool(strtobool(x)), default = True, nargs = '?', const = True, help = 'automatic tuning of the entropy coefficient.')

    parser.add_argument('--buffer-size', type = int, default = 2000000, help = 'the replay memory buffer size')
    parser.add_argument('--gamma', type = float, default = 0.99, help = 'the discount factor gamma')
    parser.add_argument('--tau', type = float, default = 0.005, help = "target smoothing coefficient")
    parser.add_argument('--max-grad-norm', type = float, default = 0.5, help = 'the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type = int, default = 256, help = "the batch size of sample from the reply memory")
    parser.add_argument('--policy-noise', type = float, default = 0.2, help = 'the scale of policy noise')
    parser.add_argument('--exploration-noise', type = float, default = 0.1, help = 'the scale of exploration noise')
    parser.add_argument('--learning-starts', type = int, default = 25e3, help = "timestep to start learning")
    parser.add_argument('--policy-frequency', type = int, default = 2, help = 'delays the update of the actor, as per the TD3 paper.')
    parser.add_argument('--noise-clip', type = float, default = 0.5, help = 'noise clip parameter of the Target Policy Smoothing Regularization')
    
    args = parser.parse_args()
    
    if not args.seed:
        args.seed = int(time.time())

    experiment_name = f"{args.gym_id}__{args.exp_name}"
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % ('\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
    
    if args.prod_mode:
        import wandb
        wandb.init(project = args.wandb_project_name, sync_tensorboard = True, config = vars(args), name = experiment_name, monitor_gym = True, save_code = True)
        writer = SummaryWriter(f"/tmp/{experiment_name}")


    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    if device == torch.device("cuda"):
        print(torch.cuda.get_device_properties(device))
    else:
        print("Running on CPU!")
    print("\n")

    env = gym.make(args.gym_id)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    assert isinstance(env.action_space, Box), "only continuous action space is supported"
    if args.capture_video:
        env = Monitor(env, f'videos/{experiment_name}', force = True)

    max_action = float(env.action_space.high[0])
    rb = ReplayBuffer(args.buffer_size)
    
    actor = Actor(env.observation_space.shape, env.action_space.shape[0]).to(device)
    target_actor = Actor(env.observation_space.shape, env.action_space.shape[0]).to(device)
    target_actor.load_state_dict(actor.state_dict())

    qf1 = QNetwork(env.observation_space.shape, env.action_space.shape[0]).to(device)
    qf1_target = QNetwork(env.observation_space.shape, env.action_space.shape[0]).to(device)
    qf1_target.load_state_dict(qf1.state_dict())

    qf2 = QNetwork(env.observation_space.shape, env.action_space.shape[0]).to(device)
    qf2_target = QNetwork(env.observation_space.shape, env.action_space.shape[0]).to(device)
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr = args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr = args.learning_rate)

    loss_fn = nn.MSELoss()

    obs = env.reset()
    episode_reward, episode_length = 0., 0
    global_episode = 0
    done = False
    best_mean_reward = None
    total_rewards = []

    for global_step in range(1, args.total_timesteps + 1):
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            action = actor.forward(obs.reshape((1,) + obs.shape), device)
            action = (action.tolist()[0] + np.random.normal(0, max_action * args.exploration_noise, size = env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

        next_obs, reward, done, info = env.step(action)
        rb.store(obs, action, reward, next_obs, done)
        
        episode_reward += reward
        total_rewards.append(episode_reward)
        mean_reward = np.mean(total_rewards[-100:])
        episode_length += 1

        if best_mean_reward is None or best_mean_reward < mean_reward:
            torch.save(actor.state_dict(), "actor-" + args.exp_name + "-best.dat")
            torch.save(qf1.state_dict(), "q1-" + args.exp_name + "-best.dat")
            torch.save(qf2.state_dict(), "q2-" + args.exp_name + "-best.dat")

        if global_step > args.learning_starts:
            qf1_loss, qf2_loss = calc_loss(rb, args.batch_size, env, action, args.policy_noise, args.noise_clip, actor, qf1, qf2, target_actor, qf1_target, qf2_target, actor_optimizer,
                                                       q_optimizer, args.gamma, args.tau, args.max_grad_norm, global_step, args.policy_frequency, writer, device)

            writer.add_scalar("losses/q_value_1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/q_value_2_loss", qf2_loss.item(), global_step)

        obs = next_obs

        if done:
            global_episode += 1
            print(f"Episode: {global_episode} | Step: {global_step} | Ep. Reward: {episode_reward:.4f} | Mean Reward: {mean_reward:.4f}")
            writer.add_scalar("charts/episode_reward", episode_reward, global_step)
            writer.add_scalar("charts/episode_length", episode_length, global_step)
            writer.add_scalar("charts/mean_reward_100", mean_reward, global_step)
            obs, episode_reward = env.reset(), 0.
            done, episode_length = False, 0

    env.close()
    writer.close()
