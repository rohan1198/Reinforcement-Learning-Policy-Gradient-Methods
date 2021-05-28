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
from lib.agent import Agent
from lib.loss import calc_loss
from lib.experience import ReplayBuffer
from lib.network import Policy, SoftQNetwork



def layer_init(layer, weight_gain = 1, bias_const = 0):
    if isinstance(layer, nn.Linear):
        if args.weights_init == "xavier":
            torch.nn.init.xavier_uniform_(layer.weight, gain = weight_gain)
        elif args.weights_init == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight, gain = weight_gain)
        if args.bias_init == "zeros":
            torch.nn.init.constant_(layer.bias, bias_const)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Soft Actor Critic')
    
    parser.add_argument('--exp-name', type = str, default = os.path.basename(__file__).rstrip(".py"), help = 'the name of this experiment')
    parser.add_argument('--gym-id', type= str, default = "LunarLanderContinuous-v2", help = 'the id of the gym environment')
    parser.add_argument('--learning-rate', type = float, default = 7e-4, help = 'the learning rate of the optimizer')
    parser.add_argument('--seed', type = int, default = 2, help = 'seed of the experiment')
    parser.add_argument('--total-timesteps', type = int, default = 1000000, help = 'total timesteps of the experiments')
    parser.add_argument('--cuda', action = "store_true", default = False, help = 'if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type = lambda x:bool(strtobool(x)), default = False, nargs = '?', const = True, help = 'run the script in production mode')
    parser.add_argument('--capture-video', type = lambda x:bool(strtobool(x)), default = False, nargs = '?', const = True, help = 'weather to capture videos of the agent performances')
    parser.add_argument('--wandb-project-name', type = str, default = "Soft-Actor-Critic", help = "the wandb's project name")
    parser.add_argument('--autotune', type = lambda x:bool(strtobool(x)), default = True, nargs = '?', const = True, help = 'automatic tuning of the entropy coefficient.')

    parser.add_argument('--buffer-size', type = int, default = 100000, help = 'the replay memory buffer size')
    parser.add_argument('--gamma', type = float, default = 0.99, help = 'the discount factor gamma')
    parser.add_argument('--target-network-frequency', type = int, default = 2, help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type = float, default = 0.5, help = 'the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type = int, default = 256, help = "the batch size of sample from the reply memory")
    parser.add_argument('--tau', type = float, default = 0.005, help = "target smoothing coefficient")
    parser.add_argument('--alpha', type = float, default = 0.2, help = "Entropy regularization coefficient")
    parser.add_argument('--learning-starts', type = int, default = 5e3, help = "timestep to start learning")


    parser.add_argument('--policy-lr', type = float, default = 3e-4, help = 'the learning rate of the policy network optimizer')
    parser.add_argument('--q-lr', type = float, default = 1e-3, help = 'the learning rate of the Q network network optimizer')
    parser.add_argument('--policy-frequency', type = int, default = 1, help = 'delays the update of the actor, as per the TD3 paper.')
    parser.add_argument('--weights-init', default = 'xavier', const = 'xavier', nargs = '?', choices = ['xavier', "orthogonal", 'uniform'], help = 'weight initialization scheme for the neural networks.')
    parser.add_argument('--bias-init', default = 'zeros', const = 'xavier', nargs = '?', choices = ['zeros', 'uniform'], help = 'weight initialization scheme for the neural networks.')

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

    rb = ReplayBuffer(args.buffer_size)
    agent = Agent(env, device)\

    pg = Policy(env.observation_space.shape[0], env.action_space.shape[0], layer_init).to(device)
    qf1 = SoftQNetwork(env.observation_space.shape[0], env.action_space.shape[0], layer_init).to(device)
    qf2 = SoftQNetwork(env.observation_space.shape[0], env.action_space.shape[0], layer_init).to(device)

    qf1_target = SoftQNetwork(env.observation_space.shape[0], env.action_space.shape[0], layer_init).to(device)
    qf1_target.load_state_dict(qf1.state_dict())

    qf2_target = SoftQNetwork(env.observation_space.shape[0], env.action_space.shape[0], layer_init).to(device)
    qf2_target.load_state_dict(qf2.state_dict())

    values_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr = args.q_lr)
    policy_optimizer = optim.Adam(list(pg.parameters()), lr = args.policy_lr)

    if args.autotune:
        target_entropy = - torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad = True, device = device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr = args.q_lr)
    else:
        alpha = args.alpha

    global_episode = 0
    obs, done = env.reset(), False
    episode_reward, episode_length = 0.,0
    best_mean_reward = None
    total_rewards = []

    for global_step in range(1, args.total_timesteps + 1):
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            action, _, _ = agent.get_action([obs], pg, device)
            action = action.tolist()[0]

        next_obs, reward, done, _ = env.step(action)
        rb.store(obs, action, reward, next_obs, done)
        episode_reward += reward
        total_rewards.append(episode_reward)
        mean_reward = np.mean(total_rewards[-100:])
        episode_length += 1
        obs = np.array(next_obs)

        if best_mean_reward is None or best_mean_reward < mean_reward:
            torch.save(pg.state_dict(), "policy-" + args.exp_name + "-best.dat")
            torch.save(qf1.state_dict(), "q1-" + args.exp_name + "-best.dat")
            torch.save(qf2.state_dict(), "q2-" + args.exp_name + "-best.dat")

        if len(rb.memory) > args.batch_size:
            qf1_loss, qf2_loss, qf_loss, policy_loss, alpha_loss = calc_loss(rb, args.batch_size, agent, pg, qf1, qf2, qf1_target, qf2_target, alpha, log_alpha, 
                                                                             args.gamma, args.tau, values_optimizer, policy_optimizer, global_step, 
                                                                             args.policy_frequency, args.autotune, target_entropy, a_optimizer, 
                                                                             args.target_network_frequency, device)

            writer.add_scalar("losses/soft_q_value_1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/soft_q_value_2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
            writer.add_scalar("losses/alpha", alpha, global_step)
            if args.autotune:
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
    
        if done:
            global_episode += 1
            writer.add_scalar("charts/episode_reward", episode_reward, global_step)
            writer.add_scalar("charts/episode_length", episode_length, global_step)
            writer.add_scalar("charts/mean_reward_100", mean_reward, global_step)
            
            print(f"Episode: {global_episode} | Step: {global_step} | Ep. Reward: {episode_reward:.4f} | Mean Reward: {mean_reward:.4f}")

            obs, done = env.reset(), False
            episode_reward, episode_length = 0., 0

    writer.close()
    env.close()
