import torch
import torch.nn as nn


def calc_loss(buffer, batch_size, agent, pg, qf1, qf2, qf1_target, qf2_target, alpha, log_alpha, gamma, tau, values_optimizer, policy_optimizer, 
              global_step, policy_frequency, autotune, target_entropy, a_optimizer, target_network_frequency, device):
    
    s_obs, s_actions, s_rewards, s_next_obses, s_dones = buffer.sample(batch_size)
        
    with torch.no_grad():
        next_state_actions, next_state_log_pi, _ = agent.get_action(s_next_obses, pg, device)
        qf1_next_target = qf1_target.forward(s_next_obses, next_state_actions, device)
        qf2_next_target = qf2_target.forward(s_next_obses, next_state_actions, device)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
        next_q_value = torch.Tensor(s_rewards).to(device) + (1 - torch.Tensor(s_dones).to(device)) * gamma * (min_qf_next_target).view(-1)

    qf1_a_values = qf1.forward(s_obs, torch.Tensor(s_actions).to(device), device).view(-1)
    qf2_a_values = qf2.forward(s_obs, torch.Tensor(s_actions).to(device), device).view(-1)
    qf1_loss = nn.MSELoss()(qf1_a_values, next_q_value)
    qf2_loss = nn.MSELoss()(qf2_a_values, next_q_value)
    qf_loss = (qf1_loss + qf2_loss) / 2

    values_optimizer.zero_grad()
    qf_loss.backward()
    values_optimizer.step()

    if global_step % policy_frequency == 0:
        for _ in range(policy_frequency): 
            pi, log_pi, _ = agent.get_action(s_obs, pg, device)
            qf1_pi = qf1.forward(s_obs, pi, device)
            qf2_pi = qf2.forward(s_obs, pi, device)
            min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
            policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            if autotune:
                with torch.no_grad():
                    _, log_pi, _ = agent.get_action(s_obs, pg, device)
                alpha_loss = ( -log_alpha * (log_pi + target_entropy)).mean()

                a_optimizer.zero_grad()
                alpha_loss.backward()
                a_optimizer.step()
                alpha = log_alpha.exp().item()

    if global_step % target_network_frequency == 0:
        for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    return qf1_loss, qf2_loss, qf_loss, policy_loss, alpha_loss