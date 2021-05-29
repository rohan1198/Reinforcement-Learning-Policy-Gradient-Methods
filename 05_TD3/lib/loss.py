import torch
import torch.nn as nn




def calc_loss(buffer, batch_size, env, action, policy_noise, noise_clip, actor, qf1, qf2, target_actor, qf1_target, qf2_target, actor_optimizer, q_optimizer, gamma, tau, max_grad_norm, global_step, policy_frequency, device):
    s_obs, s_actions, s_rewards, s_next_obses, s_dones = buffer.sample(batch_size)

    with torch.no_grad():
        clipped_noise = (torch.randn_like(torch.Tensor(action)) + policy_noise).clamp(-noise_clip, noise_clip)

        next_state_actions = (target_actor.forward(s_next_obses, device) + clipped_noise.to(device)).clamp(env.action_space.low[0], env.action_space.high[0])
        qf1_next_target = qf1_target.forward(s_next_obses, next_state_actions, device)
        qf2_next_target = qf2_target.forward(s_next_obses, next_state_actions, device)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
        next_q_value = torch.Tensor(s_rewards).to(device) + (1 - torch.Tensor(s_dones).to(device)) * gamma * (min_qf_next_target).view(-1)

    qf1_a_values = qf1.forward(s_obs, torch.Tensor(s_actions).to(device), device).view(-1)
    qf2_a_values = qf2.forward(s_obs, torch.Tensor(s_actions).to(device), device).view(-1)
    qf1_loss = nn.MSELoss()(qf1_a_values, next_q_value)
    qf2_loss = nn.MSELoss()(qf2_a_values, next_q_value)

    q_optimizer.zero_grad()
    qf1_loss.backward()
    qf2_loss.backward()
    nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), max_grad_norm)
    q_optimizer.step()

    if global_step % policy_frequency == 0:
        actor_loss = -qf1.forward(s_obs, actor.forward(s_obs, device), device).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(list(actor.parameters()), max_grad_norm)
        actor_optimizer.step()

        for param, target_param in zip(actor.parameters(), target_actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    return qf1_loss, qf2_loss, actor_loss