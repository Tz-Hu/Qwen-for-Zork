import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from config import (
    LOG_DIR, SAVE_DIR, device, PPO_LR, GAMMA_DISCNT, GAE_LAMBDA, MAX_SIZE,
    MAX_EPISODES, EPISODE_STEPS, BATCH_SIZE, PPO_EPOCHS, PPO_CLIP_EPS,
    MAX_GRAD_NORM
)
from environment import ZorkEnvWrapper
from ppo_agent import PPOAgent, PPOActor, PPOCritic 
from ppo_buffer import PPOBuffer

def train():

# --------------------------
# 1. 初始化阶段
# --------------------------
    # 初始化 Agent, 内含 Actor 和 Critic
    agent = PPOAgent(device)
    critic = agent.critic.to(device)
    actor = agent.actor.to(device)
    env = ZorkEnvWrapper(agent)
    # 优化器
    optimizer = Adam(
    filter(lambda p: p.requires_grad, list(actor.parameters()) + list(critic.parameters())),
    lr=PPO_LR
    )

    # 初始化 经验回放缓冲区
    buffer = PPOBuffer(gae_lambda=GAE_LAMBDA, max_size=MAX_SIZE)

# --------------------------
# 2. 训练循环：Episode
# --------------------------

    for ep in range(1, MAX_EPISODES + 1):
        # 重置环境、初始化变量
# 未修改： 用于及时反馈，优化局内表现
        episode_history = []    # 存储本回合的动作历史，用于增强 prompt

        buffer.reset()          # 重置经验回放缓冲区 （采用逐局更新
        obs, info = env.reset()  # 重置环境，获取初始状态和信息

        log_path = os.path.join(LOG_DIR, f"episode_{ep:03d}.txt")
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"Episode {ep:03d} 开始\n")
            log_file.write("=" * 80 + "\n")
            discnt_coeff = 1
            discnt_return =  0

            for time in range(1, 1 + EPISODE_STEPS):
                valid_actions = env.get_valid_actions()
                obs = env.obs

                # 选择 action
                action = agent.choose_action(obs, valid_actions)

                # 环境执行 动作, 并更新 buffer 需要的信息
                next_obs, step_score, done, info = env.step(action)
                # world_changed = env.change_world_reward()
                step_reward = env.my_step_reward(action)              
                state_value = critic.evaluate_values(obs)
                old_logprob = actor.compute_logprob(obs, valid_actions)
                # 写入bffer
                buffer.add(obs, step_reward, state_value, done, old_logprob, episode_history, valid_actions)
                # episode_history 晚于 buffer 更新
                # episode_history.append(f"{action} -> reward: {env.step_reward(action)}")
                
# 结合通关时间再加奖励！结合死亡时间再惩罚！
                # 判断 死亡/胜利
                if done:
                    # 胜利 奖励 100
                    if env.victory():
                        # state_value = 100
                        log_file.write(f"\n[ WIN ] Moves numbers:{info['moves']}\tTotal Score: {env.get_score()}\n")
                    # 死亡 惩罚 50
                    else:
                        # state_value = 50
                        log_file.write(f"\n[ DIED] Moves numbers:{info['moves']}\tTotal Score: {env.get_score()}\n")
                    log_file.write("=" * 90 + "\n")
                    break

                # 如果没完成 -- 记录 日志
                log_file.write(f"Time {time} | StepScore: {step_score}")
                log_file.write(f"执行动作: {action}\n")
                log_file.write(f"反馈观察: {obs}")
                log_file.write(f"Step Reward: {step_reward}\n")
                log_file.write(f"Game Step Score: {step_score}\t Game Total Score: {info['score']}\n")
                log_file.write("=" * 80 + "\n")

                print(f"[E {ep}  t= {time}] | Action: {action.ljust(20)} Reward: {step_reward:.3f}\tScore: {info['score']}") 

                discnt_coeff *= GAMMA_DISCNT
                discnt_return += discnt_coeff * step_reward
# --------------------------
# 3. PPO 更新阶段
# --------------------------
            # obs_list = buffer.observations
            # values = critic.evaluate_values(obs_list)
            # next_value = critic.evaluate_vales([obs_list[-1]]) if not buffer.done[-1] else torch.tensor([0,0], device=values.device)
            
            # compute GAE
            values = critic.evaluate_values(obs)
            next_value = critic.evaluate_values(obs[-1:]) if not done else torch.tensor([0.0], device=values.device)
            advantage  = agent.compute_advantage(buffer.rewards, values, next_value)

# 计算 returns
            returns = []
            for reward, done in zip(reversed(buffer.rewards), reversed(buffer.done)):
                if done:
                    returns.insert(0, reward)  # 如果是终止状态，直接用当前奖励
                else:
                    returns.insert(0, reward + GAMMA_DISCNT * next_value)
                next_value = returns[0]

            buffer.advantages = advantage
            buffer.returns = returns
            buffer.next_value = next_value

            # buffer 采样
            for epoch in range(PPO_EPOCHS):
                total_loss = 0
                for batch in buffer.get_batches(BATCH_SIZE):
                    # with autocast('cuda'):      # 混合精度
                    loss =  agent.compute_loss(batch)
                    total_loss += loss

                # 优化器步骤
                print(f"epoch {epoch} , loss = {total_loss}")
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), MAX_GRAD_NORM)
                optimizer.step()
            

            # 保存检查点
            # checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_ep{ep}.pt")
            # agent.save_policy(checkpoint_path)
            print(f"discnt Return : {discnt_return}")
            torch.cuda.empty_cache()

    # 保存最终模型``
    # agent.save_policy(os.path.join(SAVE_DIR, "final_policy.pt"))
    # print()
        return_log_path = os.path.join(LOG_DIR, "return_log.txt")
        with open(return_log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"Episode {ep:03d} , Move {info['moves']}, discnt Return : {discnt_return}")
    print("Training complete.")
