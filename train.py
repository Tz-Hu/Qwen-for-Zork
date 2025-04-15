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
from ppo_agent import PPOAgent
from ppo_buffer import PPOBuffer

def train():
    agent = PPOAgent(device)
    critic = agent.critic.to(device)
    actor = agent.actor.to(device)
    env = ZorkEnvWrapper()
    optimizer = Adam(
    filter(lambda p: p.requires_grad, list(actor.parameters()) + list(critic.parameters())),
    lr=PPO_LR
    )
    buffer = PPOBuffer()
    for ep in range(1, MAX_EPISODES + 1):
        agent.reset()
        env.reset()
        buffer.reset()
        log_path = os.path.join(LOG_DIR, f"episode_{ep:03d}.txt")
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"Episode {ep:03d} 开始\n")
            log_file.write("=" * 80 + "\n")
            for time in range(1, 1 + EPISODE_STEPS):
                valid_actions = env.get_valid_actions()
                obs = env.obs
                action, old_logprob = actor.generate_action_logprob(obs, valid_actions)
                buffer.add_old_logprob(old_logprob)
                _, step_score, done, info = env.step(action)
                buffer.add_done(done)
                if done:
                    if env.victory():
                        log_file.write(f"\n[ WIN ] Moves numbers:{info['moves']}\tTotal Score: {env.get_score()}\n")
                    else:
                        log_file.write(f"\n[ DIED] Moves numbers:{info['moves']}\tTotal Score: {env.get_score()}\n")
                    log_file.write("=" * 90 + "\n")
                    next_value = 0
                    break
                else:
                    next_value = critic.evaluate_value(obs)
                state_value = next_value
                buffer.add_state_value(state_value)
# 从这里入手重写一下 environment的奖励， 记得加 feedback
                step_reward = env.my_step_reward(action)
                buffer.add_step_reward(step_reward)

                log_file.write(f"Time {time} | StepScore: {step_score}")
                log_file.write(f"反馈观察: {obs}")
                log_file.write(f"执行动作: {action}\n")
                log_file.write(f"Step Reward: {step_reward}\n")
                log_file.write(f"Game Step Score: {step_score}\t Game Total Score: {info['score']}\n")
                log_file.write("=" * 80 + "\n")
                print("=" * 80 + "\n")
                print(f"[E {ep}  t= {time}] \n 观察: {obs} Action: {action.ljust(20)} StepReward: {step_reward:.3f}\tReturn: 哈哈看不到\tStepScore: {step_score}TotalScore: {info['score']}\n")

            buffer.next_value = next_value
            print(f"next value检查是否出循环被保留: {nexT_value}\n" )

            for epoch in range(PPO_EPOCHS):
                i = 1
                for batch in buffer.get_batches(BATCH_SIZE):
                    loss = agent.compute_loss(batch)
                    print(f"batch {i}:, loss = {loss:.4f}")
                    log_file.write(f"batch {i}:, loss = {loss}")
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), MAX_GRAD_NORM)
                    optimizer.step()
                    i +=1

            # 保存检查点
            # checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_ep{ep}.pt")
            # agent.save_policy(checkpoint_path)
            torch.cuda.empty_cache()

    # 保存最终模型``
    # agent.save_policy(os.path.join(SAVE_DIR, "final_policy.pt"))
    # print()
        return_log_path = os.path.join(LOG_DIR, "return_log.txt")
        with open(return_log_path, "w", encoding="utf-8") as log_file:
# 没write return等信息
            log_file.write(f"Episode {ep:03d} , Move {info['moves']}, discnt Return : {discnt_return}")
    
    print("Training complete.")
