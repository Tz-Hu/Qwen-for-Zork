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

def filter_copyright(text):
    keywords = ["Copyright", "Microsoft Corporation", "GNU General Public License", "Serial number", "ZORK is a registered trademark"]
    lines = text.split('\n')
    filtered = [line for line in lines if not any(kw in line for kw in keywords)]
    return '\n'.join(filtered).strip()

def train():
    agent = PPOAgent(device)
    critic = agent.critic.to(device)
    actor = agent.actor.to(device)
    zork_env = ZorkEnvWrapper()
    optimizer = Adam(
    filter(lambda p: p.requires_grad, list(actor.parameters()) + list(critic.parameters())),
    lr=PPO_LR
    )
    return_log_path = os.path.join(LOG_DIR, "return_log.txt")
    buffer = PPOBuffer()
    for ep in range(1, MAX_EPISODES + 1):
        agent.reset()
        obs, info = zork_env.reset()
        buffer.reset()
        log_path = os.path.join(LOG_DIR, f"episode_{ep:03d}.txt")
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"Episode {ep:03d} 开始\n")
            log_file.write("=" * 80 + "\n")
            print("=" * 80 )
            for time in range(1, 1 + EPISODE_STEPS):
                valid_actions = zork_env.get_valid_actions()
                buffer.add_valid_action(valid_actions)
                obs = filter_copyright(zork_env.obs)
                buffer.add_obs(obs)
                action, old_logprob = actor.generate_action_logprob(obs, valid_actions)
                buffer.add_old_logprob(old_logprob)
                _, step_score, done, info = zork_env.step(action)
                # buffer.add_done(done)
                if done:
                    if zork_env.victory():
                        log_file.write(f"\n[ WIN ] Moves numbers:{info['moves']}\tTotal Score: {zork_env.get_score()}\n")
                    else:
                        log_file.write(f"\n[ DIED] Moves numbers:{info['moves']}\tTotal Score: {zork_env.get_score()}\n")
                    log_file.write("=" * 90 + "\n")
                    next_value = 0
                    break
                else:
                    next_value = critic.evaluate_value(obs)
                state_value = next_value
                buffer.add_state_value(state_value)
# 从这里入手重写一下 environment的奖励， 记得加 feedback
                step_reward = zork_env.my_step_reward(action)
                buffer.add_step_reward(step_reward)

                log_file.write(f"Time {time} | StepScore: {step_score}")
                log_file.write(f"反馈观察: {obs}")
                log_file.write(f"执行动作: {action}\n")
                log_file.write(f"Step Reward: {step_reward}\n")
                log_file.write(f"Game Step Score: {step_score}\t Game Total Score: {info['score']}\n")
                log_file.write("=" * 80 + "\n")
                print("=" * 80 )
                print(f"[E {ep}  t= {time}] \n观察: {obs}\n\nAction: {action}\nStepReward: {step_reward:.3f}\tStepScore: {step_score}\tTotalScore: {info['score']}")

            buffer.next_value = next_value
            for epoch in range(PPO_EPOCHS):
                print("hahahah\n")
                i = 1
                for batch in buffer.get_batches(BATCH_SIZE):
                    print("blahbalh\n")
                    loss = agent.compute_loss(batch)
                    print(f"batch {i}:, loss = {loss:.4f}")
                    log_file.write(f"batch {i}:, loss = {loss}")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), MAX_GRAD_NORM)
                    optimizer.step()
                    i +=1

            discnt_returns = agent.compute_total_return(buffer.step_rewards)
            with open(return_log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"Episode {ep:03d} , Move {info['moves']}, discnt Return : {discnt_returns}\tscore: {zork_env.get_score()}\n")
            torch.cuda.empty_cache()
            # 保存检查点
            # checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_ep{ep}.pt")
            # agent.save_policy(checkpoint_path)

    # 保存最终模型``
    # agent.save_policy(os.path.join(SAVE_DIR, "final_policy.pt"))
    # print()
    print("Training complete.")
