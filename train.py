import os
import torch
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from config import (
    LOG_DIR, SAVE_DIR, device, PPO_LR,
    MAX_EPISODES, EPISODE_STEPS, BATCH_SIZE, PPO_EPOCHS,
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
    os.remove(return_log_path) if os.path.exists(return_log_path) else None
    buffer = PPOBuffer()
    for ep in range(1, MAX_EPISODES + 1):
        agent.reset()
        obs, info = zork_env.reset()
        log_path = os.path.join(LOG_DIR, f"episode_{ep:03d}.txt")
        feedback = ''
        step_reward = 0
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"Episode {ep:03d} 开始\n")
            log_file.write("=" * 80 + "\n")
            print("=" * 80 )
            for time in range(1, 1 + EPISODE_STEPS):
                valid_actions = zork_env.get_valid_actions()
                buffer.add_valid_action(valid_actions)
                obs = filter_copyright(zork_env.obs) if time < 5 else zork_env.obs
                buffer.add_obs(obs)
                action, old_logprob = actor.generate_action_logprob(obs, valid_actions,feedback, step_reward)
                buffer.add_feedback(feedback)
                buffer.add_old_logprob(old_logprob.detach())
                _, step_score, done, info = zork_env.step(action)
                step_reward = zork_env.step_reward(action)
                buffer.add_step_reward(step_reward)
                feedback = zork_env.feedback
                print(f"反馈: {feedback}")
                log_file.write(f"Time {time} | StepScore: {step_score}")
                log_file.write(f"观察: {obs}")
                log_file.write(f"执行动作: {action}\n")
                log_file.write(f"Step Reward: {step_reward}\n")
                log_file.write(f"Game Step Score: {step_score}\t Game Total Score: {info['score']}\n")
                log_file.write("=" * 80 + "\n")
                print("=" * 80 )
                print(f"[E {ep}  t= {time}] \n观察: {obs}\nAction: {action}\nStepReward: {step_reward:.3f}\tStepScore: {step_score}\tTotalScore: {info['score']}")
                if done:
                    if zork_env.victory():
                        log_file.write(f"\n[ WIN ] Moves numbers:{info['moves']}\tTotal Score: {zork_env.get_score()}\n")
                    else:
                        log_file.write(f"\n[ DIED] Moves numbers:{info['moves']}\tTotal Score: {zork_env.get_score()}\n")
                    log_file.write("=" * 90 + "\n")
                    buffer.add_state_value(0)
                    break
                else:
                    next_value = critic.evaluate_value(obs)
                state_value = next_value
                buffer.add_state_value(state_value)
                buffer.add_state_value(state_value.detach())

            buffer.next_value = next_value
            scaler = GradScaler()
            print(f"\nEpisode {ep} finished, total score: {zork_env.get_score()}\nNow begin loss computation:")
            for epoch in range(PPO_EPOCHS):
                print(f"\nepoch{epoch+1}/{PPO_EPOCHS}")
                i = 1
                for batch in buffer.get_batches(BATCH_SIZE):
                    optimizer.zero_grad()
                    with autocast():
                        loss = agent.compute_loss(batch)
                    print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB | Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
                    print("-"*50+f"batch {i}/{BATCH_SIZE}:, loss = {loss.item():.4f}")
                    log_file.write(f"\nepoch{epoch+1}/{PPO_EPOCHS}\n")
                    log_file.write(f"batch {i}/{EPISODE_STEPS/BATCH_SIZE}, loss = {loss:.3f}\n")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    del loss
                    torch.cuda.empty_cache()
                    i +=1
            discnt_returns = agent.compute_total_return(buffer.step_rewards)
            with open(return_log_path, "a", encoding="utf-8") as total_log_file:
                total_log_file.write(f"Ep {ep:03d} ,Move {info['moves']}\tdiscnt Return : {discnt_returns:.3f}\tscore: {zork_env.get_score()}\n")
        buffer.reset()
    print("Training complete.")
