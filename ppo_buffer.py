import torch

# PPOBuffer 参数
MAX_SIZE = 1000
GAE_LAMBDA = 0.95
GAMMA = 0.99

class PPOBuffer:
    def __init__(self, gae_lambda=GAE_LAMBDA, max_size=MAX_SIZE):
        self.gae_lambda = gae_lambda
        self.max_size = max_size
        self.reset()

    def reset(self):
        self.state_values = []  
        self.observations = []
        self.next_value = []
        self.rewards = []
        self.done = []
        self.old_logprobs = []
        self.advantages = []
        self.returns = []
        self.history = []
        self.valid_actions = []

    def add(self, obs, reward, state_value, done, old_logprob, history, valid_actions):
        self.observations.append(obs)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.done.append(int(done)) # done = 1 : 当前步骤导致环境终止
        self.old_logprobs.append(old_logprob)
        self.history.append(history.copy())
        self.valid_actions.append(valid_actions.copy())
    
    def get_batches(self, batch_size):
        indices = torch.randperm(len(self.observations))
        shuffled_obs = [self.observations[i] for i in indices]
        # shuffled_next_value = [self.next_value[i] for i in indices]
        shuffled_next_value = self.next_value
        shuffled_rewards = [self.rewards[i] for i in indices]
        shuffled_dones = [self.done[i] for i in indices]
        shuffled_old_logprobs = [self.old_logprobs[i] for i in indices]
        shuffled_advantages = [self.advantages[i] for i in indices]
        shuffled_returns = [self.returns[i] for i in indices]
        shuffled_history = [self.history[i] for i in indices]
        shuffled_valid_actions = [self.valid_actions[i] for i in indices]

        # 按固定大小切分数据
        total = len(shuffled_obs)
        for i in range(0, total, batch_size):
            yield Batch(
                observations=shuffled_obs[i:i+batch_size],
                next_value=shuffled_next_value,
                # next_value=shuffled_next_value[i:i+batch_size],
                rewards=shuffled_rewards[i:i+batch_size],
                dones=shuffled_dones[i:i+batch_size],
                old_logprobs=shuffled_old_logprobs[i:i+batch_size],
                advantages=torch.tensor(shuffled_advantages[i:i+batch_size], device="cuda", dtype=torch.float32),
                returns=torch.tensor(shuffled_returns[i:i+batch_size], device="cuda", dtype=torch.float32),
                history=shuffled_history[i:i+batch_size],
                valid_actions=shuffled_valid_actions[i:i+batch_size]
            )

class Batch:
    # def __init__(self, observations, rewards, dones, old_logprobs, advantages, returns, history, valid_actions):
    def __init__(self, observations, next_value, rewards, dones, old_logprobs, advantages, returns, history, valid_actions):
        self.observations = observations
        self.next_value = next_value
        self.rewards = rewards
        self.dones = dones
        self.old_logprobs = old_logprobs
        self.advantages = advantages
        self.returns = returns
        self.history = history
        self.valid_actions = valid_actions
