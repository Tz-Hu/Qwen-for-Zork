import torch

class PPOBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.old_logprobs = []
        self.logprobs = []
        self.actions = []
        self.done = []
        self.step_rewards = []
        self.discnt_returns = []
        self.state_values = []
        self.next_value = None


        self.obs = []
        self.next_value = []
        self.advantages = []
        self.history = []
        self.valid_actions = []

    def add_obs(self, obs):
        self.obs.append(obs)

    def add_step_reward(self, reward):
        self.step_rewards.append(reward)

    def add_state_value(self, state_value):
        self.state_values.append(state_value)

    def add_done(self, done):
        self.done.append(int(done))

    def add_old_logprob(self, old_logprob):
        self.old_logprobs.append(old_logprob)
    
    def add_logprob(self, logprob):
        self.logprobs.append(logprob)

    def add_discnt_return(self, discnt_returns):
        self.discnt_returns = discnt_returns



    def add(self, obs, reward, state_value, done, old_logprob, valid_actions):
        # self.history.append(history.copy())
        self.valid_actions.append(valid_actions.copy())
    
    def get_batches(self, batch_size):
        indices = torch.randperm(len(self.observations))
        shuffled_obs = [self.observations[i] for i in indices]
        # shuffled_next_value = [self.next_value[i] for i in indices]
        shuffled_next_value = self.next_value
        shuffled_rewards = [self.step_rewards[i] for i in indices]
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
                step_rewards=shuffled_rewards[i:i+batch_size],
                dones=shuffled_dones[i:i+batch_size],
                old_logprobs=shuffled_old_logprobs[i:i+batch_size],
                advantages=torch.tensor(shuffled_advantages[i:i+batch_size], device="cuda", dtype=torch.float32),
                returns=torch.tensor(shuffled_returns[i:i+batch_size], device="cuda", dtype=torch.float32),
                history=shuffled_history[i:i+batch_size],
                valid_actions=shuffled_valid_actions[i:i+batch_size]
            )

class Batch:
    # def __init__(self, observations, step_rewards, dones, old_logprobs, advantages, returns, history, valid_actions):
    def __init__(self, observations, next_value, step_rewards, dones, old_logprobs, advantages, returns, history, valid_actions):
        self.observations = observations
        self.next_value = next_value
        self.step_rewards = step_rewards
        self.dones = dones
        self.old_logprobs = old_logprobs
        self.advantages = advantages
        self.returns = returns
        self.history = history
        self.valid_actions = valid_actions
