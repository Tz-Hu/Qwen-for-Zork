import torch
class PPOBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.old_logprobs = []
        self.valid_actions = []
        self.step_rewards = []
        self.state_values = []
        self.next_value = None
        self.observations = []
        self.feedbacks = []

    def add_obs(self, observations):
        self.observations.append(observations)

    def add_step_reward(self, reward):
        self.step_rewards.append(reward)

    def add_feedback(self, feedback):
        self.feedbacks.append(feedback)

    def add_state_value(self, state_value):
        self.state_values.append(state_value)

    def add_old_logprob(self, old_logprob):
        self.old_logprobs.append(old_logprob)
    
    def add_valid_action(self, valid_actions):
        self.valid_actions.append(valid_actions)
    
    def get_batches(self, batch_size):
        indices = torch.randperm(len(self.observations))
        shuffled_old_logprobs = [self.old_logprobs[i] for i in indices]
        shuffled_valid_actions = [self.valid_actions[i] for i in indices]
        shuffled_step_rewards = [self.step_rewards[i] for i in indices]
        shuffled_state_values = [self.state_values[i] for i in indices]
        shuffled_next_value = self.next_value
        shuffled_obs = [self.observations[i] for i in indices]
        shuffled_feedbacks = [self.feedbacks[i] for i in indices]

        total = len(shuffled_old_logprobs)
        for i in range(0, total, batch_size):
            yield Batch(
                old_logprobs=shuffled_old_logprobs[i:i+batch_size],
                valid_actions=shuffled_valid_actions[i:i+batch_size],
                step_rewards=shuffled_step_rewards[i:i+batch_size],
                state_values=shuffled_state_values[i:i+batch_size],
                next_value=shuffled_next_value,
                observations=shuffled_obs[i:i+batch_size],
                feedbacks=shuffled_feedbacks[i:i+batch_size]

            )

class Batch:
    def __init__(self, old_logprobs, valid_actions, step_rewards, state_values, next_value, observations, feedbacks):
        self.old_logprobs = old_logprobs
        self.valid_actions = valid_actions
        self.step_rewards = step_rewards        
        self.state_values = state_values         
        self.next_value = next_value              
        self.observations = observations          
        self.feedbacks = feedbacks                  
