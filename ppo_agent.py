import re
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import get_peft_model, prepare_model_for_kbit_training
from config import (
    MODEL_PATH, device, lora_config, MAX_EPISODES,
    ENTROPY_COEFF, ENTROPY_DECAY, VALUE_LOSS_COEFF, GAMMA_DISCNT,
    GAE_LAMBDA, PPO_CLIP_EPS, PPO_EPOCHS, PPO_LR, MAX_GRAD_NORM,MAX_HISTORY_LENGTH)
from Qwen_chat7b_int4.qwen_generation_utils import make_context, get_stop_words_ids, decode_tokens


class PPOAgent():
    def __init__(self, device = device):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            pad_token='<|extra_0|>',
            eos_token='<|endoftext|>',
            output_hidden_states=True,
            padding_side='left'
        )
        self.device = device  
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            use_flash_attn=True,
            use_cache=False,  # 说是显式设置为 False 以兼容梯度检查点
            trust_remote_code=True
        )
        self.actor = PPOActor(self)
        self.critic = PPOCritic(self)
        self.history = []
        self.entropy_coeff = ENTROPY_COEFF

    def reset(self):
        self.history = []
        self.entropy_coeff = ENTROPY_COEFF

    # 更新历史记录，将新的 query 和 response 加入 history
    def update_history(self, query, response):
        self.history.append((query, response))
        
        # 如果 history 超过最大长度，裁剪掉最早的记录
        if len(self.history) > MAX_HISTORY_LENGTH:
            self.history = self.history[-MAX_HISTORY_LENGTH:]

    def compute_advantage_returns(self, 
                          step_rewards: torch.Tensor,
                          state_values: torch.Tensor, 
                          next_value: float = 0.0, 
                          gamma=GAMMA_DISCNT, 
                          gae_lambda=GAE_LAMBDA,
                          normalize: bool = True) -> torch.Tensor:
        device = step_rewards.device
        state_values = state_values.to(device)

        T = len(step_rewards)
        advantages = torch.zeros_like(step_rewards, device=device)
        last_gae = 0.0

        # 反向计算 GAE 
        with torch.no_grad():
            for t in reversed(range(T)):
                delta = step_rewards[t] + gamma * next_value - state_values[t]
                last_gae = delta + gamma * gae_lambda * last_gae
                advantages[t] = last_gae
                next_value = state_values[t]
        returns = advantages + state_values[:-1]
    
        # 计算回报值
        returns = advantages + state_values[:-1]  # 需要确保values长度=T+1
        
        # 可选：标准化优势
        if normalize and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    # 计算损失函数
    def compute_loss(self, batch)->tuple:
        old_logprob = batch.old_logprob
        logprob = batch.logprob
        state_values = batch.state_values
        step_rewards = batch.step_rewards
        advantages, returns = self.compote_advantage_returns(step_rewards, state_values)
        entropy_coef: float = ENTROPY_COEFF,  # 熵奖励权重
        clip_epsilon: float = PPO_CLIP_EPS,   # PPO截断阈值

        ratio = torch.exp(logprob - old_logprob.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(state_value, returns, reduction='mean')

        entropy = -torch.exp(logprob) * logprob  # 计算熵 [batch_size]
        entropy_loss = -entropy.mean() * entropy_coef  # 负号因为要最大化熵

        total_loss = policy_loss + value_coef * value_loss + entropy_loss

        # # 监控指标：近似KL散度（不需要反向传播）
        # with torch.no_grad():
        #     approx_kl = 0.5 * torch.mean((old_logprob - logprob)**2)  # Fisher信息近似

        return total_loss 
        # return total_loss, policy_loss, value_loss, entropy_loss, approx_kl
# # ===================
#         # 1. 策略损失：policy_loss
#         # # 计算重要性采样比率
#         ratio = torch.exp(logprobs - old_logprobs)
#         policy_loss_clip = torch.min(ratio * advantages, 
#                                      torch.clamp(ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * advantages)
#         # 负号因为希望最大化
#         policy_loss = - torch.mean(policy_loss_clip)  
#         # 2. 价值损失：value_loss
#         value_loss = VALUE_LOSS_COEFF * F.mse_loss(state_values, batch.returns)  # 均方误差损失
#         # 3. 熵损失：loss_entropy
#         entropy_loss = self.entropy_coeff * entropy
#         # 总损失：loss = loss_pi + loss_value_error + loss_entropy
#         loss = policy_loss + value_loss + entropy_loss
#         self.entropy_coeff *= ENTROPY_DECAY
#         return loss

class PPOActor(nn.Module):
    def __init__(self, agent):
        super().__init__()
        self.history = agent.history
        self.tokenizer = agent.tokenizer 
        self.actor_model = agent.model
        # 使用 LoRA 接口及低比特训练配置，仅让 LoRA 参数更新
        self.actor_model = prepare_model_for_kbit_training(self.actor_model, use_gradient_checkpointing=True)
        self.actor_model = get_peft_model(self.actor_model, lora_config)
        # 冻结除最后两层外的所有参数
        for name, param in self.actor_model.named_parameters():
            if ("h.30" not in name) and ("h.31" not in name):
                param.requires_grad = False
        self.actor_model.train()
        
        # 加载生成配置
        self.generation_config = GenerationConfig.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            temperature = 0.7, # 生成时的温度参数, 低了趋向于确定  
            max_length=9,
            # early_stopping=True,  # 提前停止生成
            length_penalty=1.2  # 惩罚值越大生成越短
        )
        # 注入辅助函数（继承qwen初始模型）
        self.make_context = make_context
        self.get_stop_words_ids = get_stop_words_ids
        self.decode_tokens = decode_tokens
    
                                # chat 源码改编。query填入每次的 prompt
    def chat_with_ids(self, query, system=f"""You are playing the role of a professional player in the text-based adventure game Zork. You can earn rewards by solve puzzles, collecting treasures and defeating bad guys. You need to choose the best action from the list of valid actions to maximize your score.""", generation_config=None, **kwargs):
        generation_config = self.generation_config

        history = self.history  # 获取历史记录
        history = copy.deepcopy(history)
        # 上下文长度，用qwen自带
        max_window_size = kwargs.get('max_window_size', generation_config.max_window_size)
        raw_text, context_tokens = self.make_context(
            self.tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
            chat_format=generation_config.chat_format,
        )
        stop_words_ids = self.get_stop_words_ids(generation_config.chat_format, self.tokenizer)
        input_ids = torch.tensor([context_tokens]).to(self.actor_model.device)
        outputs = self.actor_model.generate(
            input_ids = input_ids,
            stop_words_ids=stop_words_ids,
            return_dict_in_generate=True,
            # top_p=0.9,                # 使用 top_p 样本加速生成
            output_scores=True,
            generation_config=generation_config,
            **kwargs,
        )
        gen_tokens = outputs.sequences[:, input_ids.shape[1]:]
        response = self.decode_tokens(
            outputs.sequences[0],
            self.tokenizer,
            raw_text_len=len(raw_text),
            context_length=len(context_tokens),
            chat_format=generation_config.chat_format,
            verbose=False,
            errors='replace'
        )
        history.append((query, response))

        return response, history, input_ids, gen_tokens

    def generate_action_logprob(self, obs, valid_actions):
        prompt = self.build_prompt(obs, valid_actions)
        history = self.history
        # 生成动作
        response, _, input_ids, gen_tokens = self.chat_with_ids(prompt)
        self.update_history(prompt, response)
        
        match = re.search(r'\d+', response.strip())
        if match:
            action_idx = int(match.group()) - 1  # 转换为 0 开始的索引
            if 0 <= action_idx < len(valid_actions):
                return valid_actions[action_idx]
            
        # 计算生成动作的 log 概率
        with torch.no_grad():  # 禁用梯度计算以节省内存
            outputs = self.actor_model(input_ids)
            logits = outputs.logits[:, -gen_tokens.size(1):]
            log_probs = torch.log_softmax(logits, dim=-1)
            gen_log_probs = log_probs.gather(2, gen_tokens.unsqueeze(-1)).squeeze(-1)
            total_log_prob = gen_log_probs.sum(dim=-1)
        
        return response, total_log_prob[0]

    
    def filter_copyright(self, text):
        """ 过滤包含版权信息的行 """
        keywords = ["Copyright", "Microsoft Corporation", "GNU General Public License", "Serial number", "ZORK is a registered trademark"]
        lines = text.split('\n')
        filtered = [line for line in lines if not any(kw in line for kw in keywords)]
        return '\n'.join(filtered).strip()

    def build_prompt(self, obs, valid_actions, feedback = None):
        # obs = self.filter_copyright(obs)  # 过滤掉包含版权信息的行
        numbered_actions = "\n".join([f" {i+1}: {act}" for i, act in enumerate(valid_actions)])
        prompt = f"""Your last action reward is {feedback}
# Current Environment Description:
{obs}
# Valid Actions:  
{numbered_actions}
Only answer A SINGLE NUMBER from 1 to {len(valid_actions)}!
No any additional characters or explanation!"""
        return prompt
    
    ###############################
    #            Critic           #
    ###############################
class PPOCritic(nn.Module):
    def __init__(self,agent):
        super().__init__()
        self.tokenizer = agent.tokenizer
        self.critic_model = agent.model
        # 使用 LoRA 接口及低比特训练配置， 但不进行训练
        self.critic_model = prepare_model_for_kbit_training(self.critic_model, use_gradient_checkpointing=True)
        self.critic_model = get_peft_model(self.critic_model, lora_config)

        # 添加这个 value_head：从 pooled embedding -> 状态价值
        self.value_head = nn.Linear(self.critic_model.config.hidden_size, 1)

    # 与 Actor 部分类似，使用 Critic 模型进行前向传播
    def forward(self, obs):
        # 将输入文本转换为token ids，同时获取attention mask
        tokenized = self.tokenizer(
            obs, 
            return_tensors="pt", 
            padding=True,
            return_attention_mask=True  # 显式要求返回attention mask
        ).to(device)
        
        input_ids = tokenized.input_ids
        with torch.no_grad():
            outputs = self.critic_model(input_ids, output_hidden_states=True)
            
            # last_hidden_state = outputs.hidden_states[-1].to(dtype=torch.float32)  # 转换为 float32 类型
            # pooled = last_hidden_state.mean(dim=1)
            pooled = outputs.hidden_states[-1].mean(dim=1)

        # state_value = self.value_head(pooled).squeeze(-1) 
        # state_value = state_value.float()  # 转换为 float32 类型
        return self.value_head(pooled).squeeze(-1) 

    # 计算 状态价值
    def evaluate_value(self, obs):
        return self.forward(obs)

    # 计算 价值损失
    def compute_value_loss(self, values, returns):
        return F.mse_loss(values, returns)
    
