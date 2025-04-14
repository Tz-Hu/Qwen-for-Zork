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
    GAE_TAU, PPO_CLIP_EPS, PPO_EPOCHS, PPO_LR, MAX_GRAD_NORM,MAX_HISTORY_LENGTH)
from Qwen_chat7b_int4.qwen_generation_utils import make_context, get_stop_words_ids, decode_tokens

    ###############################
    #            Agent            #
    ###############################
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
        # 初始化 PPOAgent，传入 Actor、Critic 模型、 和有效动作集
        self.device = device  
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            use_flash_attn=False,
            use_cache=False,  # 说是显式设置为 False 以兼容梯度检查点
            trust_remote_code=True
        )
        self.history = []  # 初始化历史记录
        self.actor = PPOActor(self)  # Actor 模型
        self.critic = PPOCritic(self)  # Critic 模型
        self.entropy_coeff = ENTROPY_COEFF

    # 重置 history
    def reset_history(self):
        self.history = []

    # 更新历史记录，将新的 query 和 response 加入 history
    def update_history(self, query, response):
        self.history.append((query, response))
        
        # 如果 history 超过最大长度，裁剪掉最早的记录
        if len(self.history) > MAX_HISTORY_LENGTH:
            self.history = self.history[-MAX_HISTORY_LENGTH:]

    # 用 PPOActor 来选择动作
    def choose_action(self, obs, valid_actions):
        prompt = self.actor.build_prompt(obs, valid_actions)
        
        response, _, _, _ = self.actor.chat_with_ids(prompt)  
        self.update_history(prompt, response)
        
        # 提取 response 中的最后一个数字并将其转换为索引动作
        match = re.search(r'\d+', response.strip())
        if match:
            action_idx = int(match.group()) - 1  # 转换为 0 开始的索引
            if 0 <= action_idx < len(valid_actions):
                return valid_actions[action_idx]
        # 如果未能提取有效动作，返回默认动作
        return response

    # 计算损失函数
    def compute_loss(self, batch):
        # 获取历史 logprob
        state_values = self.critic.evaluate_values(batch.observations) 
        old_logprobs = torch.stack(batch.old_logprobs).to(device)
        rewards = torch.tensor(batch.rewards, device=device)
        advantages = self.compute_advantage(rewards, state_values, batch.next_value)
        # 计算当前 logprobs
        logprobs = self.actor.compute_logprob(batch.observations, batch.valid_actions)
        # 计算熵 , 变成负数
        entropy = -torch.mean(torch.sum(torch.exp(logprobs) * logprobs, dim=-1))
        # entropy_coeff = ENTROPY_COEFF

        # 1. 策略损失：policy_loss
        # # 计算重要性采样比率
        ratio = torch.exp(logprobs - old_logprobs)
        policy_loss_clip = torch.min(ratio * advantages, 
                                     torch.clamp(ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * advantages)
    
        # 负号因为希望最大化
        policy_loss = - torch.mean(policy_loss_clip)  

        # 2. 价值损失：value_loss
        value_loss = VALUE_LOSS_COEFF * F.mse_loss(state_values, batch.returns)  # 均方误差损失

        # 3. 熵损失：loss_entropy
        entropy_loss = self.entropy_coeff * entropy
        
        # 总损失：loss = loss_pi + loss_value_error + loss_entropy
        loss = policy_loss + value_loss + entropy_loss
        self.entropy_coeff *= ENTROPY_DECAY

        return loss

    def compute_advantage(self, rewards, values, next_value, gamma=GAMMA_DISCNT, tau=GAE_TAU): 
        values = torch.cat([values, next_value.view(1)], dim=0)
        # TD Residual
        # rewards = torch.tensor(rewards, device=values.device, dtype=values.dtype)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, device=values.device, dtype=values.dtype)
        else:
            rewards = rewards.to(device=values.device, dtype=values.dtype)
        deltas = rewards + gamma * values[1:] - values[:-1]

        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            last_gae_lam = deltas[t] + gamma * tau * last_gae_lam
            advantages[t] = last_gae_lam
            
        return advantages.detach()

    ###############################
    #            Actor            #
    ###############################
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
    def chat_with_ids(self, query, system=f"""You are playing the role of a professional player in the text-based adventure game Zork. You can earn rewards by exporing new place, solve puzzles, collecting treasures and defeating bad guys. You need to choose the best action from the list of valid actions to maximize your score.""", generation_config=None, **kwargs):
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

    # 计算生成动作的 log 概率
    def compute_logprob(self, obs, valid_actions, history=None):
        prompt = self.build_prompt(obs, valid_actions)
        # 如果没有提供历史记录，使用代理的历史记录
        if history is None:
            history = self.history
        
        # 生成动作
        response, _, input_ids, gen_tokens = self.chat_with_ids(prompt)
        
        # 计算生成动作的 log 概率
        with torch.no_grad():  # 禁用梯度计算以节省内存
            outputs = self.actor_model(input_ids)
            logits = outputs.logits[:, -gen_tokens.size(1):]
            log_probs = torch.log_softmax(logits, dim=-1)
            gen_log_probs = log_probs.gather(2, gen_tokens.unsqueeze(-1)).squeeze(-1)
            total_log_prob = gen_log_probs.sum(dim=-1)
        
        return total_log_prob[0]
        # history = self.agent.history  # 获取历史记录

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
        
        # 从tokenized对象中获取input_ids和attention_mask
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        # 禁用梯度计算以节省内存
        with torch.no_grad():
            # 使用 Critic 模型对输入的 token 序列进行前向传播，获取输出
            outputs = self.critic_model(input_ids, output_hidden_states=True)
            
            # 提取最后一层的 hidden state（作为状态价值的估计）
            last_hidden_state = outputs.hidden_states[-1].to(dtype=torch.float32)  # 转换为 float32 类型
            pooled = last_hidden_state.mean(dim=1)

        state_value = self.value_head(pooled).squeeze(-1) 
        state_value = state_value.float()  # 转换为 float32 类型
        # state_value = state_value.to(dtype = torch.float32, device = device)
        return state_value

    # 计算 状态价值
    def evaluate_values(self, obs):
        return self.forward(obs)

    # 计算 价值损失
    def compute_value_loss(self, values, returns):
        return F.mse_loss(values, returns)
    
