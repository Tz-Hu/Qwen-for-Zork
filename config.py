import torch
from peft import LoraConfig, TaskType

# ============================ 参数配置 ============================
MODEL_PATH = "/root/autodl-fs/R3LM/Qwen_chat7b_int4"      
ROM_PATH = "/root/autodl-fs/R3LM/z-machine-games-master/jericho-game-suite/zork1.z5" 
SAVE_DIR = "/root/autodl-fs/R3LM/CheckpointSaves"
LOG_DIR = "/root/autodl-fs/R3LM/RL_Logger"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO 参数
PPO_CLIP_EPS = 0.2      # PPO 的 clip epsilon
PPO_EPOCHS = 3          # 训练时 PPO 的 epoch 数
PPO_LR = 1e-6           # 学习率
MAX_GRAD_NORM = 0.5     # 梯度裁剪的最大范数， 爆炸的话增加这个
ENTROPY_COEFF = 0.03    # 熵正则系数 通常0.01-0.05
ENTROPY_DECAY = 0.995      # 熵衰减系数
VALUE_LOSS_COEFF = 0.5  # 价值损失系数：控制 critic 网络学习的重要性，大可能压制策略更新，小可能学习不到准确状态

# 训练参数
MAX_EPISODES = 50       # 总回合数
EPISODE_STEPS = 45     # 单个回合最大步数
BATCH_SIZE = 15          # 批次大小 
SAVE_INTERVAL = 50  # 保存模型的间隔步数

# Agent 参数
GAE_LAMBDA = 0.95       # Generalized Advantage Estimation 参数
GAMMA_DISCNT = 0.98         # 折扣因子
MAX_HISTORY_LENGTH = 10  # agent 的历史记录动作次数上限

# Env 参数
GAME_RATIO = 1.5         # 计算 reward 时 游戏得分 的比例 
MAX_ACTION_HISTORY = 10 # 记录过去动作
# MAX_NO_SCORE_TIME = 5   # 连续 __ 步没有得分(就惩罚)

# ============================ LoRA 配置 ============================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["attn.c_attn", "attn.c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
