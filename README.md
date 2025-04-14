# Qwen-for-Zork

基于本地部署的 Qwen-chat-7b-int4 模型实现一个基于 PPO 算法的 online RL 微调，用于在文字冒险游戏 Zork 中进行探索和决策。
项目使用 Proximal Policy Optimization (PPO) 算法，并结合了 LoRA 技术优化模型训练。

## 项目结构
```
R3LM/
├── config.py      # 配置文件，包含模型路径、训练参数等
├── environment.py # 环境封装类和奖励系统
├── ppo_agent.py   # PPO 智能体实现
├── ppo_buffer.py  # PPO 缓冲区实现
├── train.py       # 训练逻辑
├── start.py       # 项目入口文件
└── README.md      # 项目说明文件
```

### 文件说明

- **`config.py`**: 包含所有的超参数配置，例如模型路径、训练参数、设备配置等。
- **`environment.py`**: 封装了 Zork 游戏环境，并实现了自定义奖励系统。
- **`ppo_agent.py`**: 实现了基于 LoRA 的 PPO 智能体，包括负责动作选择的 actor 和策略评估的 critic。
- **`ppo_buffer.py`**: 实现了 PPO 的经验缓冲区，用于存储训练数据并计算优势估计。
- **`train.py`**: 包含训练逻辑，包括与环境交互、策略更新和模型保存。
- **`start.py`**: 项目入口文件，负责启动训练。


## Requirements

Linux, Python 3.9+, Spacy, and basic build tools like gcc,make & curl.

## Install
可以通过以下命令安装依赖：

```bash
python -m venv jericho
source jericho/bin/activate

python -m pip install jericho
python -m spacy download en_core_web_sm

pip install torch transformers peft jericho
```

```bash
wget https://github.com/BYU-PCCL/z-machine-games/archive/master.zip
unzip master.zip
```

## 使用方法

1. 配置参数
在 config.py 文件中设置模型路径、ROM 文件路径以及其他训练参数。

2. 启动训练
运行以下命令启动训练：
```bash
python start.py
```
训练过程中会生成日志文件和模型检查点，分别保存在 LOG_DIR 和 SAVE_DIR 指定的路径中。

3. 查看日志
训练日志会保存在 LOG_DIR 中的文本文件中，每个回合生成一个日志文件。
