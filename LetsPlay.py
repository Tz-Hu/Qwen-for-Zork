# autodl-fs/R3LM/LetsPlay.py
from jericho import FrotzEnv
import pickle
import glob
import os

SAVE_DIR = "/root/autodl-fs/R3LM/saves"  # 新增常量（建议改成相对路径）

# 初始化环境
rom_path = "/root/autodl-fs/R3LM/z-machine-games-master/jericho-game-suite/zork1.z5"
env = FrotzEnv(rom_path)
os.makedirs(SAVE_DIR, exist_ok=True)  # 新增目录创建
print("请等待...加载中...")
obs, info = env.reset()

def save_game(state, moves, filename_prefix="save"):
    """保存到指定目录"""
    # if not isinstance(moves, int):
    #     raise ValueError("步数必须是整数")
    filename = os.path.join(SAVE_DIR, f"{filename_prefix}{moves:04d}.sav")
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    return filename

def load_game(filename):
    """从指定目录加载"""
    full_path = os.path.join(SAVE_DIR, filename)
    with open(full_path, 'rb') as f:
        return pickle.load(f)

def list_saves():
    """获取指定目录的存档"""
    save_files = glob.glob(os.path.join(SAVE_DIR, "save*.sav"))
    valid_saves = []
    
    for f in save_files:
        base_name = os.path.basename(f)
        digits = ''.join(filter(str.isdigit, base_name))
        if not digits:
            continue
        
        try:
            moves = int(digits)
            valid_saves.append( (base_name, moves) )  # 改为存储文件名
        except ValueError:
            continue
    
    return [f[0] for f in sorted(valid_saves, key=lambda x: x[1])]

# 游戏主循环
done = False
while not done:
    current_moves = info['moves']
    
    # 显示可选动作
    valid_actions = env.get_valid_actions()
    print("\n---------- 可选动作 ------------")
    for idx, action in enumerate(valid_actions, 1):
        print(f"{idx}. {action}")

    # 用户输入处理
    user_input = input("\n请输入动作命令（save/load/quit）> ").strip().lower()
    
    # 存档
    if user_input == 'save':
        try:
            saved_state = env.get_state()
            filename = save_game(saved_state, current_moves)
            print(f"存档成功：{filename}")
        except Exception as e:
            print(f"存档失败：{str(e)}")
        continue
        
    # 读档
    if user_input == 'load':
        saves = list_saves()
        if not saves:
            print("没有找到有效存档")
            continue
            
        print("\n=== 存档列表 ===")
        for idx, save_path in enumerate(saves, 1):
            moves = ''.join(filter(str.isdigit, os.path.basename(save_path)))
            print(f"{idx}. 步数 {moves}")
        
        try:
            choice = int(input("请输入存档编号："))
            if 1 <= choice <= len(saves):
                loaded_state = load_game(saves[choice-1])
                env.set_state(loaded_state)
                obs, _, done, info = env.step('look')
                print(f"\n {obs}")
            else:
                print("编号超出范围")
        except ValueError:
            print("请输入数字编号")
        except Exception as e:
            print(f"加载失败：{str(e)}")
        continue
        
    # 退出命令
    if user_input == 'quit':
        break

    # 处理游戏动作
    try:
        if user_input.isdigit():
            choice_idx = int(user_input) - 1
            if 0 <= choice_idx < len(valid_actions):
                action = valid_actions[choice_idx]
            else:
                print("无效编号")
                continue
        else:
            action = user_input
        
        obs, reward, done, info = env.step(action)
        print(f"\n=== 执行结果 ===\n{obs}\n得分: {info['score']} 步数: {info['moves']}")
        
    except Exception as e:
        print(f"操作失败：{str(e)}")

print("\n=== 游戏结束 ===")
print(f"最终得分: {info['score']} 步数: {info['moves']}")