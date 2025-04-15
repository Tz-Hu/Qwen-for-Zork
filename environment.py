from collections import deque
from jericho import FrotzEnv
from config import ROM_PATH, GAME_RATIO, GAMMA_DISCNT, MAX_ACTION_HISTORY
from math import log
    
# ============================ Zork 环境类 ============================
class ZorkEnvWrapper:
    def __init__(self):
        self.env = FrotzEnv(ROM_PATH)
        self.reset()

    def reset(self):
        """"        
        self.close()
        rom, _, _ = self._cache[self.story_file.decode()]
        obs_ini = self.frotz_lib.setup(self.story_file, self._seed, rom, len(rom)).decode('cp1252')
        score = self.frotz_lib.get_score()
        return obs_ini, {'moves':self.get_moves(), 'score':score}
        """
        self.visited_rooms = set()
        self.obj_touched = set()
        self.valid_actions = self.env.get_valid_actions()
        self.action_history = deque(maxlen=MAX_ACTION_HISTORY)
        self.feedback = ""
        self.obs, self.info = self.env.reset()

    def get_valid_actions(self, use_object_tree=True, use_ctypes=True, use_parallel=True):
        """
        Attempts to generate a set of unique valid actions from the current game state.
        尝试从当前游戏状态生成一组唯一的有效动作。

        :param use_object_tree: Query the :doc:`object_tree` for names of surrounding objects.
        :param use_object_tree: 查询 :doc:`object_tree` 以获取周围对象的名称。
        :type use_object_tree: boolean
        :type use_object_tree: 布尔值

        :param use_ctypes: Uses the optimized ctypes implementation of valid action filtering.
        :param use_ctypes: 使用优化的 ctypes 实现来过滤有效动作。
        :type use_ctypes: boolean
        :type use_ctypes: 布尔值

        :param use_parallel: Uses the parallized implementation of valid action filtering.
        :param use_parallel: 使用并行化实现来过滤有效动作。
        :type use_parallel: boolean
        :type use_parallel: 布尔值

        :returns: A list of valid actions.
        :returns: 一个有效动作的列表。
        """
        return self.env.get_valid_actions(use_object_tree, use_ctypes, use_parallel)
    def track_visited_room(self):
        current_room_hash = self.get_wold_state_hash()
        if current_room_hash not in self.visited_rooms:
            self.visited_rooms.add(current_room_hash)
    def get_inverntory(self):
        ''' Returns a list of :class:`jericho.ZObject` in the player's posession. '''
        return self.env.get_inventory()
    def step(self, action):
        # old_score = self.frotz_lib.get_score()
        # next_state = self.frotz_lib.step(action_bytes + b'\n').decode('cp1252')
        # score = self.frotz_lib.get_score()
        # reward = score - old_score
        # return next_state, reward, (self.game_over() or self.victory()),\
        #     {'moves':self.get_moves(), 'score':score}
        self.action_history.append(action)
        self.track_visited_room()
        self.feedback = ""
        return self.env.step(action)




# --------------------------# --------------------------# --------------------------
# # # # # # # 以下为jericho中的复用

# 玩家物品？ 不知道这个返回值是什么形式
    def get_player_object(self):
        ''' Returns the :class:`jericho.ZObject` corresponding to the player. '''
        return self.env.get_object(self.env.player_obj_num)

# 输入编号， 返回物品
    def get_object(self, obj_num):
        '''
        Returns a :class:`jericho.ZObject` with the corresponding number or `None` if the object\
        doesn't exist in the :doc:`object_tree`.

        :param obj_num: Object number between 0 and len(get_world_objects()).
        :type obj_num: int
        '''
        return self.env.get_object(obj_num)

# 返回一个独有编码，可以用于判断是否来过这个环境
    def get_wold_state_hash(self):
        """ Returns a MD5 hash of the clean world-object-tree. Such a hash may be
        useful for identifying when the agent has reached new states or returned
        to existing ones.

        :Example:

        >>> env = FrotzEnv('zork1.z5')
        >>> env.reset()
        # Start at West of the House with the following hash
        >>> env.get_world_state_hash()
        '79c750fff4368efef349b02ff50ffc23'
        >>> env.step('n')
        # Moving to North of House changes the hash
        >>> get_world_state_hash(env)
        '8a3a8538c0019a69128f755e4b719fbd'
        >>> env.step('w')
        # Moving back to West of House we recover the original hash
        >>> env.get_world_state_hash()
        '79c750fff4368efef349b02ff50ffc23'

        """
        return self.env.get_world_state_hash()

    def get_moves(self):
        ''' Returns the integer number of moves taken by the player in the current episode. '''
        return self.env.get_moves()


# 返回 True 如果上一个动作导致了世界的变化。
    def _world_changed(self):
        ''' Returns True if the last action caused a change in the world.
        :Example:

        >>> from jericho import *
        >>> env = FrotzEnv('zork1.z5')
        >>> env.step('north')[0]
        'North of House You are facing the north side of a white house.'
        >>> env.world_changed()
        True
        >>> env.step('south')[0]
        'The windows are all boarded.'
        >>> env.world_changed()
        False
        '''
        return self.env._world_changed()
    
    # 如果 Episode 中胜利则 True
    def victory(self):
        ''' Returns `True` if the game is over and the player has won. '''
        return self.env.victory() 
    
    # 如果 Episode 中失败则 True
    def game_over(self):
        ''' Returns `True` if the game is over and the player has lost. '''
        return self.env.game_over()



# --------------------------# 
# 不知道啥用

# 查看游戏种子, 没什么用
    def seed(self, seed=None):
        '''
        Changes seed used for the emulator's random number generator.

        :param seed: Seed the random number generator used by the emulator.
                     Default: use walkthrough's seed if it exists,
                              otherwise use value of -1 which changes with time.
        :returns: The value of the seed.

        .. note:: :meth:`jericho.FrotzEnv.reset()` must be called before the seed takes effect.
        '''
        return self.env.seed(seed)
    
# 这个函数在评什么分？ 似乎有用
    def _score_object_names(self, interactive_objs):
        """ Attempts to choose a sensible name for an object, typically a noun. """
        def score_fn(obj):
            score = -.01 * len(obj[0])
            if obj[1] == 'NOUN':
                score += 1
            if obj[1] == 'PROPN':
                score += .5
            if obj[1] == 'ADJ':
                score += 0
            if obj[2] == 'OBJTREE':
                score += .1
            return score
        best_names = []
        for desc, objs in interactive_objs.items():
            sorted_objs = sorted(objs, key=score_fn, reverse=True)
            best_names.append(sorted_objs[0][0])
        return best_names

# 辨认可交互的 物品/地点， 不知道咋用
    def _identify_interactive_objects(self, observation='', use_object_tree=False):
        """
        Identifies objects in the current location and inventory that are likely
        to be interactive.
        确定当前地点和库存中可能是可交互的对象。

        :param observation: (optional) narrative response to the last action, used to extract candidate objects.
        :param observation: （可选）对上一个动作的叙述响应，用于提取候选对象。
        :type observation: string
        :type observation: 字符串
        :param use_object_tree: Query the :doc:`object_tree` for names of surrounding objects.
        :param use_object_tree: 查询 :doc:`object_tree` 以获取周围对象的名称。
        :type use_object_tree: boolean
        :type use_object_tree: 布尔值
        :returns: A list-of-lists containing the name(s) for each interactive object.
        :returns: 一个包含每个可交互对象名称的列表的列表。

        :Example:
        :示例:

        >>> from jericho import *
        >>> env = FrotzEnv('zork1.z5')
        >>> obs, info = env.reset()
        'You are standing in an open field west of a white house with a boarded front door. There is a small mailbox here.'
        >>> env.identify_interactive_objects(obs)
        [['mailbox', 'small'], ['boarded', 'front', 'door'], ['white', 'house']]

        .. note:: Many objects may be referred to in a variety of ways, such as\
        Zork1's brass latern which may be referred to either as *brass* or *lantern*.\
        This method groups all such aliases together into a list for each object.
        
        .. 注意:: 许多对象可能有多种称呼，例如 Zork1 的黄铜灯笼可以称为 *brass* 或 *lantern*。\
        此方法将所有这些别名分组到每个对象的一个列表中。
        """
        return self.env._identify_interactive_objects(observation, use_object_tree)

# 依旧不知道咋用
    def _filter_candidate_actions(self, candidate_actions, use_ctypes=False, use_parallel=False):
        """
        给定一个候选动作列表，返回一个字典，将世界差异映射到导致该差异的候选动作列表。
        仅返回导致有效世界差异的动作。

        :param candidate_actions: 要测试 validity 的 candidate action
        :type candidate_actions: list
        :param use_ctypes: 使用优化的 ctypes 实现来过滤有效动作。
        :type use_ctypes: boolean
        :param use_parallel: 使用并行化实现来过滤有效动作。
        :type use_parallel: boolean
        :returns: 世界差异到动作列表的字典。
        """
        return self.env._filter_candidate_actions(candidate_actions, use_ctypes, use_parallel)
        
    

# --------------------------# --------------------------# --------------------------
#                                    reward & score                                #
# --------------------------# --------------------------# --------------------------
#                       Reward                          #
# --------------------------# --------------------------# 
#           小分             #
# --------------------------# 

# 1.重复惩罚 暂时写完了
    def repetitive_action_penalty(self, action):
        penalty = 0.0
        # 连续重复动作
        if len(self.action_history) >= 5 and all(a == action for a in list(self.action_history)[-5:]):
            penalty -= 0.3

        # 循环往返检测，如 ABAB 或 ABCABC
        for cycle_len in [2, 3]:
            if len(self.action_history) >= cycle_len * 2:
                seq = list(self.action_history)[-cycle_len*2:]
                first, second = seq[:cycle_len], seq[cycle_len:]
                if first == second:
                    penalty -= 3 
                    break
        return penalty
    
# 2.新地点奖励, 结合 visited_room 变量 和 get_wold_state_hash 函数 来写
    def new_place_reward(self):
        current_room_hash = self.get_wold_state_hash()
        if current_room_hash not in self.visited_rooms:
            # self.track_visited_room(current_room_hash)
            # 奖励值与已访问房间数量成对数增长
            return 0.2
        return 0
    

# 3.Win/Died 奖惩：
# 上面有 victory 和 game_over 函数 判断输赢
# 总体逻辑：基于 moves 奖励大小： 
# win + move少 > win + move多  >> dead + move多 > dead + move少
# 若 不赢不输 （步数到了） 返回 0
    def win_loss_reward(self):
        if self.victory():
            # 胜利奖励，步数越少奖励越高
            return 10000 / self.get_moves() 
        elif self.game_over():
            # 失败惩罚，步数越多惩罚越轻
            return -50 + 2 * self.get_moves()
        else:
            # 游戏未结束
            return 0

# 4.玩家拿到物品奖励
    def obj_reward(self):
        inventory = self.get_inverntory()
        reward = 0
        # for obj in inventory:
        #     obj_hash = hash(obj)  # 使用物品的哈希值作为唯一标识
        #     if obj_hash not in self.obj_touched:  # 复用 visited_rooms 记录已获得物品
        #         self.obj_touched.add(obj_hash)
        #         reward += 0.8
        return reward


    # 5.若 agent 没有改变世界，小幅度扣分
    def change_world_reward(self):
        world_changed = int(self._world_changed())
        return 0.1 * (world_changed  - 1)
        
    
# 6.可能没必要: 增加一个存活时间奖励
# 7.可能没必要(得分情况太稀疏了): 连续 no_score_time 步没有得分惩罚,
# # 倒是可以设置一个 连续 no_score_time 步没有正向 reward 惩罚

# ------------------------------------------# 
#             单步总Reward                   #
# ------------------------------------------# 
    def my_step_reward(self, action):
        my_step_reward = (
                        self.repetitive_action_penalty(action) 
                        + self.new_place_reward()
                        + self.win_loss_reward()
                        + self.obj_reward()
                        # + self.change_world_reward()
                        )
        return my_step_reward
    
    # 单步奖励 = 游戏得分 + 自定义单步奖励
    def step_reward(self, action):
        _, step_score, _, _= self.step(action)
        return step_score * GAME_RATIO + self.my_step_reward(action)
    
# ------------------------------------------# 
#                   总计(returns)             #
# ------------------------------------------# 
# 输入步数（moves）得到在 moves 步的累计奖励
    def my_return(self, moves):
        total_reward = 0
        # for _ in range(moves):
        #     total_reward += self.my_step_reward(action)
        return total_reward

# 输入步数（moves）得到在 moves 步的累计折扣奖励
    def my_discnt_return(self, moves):
        total_reward = 0
        # discount = 1
        # for _ in range(moves):
        #     total_reward += discount * self.my_step_reward(action)
        #     discount *= GAMMA_DISCNT
        return total_reward



# --------------------------# --------------------------# 
#                          Score                        #
# --------------------------# --------------------------# 
#                  单步 (step)               #
# ------------------------------------------# 

    
# ------------------------------------------# 
#                     总计                   #
# ------------------------------------------# 

    def get_score(self):
        ''' Returns the current integer game score  '''
        return self.env.get_score()
