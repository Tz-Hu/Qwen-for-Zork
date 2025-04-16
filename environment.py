from collections import deque
from jericho import FrotzEnv
from config import ROM_PATH, GAME_RATIO, MAX_ACTION_HISTORY
from math import log
from jericho import util
    
class ZorkEnvWrapper:
    def __init__(self):
        self.env = FrotzEnv(ROM_PATH)
        self.obs, self.info = self.reset()

    def reset(self):
        self.visited_rooms = set()
        self.obj_touched = set()
        self.step_score = 0
        self.valid_actions = self.env.get_valid_actions()
        self.action_history = deque(maxlen=MAX_ACTION_HISTORY)
        self.feedback = ""
        self.current_inventory = []
        self.obs, self.info = self.env.reset()
        return self.obs, self.info
    def step(self, action):
        self.action_history.append(action)
        self.obs, self.step_score, self.done, self.info = self.env.step(action)
        return self.obs, self.step_score, self.done, self.info
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
            return True
        return False
    def track_touched_obj(self):
        inventory = self.get_inverntory()
        for obj in inventory:
            obj_hash = hash(obj)  # 使用物品的哈希值作为唯一标识
            if obj_hash not in self.obj_touched: 
                self.obj_touched.add(obj_hash)
                return True
        return False
    def get_inverntory(self):
        ''' Returns a list of :class:`jericho.ZObject` in the player's posession. '''
        return self.env.get_inventory()
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
    def _world_changed(self):
    # 返回 True 如果上一个动作导致了世界的变化。
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
    def victory(self):
        ''' Returns `True` if the game is over and the player has won. '''
        return self.env.victory()     
    def game_over(self):
        ''' Returns `True` if the game is over and the player has lost. '''
        return self.env.game_over()    

    def repetitive_action_penalty(self, action):
        penalty = 0.0
        # 连续重复动作
        if len(self.action_history) >= 5 and all(a == action for a in list(self.action_history)[-5:]):
            penalty -= 0.3
            self.feedback = f" Don't repeate {action}\n"

        # 循环往返检测，如 ABAB 或 ABCABC
        for cycle_len in [2, 3]:
            if len(self.action_history) >= cycle_len * 2:
                seq = list(self.action_history)[-cycle_len*2:]
                first, second = seq[:cycle_len], seq[cycle_len:]
                if first == second:
                    penalty -= 3 
                    self.feedback = f" Don't loop {first} {second}\n"
                    break
        return penalty
    
    def new_place_reward(self):
        reward = 0.05
        return reward * self.track_visited_room()
    
    def win_loss_reward(self):
        moves = self.get_moves()
        if self.victory():
            return 10000 / moves
        elif self.game_over():
            return -10000 / moves
        else:
            return 0
        
    def unrecognized_panalty(self, action):
        recognized : bool = util.recognized(action)
        if not recognized:
            self.feedback = f" Answer only with NUMBERs(action index)!\n"
            return -1
        return 0

    def obj_reward(self):
        reward = 0.7
        return self.track_touched_obj() * reward

    # 5.若 agent 没有改变世界，小幅度扣分
    def change_world_reward(self):
        world_changed = int(self._world_changed())
        return 0.1 * (world_changed  - 1)

    def my_step_reward(self, action):
        my_step_reward = (
                        self.repetitive_action_penalty(action) 
                        + self.unrecognized_panalty(action)
                        + self.new_place_reward()
                        # + self.win_loss_reward()
                        + self.obj_reward()
                        + self.change_world_reward()
                        )
        return my_step_reward
    
    # 单步奖励 = 游戏得分 + 自定义单步奖励
    def step_reward(self, action):
        return self.step_score * GAME_RATIO + self.my_step_reward(action)

    def get_score(self):
        ''' Returns the current integer game score  '''
        return self.env.get_score()
