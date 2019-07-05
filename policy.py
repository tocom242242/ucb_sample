import numpy as np
import math

class UCB():
    def __init__(self, c, actions):
        self.average_rewards = np.repeat(0.0, len(actions)) # 各腕の平均報酬
        self.ucbs = np.repeat(10.0, len(actions))   # UCB値
        self.counters = np.repeat(0, len(actions))  # 各腕の試行回数
        self.c = c
        self.all_conter = 0 # 全試行回数

    def select_action(self):
        action_id = np.argmax(self.ucbs)
        self.counters[action_id] += 1
        self.all_conter += 1
        return action_id

    def update_ucbs(self, action_id, reward):
        self.update_average_rewards(action_id, reward)
        self.ucbs[action_id] = self.average_rewards[action_id] + self.c * math.log(2*self.all_conter)/np.sqrt(self.counters[action_id])

    def update_average_rewards(self, action_id, reward):
        self.average_rewards[action_id] = self.average_rewards[action_id] + (reward-self.average_rewards[action_id])/(self.counters[action_id]+1)
