import numpy as np

class SimpleRLAgent():
    def __init__(self, policy, actions):
        self.policy = policy
        self.actions = actions  # 選択肢
        self.last_action_id = None

    def act(self, q_values=None):
        action_id = self.policy.select_action()    # 行動選択
        self.last_action_id = action_id
        action = self.actions[action_id]
        return action

    def observe(self, reward):
        self.policy.update_ucbs(self.last_action_id, reward)
