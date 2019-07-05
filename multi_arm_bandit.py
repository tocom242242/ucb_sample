import random
import numpy as np
import matplotlib.pyplot as plt

class Arm():
    def __init__(self, idx, mu, sd):
        self.idx = idx    # ランダムでこのアームを引いた時の報酬を設定
        self.mu = mu
        self.sd = sd

    def pull(self):
        return np.random.normal(self.mu, self.sd)

class MultiArmBandit():

    def __init__(self, arm_confs):
        self.arms = self._init_arms(arm_confs)

    def _init_arms(self, arm_confs):
        arms = []
        for arm_conf in arm_confs:
            arm = Arm(arm_conf["id"], arm_conf["mu"], arm_conf["sd"])
            arms.append(arm)

        return arms

    def step(self, arm_id):
        """
            pull lever
        """
        return self.arms[arm_id].pull()
