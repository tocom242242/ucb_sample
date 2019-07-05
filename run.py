import numpy as np
import matplotlib.pyplot as plt
from simple_rl import SimpleRLAgent
from policy import UCB
from multi_arm_bandit import MultiArmBandit

if __name__ == '__main__':
    arm_confs = [{"id":0, "mu":0.1, "sd":0.1}, # 平均 0.1、分散0.1の正規分布に従う乱数によって報酬を設定
                {"id":1, "mu":0.5, "sd":0.1},
                {"id":2, "mu":2, "sd":0.1},
                {"id":3, "mu":0.2, "sd":0.1},
                {"id":4, "mu":0.4, "sd":0.1}]

    game = MultiArmBandit(arm_confs=arm_confs) # 5本のアームを設定
    policy = UCB(c=0.2, actions=np.arange(len(arm_confs)))    # UCBアルゴリズム
    agent = SimpleRLAgent(policy=policy, actions=np.arange(len(arm_confs)))  # agentの設定
    nb_step = 100   # ステップ数
    reward_history = []
    for step in range(nb_step):
        action = agent.act()    # レバーの選択
        reward = game.step(action) # レバーを引く
        agent.observe(reward) #　エージェントは報酬を受け取り学習
        reward_history.append(reward)

    plt.plot(np.arange(nb_step), reward_history)
    plt.ylabel("reward")
    plt.xlabel("steps")
    plt.savefig("result_ucb.png")
    plt.show()
