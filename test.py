from stable_baselines3 import PPO
from bin_packing_env import BinPackingEnv
import matplotlib.pyplot as plt
import numpy as np
import os

# 加載模型的目錄
model_path = "./models/best_model.zip"
# model_path = "./models/model_480000_steps.zip"

# 創建環境
env = BinPackingEnv()

# 加載訓練好的模型
model = PPO.load(model_path)

# 測試模型
obs, _ = env.reset()  # 只提取觀察值
done = False

episode_rewards = []

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)  # 解包五個值
    done = terminated or truncated
    episode_rewards.append(rewards)

    if done:  # 如果回合結束
        # 計算填充率
        filled_volume = env.container.sum()  # 容器中填充的體積
        total_volume = env.container_length * env.container_width * env.container_height
        fill_rate = filled_volume / total_volume * 100

        print(f"填充率: {fill_rate:.2f}%")
        print(f"回合總獎勵: {np.sum(episode_rewards)}")

        # 顯示最終的可視化結果
        env.render()

        # 暫留視窗，直到用戶關閉
        plt.show()

        # 結束測試
        break
