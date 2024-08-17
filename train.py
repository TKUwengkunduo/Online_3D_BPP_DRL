from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
from bin_packing_env import BinPackingEnv
import matplotlib.pyplot as plt
import os
import numpy as np

# 創建保存模型的目錄
save_path = "./models/"
os.makedirs(save_path, exist_ok=True)

# 創建 TensorBoard 日志目錄
log_dir = "./tensorboard_logs/"
os.makedirs(log_dir, exist_ok=True)

# 超參數設置
hyperparams = {
    "policy": "MlpPolicy",         # 策略網絡類型
    "learning_rate": 3e-4,         # 學習率
    "n_steps": 2048,               # 每次更新的步數
    "batch_size": 64,              # 批次大小
    "n_epochs": 3,                # 每次更新的epoch數量
    "gamma": 0.99,                 # 折扣因子
    "gae_lambda": 0.95,            # GAE lambda
    "clip_range": 0.2,             # 剪裁範圍
    "ent_coef": 0.0,               # 熵損失係數
    "vf_coef": 0.5,                # 值函數損失係數
    "max_grad_norm": 0.5,          # 梯度裁剪
    "verbose": 1                   # 詳細輸出級別
}

# 自定義回調，用於定期保存模型
class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_save_path = os.path.join(self.save_path, f"model_{self.num_timesteps}_steps")
            self.model.save(model_save_path)
            if self.verbose > 0:
                print(f"Model saved to {model_save_path} at step {self.num_timesteps}")
        return True

# 創建環境
env = BinPackingEnv()

# 檢查環境是否符合 Gym 的要求
check_env(env, warn=True)

# 創建 PPO 模型，使用超參數並配置 TensorBoard 日志
model = PPO(
    policy=hyperparams["policy"],
    env=env,
    learning_rate=hyperparams["learning_rate"],
    n_steps=hyperparams["n_steps"],
    batch_size=hyperparams["batch_size"],
    n_epochs=hyperparams["n_epochs"],
    gamma=hyperparams["gamma"],
    gae_lambda=hyperparams["gae_lambda"],
    clip_range=hyperparams["clip_range"],
    ent_coef=hyperparams["ent_coef"],
    vf_coef=hyperparams["vf_coef"],
    max_grad_norm=hyperparams["max_grad_norm"],
    verbose=hyperparams["verbose"],
    tensorboard_log=log_dir
)

# 創建回調，用於保存模型
save_model_callback = SaveModelCallback(save_freq=10000, save_path=save_path)

# 創建 EvalCallback 回調，用於評估和保存最好的模型
eval_callback = EvalCallback(
    env,
    best_model_save_path=save_path,
    log_path=log_dir,
    eval_freq=5000,  # 每 5000 個 timesteps 評估一次
    deterministic=True,
    render=False
)

# 訓練模型並使用回調保存模型
total_timesteps = 1000000  # 總訓練步數，可以靈活調整
model.learn(total_timesteps=total_timesteps, callback=[save_model_callback, eval_callback])

# 保存最後的模型
model.save(os.path.join(save_path, "ppo_bin_packing_final"))

# 測試模型並記錄到 TensorBoard
obs, _ = env.reset()  # 只提取觀察值
done = False

# 設置自定義的TensorBoard記錄器
logger = configure(log_dir, ["stdout", "tensorboard"])

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

        # 記錄填充率到TensorBoard
        logger.record("填充率", fill_rate)
        logger.record("回合總獎勵", np.sum(episode_rewards))
        logger.dump(step=total_timesteps)

        # 顯示最終的可視化結果
        env.render()

        # 暫留視窗，直到用戶關閉
        plt.show()

        obs, _ = env.reset()  # 只提取觀察值
        done = False
        episode_rewards = []  # 重置回合獎勵列表
