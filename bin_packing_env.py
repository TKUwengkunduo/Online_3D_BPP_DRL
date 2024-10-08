import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import configparser
import random

class BinPackingEnv(gym.Env):
    def __init__(self):
        super(BinPackingEnv, self).__init__()
        
        # 讀取配置文件
        config = configparser.ConfigParser()
        config.read('config.txt')
        
        self.container_length = int(config['Container']['length'])
        self.container_width = int(config['Container']['width'])
        self.container_height = int(config['Container']['height'])
        self.precision = int(config['Container']['precision'])

        self.items = [
            tuple(map(float, config['ItemSizes']['item1'].split(','))),
            tuple(map(float, config['ItemSizes']['item2'].split(','))),
            tuple(map(float, config['ItemSizes']['item3'].split(',')))
        ]

        # 讀取物品出現的機率
        self.item_probabilities = [
            float(config['ItemProbabilities']['item1']),
            float(config['ItemProbabilities']['item2']),
            float(config['ItemProbabilities']['item3'])
        ]
        
        self.visualization_enabled = config['Visualization'].getboolean('enabled')

        # 顏色對應表，對應每個item
        self.item_colors = ['b', 'g', 'r']

        # 計算狀態和動作空間的大小
        self.action_space = spaces.Discrete(self.container_length * self.container_width * 2)  # 每個位置有兩個方向
        self.observation_space = spaces.Box(
            low=0, 
            high=max(self.container_height, max(max(self.items))),  # max用於處理item的最大尺寸
            shape=(self.container_length * self.container_width + 3,),  # +3 表示item的長、寬、高
            dtype=np.float32
        )
        
        # 初始化可視化
        if self.visualization_enabled:
            self.init_visualization()

        # 初始化環境狀態
        self.reset()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # 確保 seed 被傳遞給父類的 reset 方法
        
        self.container = np.zeros((self.container_length, self.container_width), dtype=np.float32)
        self.current_item_index = 0
        self.done = False
        
        # 選擇第一個item
        self.current_item = self.items[np.random.choice(len(self.items), p=self.item_probabilities)]

        if self.visualization_enabled:
            self.reset_visualization()
        
        # 返回包含item信息的觀測值，轉換為float32
        observation = np.concatenate([self.container.flatten(), np.array(self.current_item)]).astype(np.float32)
        return observation, {}


    def init_visualization(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # 設置比例尺
        self.ax.set_box_aspect([self.container_length, self.container_width, self.container_height])  # 調整軸的比例

        self.ax.set_xlim([0, self.container_length])
        self.ax.set_ylim([0, self.container_width])
        self.ax.set_zlim([0, self.container_height])
        self.ax.set_xlabel('Length')
        self.ax.set_ylabel('Width')
        self.ax.set_zlabel('Height')

    def update_visualization(self, item_position, item_size, item_index):
        if self.visualization_enabled and hasattr(self, 'ax'):
            color = self.item_colors[item_index]
            self.ax.bar3d(item_position[0], item_position[1], item_position[2], 
                          item_size[0], item_size[1], item_size[2], color=color, alpha=0.6)
            plt.draw()
            plt.pause(0.01)

    def reset_visualization(self):
        if self.visualization_enabled and hasattr(self, 'ax'):
            self.ax.cla()
            self.ax.set_box_aspect([self.container_length, self.container_width, self.container_height])  # 調整軸的比例

            self.ax.set_xlim([0, self.container_length])
            self.ax.set_ylim([0, self.container_width])
            self.ax.set_zlim([0, self.container_height])

    def step(self, action):
        if self.done:
            return self.container.flatten().astype(np.float32), 0, True, False, {}

        # 使用當前item進行操作
        position_index = action // 2
        rotation = action % 2

        # 旋轉物品
        item_size = self.current_item
        if rotation == 1:
            item_size = (item_size[1], item_size[0], item_size[2])

        # 計算放置位置
        x = position_index // self.container_width
        y = position_index % self.container_width
        z = np.max(self.container[x:x+int(item_size[0]), y:y+int(item_size[1])])

        # 檢查是否可以放置
        if (x + item_size[0] <= self.container_length and 
            y + item_size[1] <= self.container_width and 
            z + item_size[2] <= self.container_height):
            # 更新容器狀態
            self.container[x:x+int(item_size[0]), y:y+int(item_size[1])] = z + item_size[2]
            reward = 0.3  # 成功放置獎勵


            # 計算接觸面的數量
            contact_faces = 0

            # 檢查左邊接觸
            if x == 0 or np.any(self.container[x-1, y:y+int(item_size[1])] >= z):
                contact_faces += 1
            # 檢查右邊接觸
            if x + item_size[0] == self.container_length or np.any(self.container[x+int(item_size[0]):x+int(item_size[0])+1, y:y+int(item_size[1])] >= z):
                contact_faces += 1
            # 檢查前邊接觸
            if y == 0 or np.any(self.container[x:x+int(item_size[0]), y-1] >= z):
                contact_faces += 1
            # 檢查後邊接觸
            if y + item_size[1] == self.container_width or np.any(self.container[x:x+int(item_size[0]), y+int(item_size[1]):y+int(item_size[1])+1] >= z):
                contact_faces += 1
            
            # 根據接觸面數量計算獎勵
            if contact_faces == 1:
                reward -= 0.1
            elif contact_faces == 2:
                reward += 0.3
            elif contact_faces == 3:
                reward += 0.5
            elif contact_faces == 4:
                reward += 0.8

            

            # 計算底面積接觸
            # bottom_contact_area = 0
            # total_bottom_area = item_size[0] * item_size[1]
            # for i in range(int(item_size[0])):
            #     for j in range(int(item_size[1])):
            #         if self.container[x+i, y+j] == z:
            #             bottom_contact_area += 1

            # # 計算接觸比例
            # contact_ratio = bottom_contact_area / total_bottom_area

            # # 根據接觸面積比例計算獎勵
            # if z == 0:  # 接觸容器底部
            #     reward += 0.5
            # else:  # 接觸其他物品
            #     if contact_ratio < 0.5:
            #         reward -= 0.5
            #     elif contact_ratio < 0.75:
            #         reward += 0.3
            #     else:
            #         reward += 0.5



            # 更新觀測值
            self.current_item = self.items[np.random.choice(len(self.items), p=self.item_probabilities)]
            observation = np.concatenate([self.container.flatten(), np.array(self.current_item)]).astype(np.float32)

            # 找到與 item_size 匹配的 item_index
            item_index = None
            for i, item in enumerate(self.items):
                if (item[0] == item_size[0] and item[1] == item_size[1] and item[2] == item_size[2]) or \
                (item[1] == item_size[0] and item[0] == item_size[1] and item[2] == item_size[2]):
                    item_index = i
                    break

            if item_index is not None:
                self.update_visualization((x, y, z), item_size, item_index)
            
             # 檢查是否無法繼續放置
            if self.is_draw():
                self.done = True
                terminated = True
                return observation, reward, terminated, False, {}
        
        else:
            reward = -1  # 放置失敗懲罰 -1
            self.done = True
            # 即使放置失敗，也需要生成一個觀測值
            observation = np.concatenate([self.container.flatten(), np.array(self.current_item)]).astype(np.float32)

        terminated = self.done
        truncated = False  # 可以根據需求來決定是否提早終止

        # 回傳包含item信息的觀測值，轉換為float32
        return observation, reward, terminated, truncated, {}



    def render(self, mode='human'):
        pass  # 渲染已經通過 update_visualization 完成

    def calculate_final_reward(self):
        # 計算填充率
        filled_volume = np.sum(self.container)  # 容器中填充的體積
        total_volume = self.container_length * self.container_width * self.container_height
        fill_rate = filled_volume / total_volume * 100

        # 基於填充率的最終獎勵
        final_reward = fill_rate / 10
        return final_reward

    def is_draw(self):
        # 定義平局條件，例如檢查剩餘空間是否足以放置最小的物品
        for i in range(self.container_length):
            for j in range(self.container_width):
                if self.container[i, j] < self.container_height:
                    space_height = self.container_height - self.container[i, j]
                    for item in self.items:
                        item_length, item_width, item_height = item
                        if i + item_length <= self.container_length and \
                        j + item_width <= self.container_width and \
                        space_height >= item_height:
                            return False  # 還能放置物品，不是平局
        return True  # 無法放置任何物品，視為平局