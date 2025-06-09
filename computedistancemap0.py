#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# 根据你的项目结构调整导入路径
from domains_cc.footprint import createFootprint
from domains_cc.map_generator import generate_random_map
from domains_cc.worldCC import WorldCollisionChecker

class PathFindingEnvWithMap(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 grid_size=(20,20),
                 obstacle_prob=0.1,
                 footprint_spec=None,
                 max_steps=300):
        super().__init__()

        # 全局地图 & 碰撞检测（grid 值：0 空地，1 障碍）
        self.grid = generate_random_map(grid_size, obstacle_prob).astype(np.float32)
        self.world_cc = WorldCollisionChecker(self.grid)
        self.H, self.W = self.grid.shape

        # 机器人 footprint
        if footprint_spec is None:
            footprint_spec = dict(type="rectangle", width=0.5,
                                  height=1.5, resolution=0.1)
        self.footprint = createFootprint(
            footprint_spec["type"],
            dict(width=footprint_spec["width"],
                 height=footprint_spec["height"],
                 resolution=footprint_spec["resolution"])
        )

        # 动作/观测空间（为示例简化）
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict({
            "vec_obs":     spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
            "next_obs":    spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(6),
        })

        self.max_steps     = max_steps
        self.current_state = np.zeros(3, np.float32)
        self.goal_state    = np.zeros(3, np.float32)
        self.step_count    = 0

        # 只保留距离图
        self.dist_map = np.full((self.H, self.W), np.inf, np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # 随机起点：浮点坐标，只要可放即行
        def sample_pose_continuous():
            return np.array([
                np.random.uniform(0, self.W),
                np.random.uniform(0, self.H),
                np.random.uniform(0, 360)
            ], np.float32)

        # 起点
        while True:
            s = sample_pose_continuous()
            if self.world_cc.isValid(self.footprint, s.reshape(1,3))[0]:
                break

        # 随机终点：先随机格子索引，再对齐到格子中心，确保可放
        while True:
            gi = np.random.randint(0, self.H)
            gj = np.random.randint(0, self.W)
            theta = np.random.uniform(0, 360)
            g = np.array([gj + 0.5, gi + 0.5, theta], np.float32)
            if self.world_cc.isValid(self.footprint, g.reshape(1,3))[0] and \
               np.linalg.norm(g[:2] - s[:2]) > 0.1:
                break

        self.current_state = s.copy()
        self.goal_state    = g.copy()
        self.step_count    = 0

        # 计算纯格子级曼哈顿距离图
        self.dist_map = self.compute_distance_map(self.grid, gi, gj)

        return {}, {}

    @staticmethod
    def compute_distance_map(grid: np.ndarray, goal_i: int, goal_j: int) -> np.ndarray:
        """
        基于 occupancy grid 用 BFS 计算曼哈顿距离图。
        grid: 障碍为 1，空地为 0
        goal_i, goal_j: 目标格子索引
        返回 dist 数组，不可达或障碍处为 np.inf
        """
        H, W = grid.shape
        INF = np.inf
        dist = np.full((H, W), INF, dtype=np.float32)

        free = (grid < 0.5)
        queue = deque()
        if free[goal_i, goal_j]:
            dist[goal_i, goal_j] = 0
            queue.append((goal_i, goal_j))

        neighs = [(-1,0),(1,0),(0,-1),(0,1)]
        while queue:
            i, j = queue.popleft()
            d0 = dist[i, j]
            for di, dj in neighs:
                ni, nj = i+di, j+dj
                if 0 <= ni < H and 0 <= nj < W and free[ni, nj]:
                    if dist[ni, nj] == INF:
                        dist[ni, nj] = d0 + 1
                        queue.append((ni, nj))
        return dist

    def render(self, save_path="distance_map.png"):
        # 创建画布
        fig, ax = plt.subplots(figsize=(6,6), dpi=100)

        # 1) 背景障碍图
        ax.imshow(self.grid, cmap='Greys', origin='lower',
                  extent=(0, self.W, 0, self.H))

        # 2) 距离热力图
        im = ax.imshow(
            self.dist_map,
            cmap='viridis',
            origin='lower',
            extent=(0, self.W, 0, self.H)
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Distance to goal')

        # 3) 绘制机器人和目标（目标在格子中心）
        self.world_cc.addXYThetaToPlot(ax, self.footprint, self.current_state, computeValidity=True)
        self.world_cc.addXYThetaToPlot(ax, self.footprint, self.goal_state,    computeValidity=True)

        # 4) 美化
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, self.W); ax.set_ylim(0, self.H)
        ax.set_aspect('equal')

        # 5) 保存到文件
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    env = PathFindingEnvWithMap()
    env.reset(seed=42)
    env.render("distance_map.png")
    print("✅ Saved distance_map.png")
