#!/usr/bin/env python3
"""
PathFindingEnvWithMap  —  坐标顺序全部统一 (row, col) / (y, x)

✅ 修复：
1. reset() 里调用 _bfs_distance() 时把 x,y 传反的问题。
2. _bfs_distance() 的参数名与注释改为 col_idx, row_idx，避免再次混淆。
"""

import copy
import heapq
from collections import deque
from typing import Optional

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# 环境依赖
from domains_cc.footprint import createFootprint
from domains_cc.map_generator import generate_random_map
from domains_cc.worldCC import WorldCollisionChecker

matplotlib.use("Agg")  # 服务器端渲染


class PathFindingEnvWithMap(gym.Env):
    metadata = {"render_modes": ["human"]}

    # --------------------------------------------------------------------- #
    #                               初始化                                   #
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        grid_size=(20, 20),
        obstacle_prob=0.1,
        footprint_spec=None,
        max_steps=300,
        prev_action_len: int = 1,
    ):
        super().__init__()

        # --- 地图与碰撞检查 ------------------------------------------------- #
        self.grid = generate_random_map(grid_size, obstacle_prob).astype(np.float32)
        self.world_cc = WorldCollisionChecker(self.grid)
        self.H, self.W = self.grid.shape  # (row, col)

        # --- 机器人外形 ------------------------------------------------------ #
        if footprint_spec is None:
            footprint_spec = dict(type="rectangle", width=0.5, height=1.5, resolution=0.1)
        self.footprint = createFootprint(
            footprint_spec["type"],
            dict(
                width=footprint_spec["width"],
                height=footprint_spec["height"],
                resolution=footprint_spec["resolution"],
            ),
        )

        # --- 离散动作 (dx, dy, dtheta) -------------------------------------- #
        self.discrete_actions = [
            np.array([0.25, 0.0, 0.0], np.float32),
            np.array([-0.25, 0.0, 0.0], np.float32),
            np.array([0.0, 0.25, 0.0], np.float32),
            np.array([0.0, -0.25, 0.0], np.float32),
            np.array([0.0, 0.0, 3.5], np.float32),
            np.array([0.0, 0.0, -3.5], np.float32),
        ]
        self.action_space = spaces.Discrete(len(self.discrete_actions))

        # --- 观测空间 -------------------------------------------------------- #
        self.prev_action_len = prev_action_len
        self.prev_actions = deque(maxlen=self.prev_action_len)
        self.observation_space = spaces.Dict(
            {
                "vec_obs": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
                "next_obs": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
                "action_mask": spaces.MultiBinary(len(self.discrete_actions)),
                "prev_actions": spaces.Box(
                    low=-1,
                    high=len(self.discrete_actions) - 1,
                    shape=(self.prev_action_len,),
                    dtype=np.int64,
                ),
            }
        )

        # --- 其它 ------------------------------------------------------------ #
        self.max_steps = max_steps
        self.current_state = np.zeros(3, np.float32)
        self.goal_state = np.zeros(3, np.float32)
        self.step_count = 0
        self.dist_map = np.full((self.H, self.W), np.inf, np.float32)
        self.flow_field = np.zeros((self.H, self.W, 2), np.float32)

    # --------------------------------------------------------------------- #
    #                               重置                                     #
    # --------------------------------------------------------------------- #
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # --- 随机采样起点 --------------------------------------------------- #
        def sample_pose():
            return np.array(
                [
                    np.random.uniform(0, self.W),  # x  → col
                    np.random.uniform(0, self.H),  # y  → row
                    np.random.uniform(0, 360),
                ],
                np.float32,
            )

        while True:
            s = sample_pose()
            if self.world_cc.isValid(self.footprint, s.reshape(1, 3))[0]:
                break

        # --- 随机采样终点（网格中心） --------------------------------------- #
        while True:
            row = np.random.randint(0, self.H)
            col = np.random.randint(0, self.W)
            theta = np.random.uniform(0, 360)
            g = np.array([col + 0.5, row + 0.5, theta], np.float32)
            if self.world_cc.isValid(self.footprint, g.reshape(1, 3))[0] and np.linalg.norm(
                g[:2] - s[:2]
            ) > 0.1:
                break

        self.current_state = s.copy()
        self.goal_state = g.copy()
        self.step_count = 0

        # --- 重新计算距离图 & 流场 (先 col, 后 row) -------------------------- #
        col_idx = int(self.goal_state[0])  # x
        row_idx = int(self.goal_state[1])  # y
        self.dist_map = self._bfs_distance(self.grid, col_idx, row_idx)
        self.flow_field = self._build_flow_field(self.dist_map)

        # --- prev_actions 清空 --------------------------------------------- #
        self.prev_actions.clear()
        for _ in range(self.prev_action_len):
            self.prev_actions.append(-1)

        obs, _ = self._get_obs()
        return obs, {}

    # --------------------------------------------------------------------- #
    #                       BFS & FLOW FIELD 生成                            #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _bfs_distance(grid: np.ndarray, col_idx: int, row_idx: int) -> np.ndarray:
        """
        计算从目标格 (col_idx, row_idx)（即 (x_int, y_int)）到所有格子的曼哈顿距离。
        """
        H, W = grid.shape
        INF = np.inf
        dist = np.full((H, W), INF, np.float32)
        free = grid < 0.5  # True 表示可通行

        dq = deque()
        if free[row_idx, col_idx]:
            dist[row_idx, col_idx] = 0
            dq.append((row_idx, col_idx))

        neighs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (drow, dcol)
        while dq:
            r, c = dq.popleft()
            for dr, dc in neighs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and free[nr, nc] and not np.isfinite(dist[nr, nc]):
                    dist[nr, nc] = dist[r, c] + 1
                    dq.append((nr, nc))

        return dist

    @staticmethod
    def _build_flow_field(dist: np.ndarray) -> np.ndarray:
        H, W = dist.shape
        flow = np.zeros((H, W, 2), np.float32)
        neighs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for row in range(H):
            for col in range(W):
                d0 = dist[row, col]
                if not np.isfinite(d0):
                    continue
                best, vec = d0, None
                for dr, dc in neighs:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < H and 0 <= nc < W and dist[nr, nc] < best:
                        best = dist[nr, nc]
                        vec = (dc, dr)  # (dx, dy)
                if vec is not None:
                    dx, dy = vec
                    norm = np.hypot(dx, dy) + 1e-8
                    flow[row, col, 0] = dx / norm
                    flow[row, col, 1] = dy / norm
        return flow

    # --------------------------------------------------------------------- #
    #                              工具函数                                  #
    # --------------------------------------------------------------------- #
    def _grid_pos(self, state):
        """将连续坐标转换为格索引 (row, col)"""
        return int(state[1]), int(state[0])

    # --------------------------------------------------------------------- #
    #                               观测                                     #
    # --------------------------------------------------------------------- #
    def _get_obs(self):
        # --- 向目标矢量 ---------------------------------------------------- #
        dx = (self.goal_state[0] - self.current_state[0]) / self.W
        dy = (self.goal_state[1] - self.current_state[1]) / self.H
        err = (self.goal_state[2] - self.current_state[2] + 180) % 360 - 180
        vec_obs = np.array([dx, dy, np.sin(np.deg2rad(err)), np.cos(np.deg2rad(err))], np.float32)

        # --- 当下流向 ------------------------------------------------------ #
        row, col = self._grid_pos(self.current_state)
        fx, fy = self.flow_field[row, col]
        angle = np.arctan2(fy, fx)
        next_obs = np.array([fx, fy, np.sin(angle), np.cos(angle)], np.float32)

        # --- 动作合法性掩码 ----------------------------------------------- #
        mask = np.array(
            [
                self.world_cc.isValidEdge(
                    self.footprint,
                    self.current_state.reshape(1, 3),
                    (self.current_state + mv).reshape(1, 3),
                )[0]
                for mv in self.discrete_actions
            ],
            dtype=bool,
        )

        prev_arr = np.array(self.prev_actions, dtype=np.int64)
        return {
            "vec_obs": vec_obs,
            "next_obs": next_obs,
            "action_mask": mask,
            "prev_actions": prev_arr,
        }, None

    # --------------------------------------------------------------------- #
    #                               step                                    #
    # --------------------------------------------------------------------- #
    def step(self, action: int):
        prev = self.current_state.copy()
        self.current_state = prev + self.discrete_actions[action]
        self.step_count += 1

        # 奖励：A* 路径长度差
        pd = len(self._astar_path(prev, self.goal_state))
        nd = len(self._astar_path(self.current_state, self.goal_state))
        reward = -0.1 + float(pd - nd)

        # 终止判定
        d = np.linalg.norm(self.current_state[:2] - self.goal_state[:2])
        err = (self.current_state[2] - self.goal_state[2] + 180) % 360 - 180
        done = False
        info = {}
        if d < 0.5 and abs(err) < 10:
            done = True
            reward += 100.0
            info["info"] = "reached_goal"
        elif self.step_count >= self.max_steps:
            done = True

        self.prev_actions.append(action)
        obs, _ = self._get_obs()
        return obs, reward, done, False, info

    def action_masks(self):
        return self._get_obs()[0]["action_mask"]

    # --------------------------------------------------------------------- #
    #                               渲染                                    #
    # --------------------------------------------------------------------- #
    def render(self):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.imshow(self.grid, cmap="Greys", origin="lower", extent=(0, self.W, 0, self.H))

        xs = np.arange(self.W) + 0.5
        ys = np.arange(self.H) + 0.5
        X, Y = np.meshgrid(xs, ys)
        U = self.flow_field[Y.astype(int), X.astype(int), 0]
        V = self.flow_field[Y.astype(int), X.astype(int), 1]

        ax.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=1, width=0.003, alpha=0.7)
        self.world_cc.addXYThetaToPlot(ax, self.footprint, self.current_state, True)
        self.world_cc.addXYThetaToPlot(ax, self.footprint, self.goal_state, True)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, self.W)
        ax.set_ylim(0, self.H)

    # --------------------------------------------------------------------- #
    #                               A*                                      #
    # --------------------------------------------------------------------- #
    def _astar_path(self, start, goal):
        start_rc, goal_rc = self._grid_pos(start), self._grid_pos(goal)  # (row, col)
        open_set = [
            (
                abs(start_rc[0] - goal_rc[0]) + abs(start_rc[1] - goal_rc[1]),
                0,
                start_rc,
            )
        ]
        came, g_score, visited = {}, {start_rc: 0}, set()
        while open_set:
            _, cost, cur = heapq.heappop(open_set)
            if cur in visited:
                continue
            visited.add(cur)
            if cur == goal_rc:
                break
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nb = (cur[0] + dr, cur[1] + dc)
                if not (0 <= nb[0] < self.H and 0 <= nb[1] < self.W):
                    continue
                if self.grid[nb[0], nb[1]] > 0.5:
                    continue
                nc = cost + 1
                if nc < g_score.get(nb, 1e9):
                    came[nb] = cur
                    g_score[nb] = nc
                    heapq.heappush(
                        open_set,
                        (
                            nc + abs(nb[0] - goal_rc[0]) + abs(nb[1] - goal_rc[1]),
                            nc,
                            nb,
                        ),
                    )
        path = []
        node = goal_rc
        while node != start_rc and node in came:
            path.append(node)
            node = came[node]
        if node == start_rc:
            path.append(start_rc)
        return list(reversed(path))
