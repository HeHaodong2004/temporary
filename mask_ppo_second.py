#!/usr/bin/env python3
"""
Maskable PPO for PathFindingEnvWithMap with SB3 v2.6.0 & sb3-contrib v2.6.0
- 离散动作
- Dict 观测 {"obs_vec": Box, "action_mask": MultiBinary}
- 提供 action_masks() 接口
- VecNormalize 仅归一化 "obs_vec"
- SubprocVecEnv 真并行采样
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import heapq
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use('Agg')

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from sb3_contrib import MaskablePPO

# 你的工程依赖
from domains_cc.footprint import createFootprint
from domains_cc.map_generator import generate_random_map
from domains_cc.worldCC import WorldCollisionChecker

class PathFindingEnvWithMap(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 grid_size=(20, 20),
                 map_obstacle_prob=0.2,
                 footprint_spec=None,
                 max_steps=200):
        super().__init__()
        if footprint_spec is None:
            footprint_spec = dict(type="rectangle", width=0.5,
                                  height=1.5, resolution=0.05)
        # 地图与碰撞检查
        self.grid = generate_random_map(grid_size,
                                        obstacle_prob=map_obstacle_prob).astype(np.float32)
        self.world_cc = WorldCollisionChecker(self.grid)
        self.footprint = createFootprint(
            footprint_spec["type"],
            dict(width=footprint_spec["width"],
                 height=footprint_spec["height"],
                 resolution=footprint_spec["resolution"])
        )
        H, W = grid_size
        self.max_astar = H + W  # A* 最大距离上限
        # 离散动作集
        self.discrete_actions = [
            np.array([0.5, 0.0, 0.0], np.float32),
            np.array([-0.5, 0.0, 0.0], np.float32),
            np.array([0.0, 0.5, 0.0], np.float32),
            np.array([0.0, -0.5, 0.0], np.float32),
            np.array([0.0, 0.0, 10.0], np.float32),
            np.array([0.0, 0.0, -10.0], np.float32),
        ]
        self.action_space = spaces.Discrete(len(self.discrete_actions))
        # 定义观测空间
        mask_dim = len(self.discrete_actions)
        low = np.array([-1.0, -1.0, -1.0, -1.0] + [0]*mask_dim, dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0] + [1]*mask_dim, dtype=np.float32)
        self.observation_space = spaces.Dict({
            "obs_vec": spaces.Box(low=low, high=high, dtype=np.float32),
            "action_mask": spaces.MultiBinary(mask_dim),
        })
        self.max_steps = max_steps
        self.current_state = np.zeros(3, np.float32)
        self.goal_state = np.zeros(3, np.float32)
        self.initial_dist = 0.0
        self.step_count = 0

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            np.random.seed(seed)
        H, W = self.grid.shape
        def sample():
            return np.array([np.random.uniform(0, W),
                             np.random.uniform(0, H),
                             np.random.uniform(0, 360)], np.float32)
        # 随机合法起点/目标
        while True:
            s = sample()
            if self.world_cc.isValid(self.footprint, s.reshape(1,3))[0]: break
        while True:
            g = sample()
            if self.world_cc.isValid(self.footprint, g.reshape(1,3))[0] and np.linalg.norm(g[:2]-s[:2])>0.1:
                break
        self.current_state, self.goal_state = s.copy(), g.copy()
        self.initial_dist = np.linalg.norm(s[:2] - g[:2])
        self.step_count = 0
        return self._get_obs(), {}

    def _grid_pos(self, state):
        x,y,_ = state
        return (int(x), int(y))

    def _astar_distance(self, start, goal):
        start, goal = self._grid_pos(start), self._grid_pos(goal)
        H,W = self.grid.shape
        def h(p): return abs(p[0]-goal[0])+abs(p[1]-goal[1])
        open_set = [(h(start),0,start)]
        g_score = {start:0}
        visited = set()
        while open_set:
            _,g,pos = heapq.heappop(open_set)
            if pos in visited: continue
            visited.add(pos)
            if pos==goal: return g
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nb=(pos[0]+dx,pos[1]+dy)
                if not(0<=nb[0]<W and 0<=nb[1]<H): continue
                if self.grid[nb[1],nb[0]]>0.5: continue
                tg=g+1
                if tg<g_score.get(nb,float('inf')):
                    g_score[nb]=tg
                    heapq.heappush(open_set,(tg+h(nb),tg,nb))
        return self.max_astar

    def _get_obs(self):
        dx,dy = self.goal_state[:2]-self.current_state[:2]
        dx/=self.grid.shape[1]; dy/=self.grid.shape[0]
        err=(self.goal_state[2]-self.current_state[2]+180)%360-180
        sin_e,cos_e = np.sin(np.deg2rad(err)), np.cos(np.deg2rad(err))
        mask = np.array([self.world_cc.isValidEdge(self.footprint,
                self.current_state.reshape(1,3),(self.current_state+move).reshape(1,3))[0]
                for move in self.discrete_actions],bool)
        obs_vec = np.concatenate([[dx,dy,sin_e,cos_e],mask.astype(np.float32)])
        return {"obs_vec":obs_vec.astype(np.float32),"action_mask":mask}

    def step(self, action:int):
        prev=self.current_state.copy()
        self.current_state=prev+self.discrete_actions[action]
        self.step_count+=1
        # 时间惩罚 + A* 进度
        tp=-0.1
        pd,nd=self._astar_distance(prev,self.goal_state),self._astar_distance(self.current_state,self.goal_state)
        ar=float(pd-nd)
        # 成功
        d=np.linalg.norm(self.current_state[:2]-self.goal_state[:2])
        err=(self.current_state[2]-self.goal_state[2]+180)%360-180
        if d<0.01 and abs(err)<5:
            return self._get_obs(),100.0,True,False,{"info":"reached_goal"}
        reward=tp+ar
        return self._get_obs(),reward,False,self.step_count>=self.max_steps,{}

    def action_masks(self): return self._get_obs()["action_mask"]

    def render(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        fig, ax = plt.subplots(figsize=(6,6), dpi=100)
        # 2) 先画网格
        ax.imshow(self.grid, cmap='Greys', origin='lower',
                  extent=(0, self.grid.shape[1], 0, self.grid.shape[0]),
                  interpolation='nearest')

        # 3) 然后用 world_cc 直接把“车身”画出来
        #    addXYThetaToPlot 会根据 self.current_state 的 (x,y,θ) 旋转+平移 footprint
        self.world_cc.addXYThetaToPlot(ax, self.footprint,
                                       self.current_state, computeValidity=True)

        # 4) 画目标点（也用同一个函数，或 scatter）
        self.world_cc.addXYThetaToPlot(ax, self.footprint,
                                        self.goal_state, computeValidity=False)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, self.grid.shape[1])
        ax.set_ylim(0, self.grid.shape[0])
        ax.set_aspect('equal')

        # 5) 转成 RGB 数组
        canvas = FigureCanvas(fig)
        canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
        plt.close(fig)
        return img
    
# 环境工厂
def make_env(): return PathFindingEnvWithMap()

if __name__ == "__main__":
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
    from sb3_contrib import MaskablePPO

    # 学习率调度：线性衰减
    def linear_schedule(initial_value):
        def scheduler(progress_remaining):
            return initial_value * progress_remaining
        return scheduler

    # 1. 并行训练环境
    train_env = SubprocVecEnv([make_env] * 32)
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True,
        clip_obs=10.0, norm_obs_keys=["obs_vec"]
    )

    # 2. 单进程评估环境（必须是 VecEnv）
    eval_env = DummyVecEnv([make_env])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(
        eval_env, training=False, norm_obs=True, norm_reward=True,
        clip_obs=10.0, norm_obs_keys=["obs_vec"]
    )

    # 3. 早停回调（但别单独放到列表里）
    early_stop = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=5,
        verbose=1
    )

    # 4. EvalCallback：挂载最佳模型保存 + 早停
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_maskppo/",
        log_path="./logs/eval/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
        callback_on_new_best=early_stop,    # ← 把早停挂这里
    )

    callbacks = CallbackList([eval_callback])

    # 5. 创建并训练模型
    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=train_env,
        verbose=1,
        batch_size=256,
        learning_rate=linear_schedule(3e-4),
        tensorboard_log="./tensorboard_logs/",
        device="auto"
    )

    model.learn(total_timesteps=2_000_000, callback=callbacks)

    # 6. 手动保存最终模型
    model.save("maskableppo_pathfinder_final")
    print("✅ 训练完成，模型保存在 maskableppo_pathfinder_final.zip")
