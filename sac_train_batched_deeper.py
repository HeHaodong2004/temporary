import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import torch
from torch import nn

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from domains_cc.footprint import createFootprint
from domains_cc.map_generator import generate_random_map
from domains_cc.worldCC import WorldCollisionChecker


class PathFindingEnvWithMap(gym.Env):
    """
    单实例 PathFinding 环境（与你现有实现相同）
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_size=(20, 20),
        map_obstacle_prob=0.2,
        footprint_spec={'type': 'rectangle', 'width': 0.5, 'height': 1.5, 'resolution': 0.05},
        max_steps: int = 200
    ):
        super().__init__()

        self.grid = generate_random_map(grid_size, obstacle_prob=map_obstacle_prob).astype(np.float32)
        self.world_cc = WorldCollisionChecker(self.grid)

        self.footprint = createFootprint(
            footprint_spec['type'],
            {
                'width': footprint_spec['width'],
                'height': footprint_spec['height'],
                'resolution': footprint_spec['resolution']
            }
        )

        H, W = grid_size
        self.H, self.W = H, W

        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.5, -10.0], dtype=np.float32),
            high=np.array([ 0.5,  0.5,  10.0], dtype=np.float32),
            dtype=np.float32
        )

        low_obs = np.concatenate([
            np.zeros(H * W, dtype=np.float32),
            np.array([0.0, 0.0,   0.0], dtype=np.float32),
            np.array([0.0, 0.0,   0.0], dtype=np.float32),
        ])
        high_obs = np.concatenate([
            np.ones(H * W, dtype=np.float32),
            np.array([W,   H, 360.0], dtype=np.float32),
            np.array([W,   H, 360.0], dtype=np.float32),
        ])
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float32
        )

        self.max_steps = max_steps
        self.step_count = 0
        self.current_state = np.zeros(3, dtype=np.float32)
        self.goal_state    = np.zeros(3, dtype=np.float32)
        self.initial_dist  = 0.0
        self.distance_map = np.full((H, W), np.inf, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)

        # 采样起点
        while True:
            x0 = np.random.uniform(0, self.W)
            y0 = np.random.uniform(0, self.H)
            θ0 = np.random.uniform(0, 360)
            cand = np.array([x0, y0, θ0], dtype=np.float32)
            if self.world_cc.isValid(self.footprint, cand.reshape(1, 3))[0]:
                break

        # 采样目标
        while True:
            xg = np.random.uniform(0, self.W)
            yg = np.random.uniform(0, self.H)
            θg = np.random.uniform(0, 360)
            cand_goal = np.array([xg, yg, θg], dtype=np.float32)
            if self.world_cc.isValid(self.footprint, cand_goal.reshape(1, 3))[0]:
                if np.linalg.norm(cand_goal[:2] - cand[:2]) > 0.1:
                    break

        self.current_state = cand.copy()
        self.goal_state    = cand_goal.copy()
        self.step_count    = 0
        self.initial_dist  = np.linalg.norm(self.current_state[:2] - self.goal_state[:2])

        H, W = self.H, self.W
        self.distance_map = np.full((H, W), np.inf, dtype=np.float32)
        goal_i = int(np.clip(np.floor(self.goal_state[1]), 0, H - 1))
        goal_j = int(np.clip(np.floor(self.goal_state[0]), 0, W - 1))
        if self.grid[goal_i, goal_j] == 0.0:
            from collections import deque
            queue = deque()
            self.distance_map[goal_i, goal_j] = 0.0
            queue.append((goal_i, goal_j))
            while queue:
                i, j = queue.popleft()
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < H) and (0 <= nj < W):
                        if (self.grid[ni, nj] == 0.0) and (self.distance_map[ni, nj] == np.inf):
                            self.distance_map[ni, nj] = self.distance_map[i, j] + 1.0
                            queue.append((ni, nj))

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        map_flat = self.grid.reshape(-1)
        return np.concatenate([map_flat, self.current_state, self.goal_state], axis=0)

    def step(self, action: np.ndarray):
        prev_x, prev_y, prev_θ = self.current_state
        prev_dist = np.linalg.norm(self.current_state[:2] - self.goal_state[:2])
        prev_i = int(np.clip(np.floor(prev_y), 0, self.H - 1))
        prev_j = int(np.clip(np.floor(prev_x), 0, self.W - 1))
        prev_pot = self.distance_map[prev_i, prev_j]
        if not np.isfinite(prev_pot):
            prev_pot = float(self.H + self.W)

        dx, dy, dθ = action.astype(np.float32)
        proposed_state = self.current_state + np.array([dx, dy, dθ], dtype=np.float32)

        valid_edge = self.world_cc.isValidEdge(
            self.footprint,
            self.current_state.reshape(1, 3),
            proposed_state.reshape(1, 3)
        )[0]

        # 碰撞处理
        if not valid_edge:
            r_collision = -10.0
            r_time      = -0.01
            reward      = r_collision + r_time

            self.step_count += 1
            terminated = False
            truncated  = False
            if self.step_count >= self.max_steps:
                truncated = True

            obs  = self._get_obs()
            info = {"info": "collision"}
            return obs, reward, terminated, truncated, info

        # 接受新状态
        self.current_state = proposed_state.copy()
        self.step_count   += 1

        new_dist = np.linalg.norm(self.current_state[:2] - self.goal_state[:2])
        new_i = int(np.clip(np.floor(self.current_state[1]), 0, self.H - 1))
        new_j = int(np.clip(np.floor(self.current_state[0]), 0, self.W - 1))
        new_pot = self.distance_map[new_i, new_j]
        if not np.isfinite(new_pot):
            new_pot = float(self.H + self.W)

        ang_diff = (self.current_state[2] - self.goal_state[2] + 180.0) % 360.0 - 180.0
        if (new_dist < 0.2) and (abs(ang_diff) < 5.0):
            reward     = 150.0
            terminated = True
            truncated  = False
            obs  = self._get_obs()
            info = {"info": "reached_goal"}
            return obs, reward, terminated, truncated, info

        # 中间奖励计算
        if self.initial_dist > 1e-6:
            r_dist = (prev_dist - new_dist) / self.initial_dist
        else:
            r_dist = 0.0
        α = 5.0
        r_dist *= α

        r_shape = (prev_pot - new_pot) / float(self.H + self.W)
        r_time  = -0.01
        reward  = r_dist + r_shape + r_time
        reward  = float(np.clip(reward, -10.0, 10.0))

        terminated = False
        truncated  = False
        if self.step_count >= self.max_steps:
            truncated = True

        obs  = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class UltraDeepMapExtractor(BaseFeaturesExtractor):
    """
    适当深度的卷积 + 全连接特征提取器（为 3090 调整后版本）
    """
    def __init__(self, observation_space, feature_dim=512):
        super().__init__(observation_space, feature_dim)
        total_dim = observation_space.shape[0]
        H = W = int((total_dim - 6) ** 0.5)
        self.map_shape = (1, H, W)

        # 4 层卷积：通道数逐步增加到 256
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),   # (B,32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (B,64,20,20)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (B,128,20,20)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),# (B,256,20,20)
            nn.ReLU(),
            nn.Flatten()                                            # (B,256*20*20=102400)
        )
        # 卷积后将 102400 维降到 (feature_dim - 6)=506 维
        with torch.no_grad():
            dummy = torch.zeros(1, *self.map_shape)
            conv_out = self.cnn(dummy)
            conv_dim = conv_out.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(conv_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, feature_dim - 6),
            nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        map_flat = obs[:, : self.map_shape[1] * self.map_shape[2]]
        map_tensor = map_flat.view(batch_size, *self.map_shape)
        conv_feat = self.cnn(map_tensor)
        conv_feat = self.linear(conv_feat)
        curr_goal = obs[:, -6:]
        return torch.cat([conv_feat, curr_goal], dim=1)


if __name__ == "__main__":
    # ----- 1) 并行环境设置 -----
    # 3090 单卡约 24GB 显存，使用 32 个并行进程比较稳妥
    num_parallel_envs = 32
    env_fns = [
        (lambda: PathFindingEnvWithMap(
            grid_size=(20, 20),
            map_obstacle_prob=0.2,
            footprint_spec={'type': 'rectangle', 'width': 0.5, 'height': 1.5, 'resolution': 0.05},
            max_steps=200
        )) for _ in range(num_parallel_envs)
    ]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ----- 2) 评估环境 -----
    eval_env = DummyVecEnv([
        lambda: PathFindingEnvWithMap(
            grid_size=(20, 20),
            map_obstacle_prob=0.2,
            footprint_spec={'type': 'rectangle', 'width': 0.5, 'height': 1.5, 'resolution': 0.05},
            max_steps=200
        )
    ])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ----- 3) 准备回调：评估 + 早停 -----
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=20,
        verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path="./best_model_3090/",
        log_path="./eval_logs_3090/",
        eval_freq=5_000,        # 比之前更频繁地评估（每 5000 步）
        n_eval_episodes=20,     # 每次评估 20 个 episode
        deterministic=True,
        render=False,
        verbose=1
    )

    # ----- 4) 定义网络结构 -----
    policy_kwargs = dict(
        features_extractor_class=UltraDeepMapExtractor,
        features_extractor_kwargs=dict(feature_dim=512),  # 提取后特征 512 维
        net_arch=[512, 512, 256],                         # Actor/Critic 头：512→512→256
        activation_fn=nn.ReLU
    )

    # ----- 5) 创建 SAC（针对 3090 优化） -----
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,

        # —— 回放 & 批量尺寸 ——  
        buffer_size=int(1e6),      # 1M 容量
        batch_size=512,            # 每次梯度更新用 512 条样本

        # —— 目标网络 & 折扣因子 & 熵系数 ——  
        tau=0.005,                 # 软更新速率 0.005
        gamma=0.995,               # 折扣因子 0.995
        ent_coef="auto",           # 自动学习熵系数

        # —— 训练频率 & 梯度步数 ——  
        train_freq=(1, "step"),    # 每一步都更新
        gradient_steps=32,         # 每次更新 32 次梯度

        # —— 学习率 ——  
        learning_rate=3e-4,       # 经典的 3e-4，3090 上够用

        # —— 网络结构 & 其他 ——  
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard_logs_3090/",
        device="cuda"
    )

    # ----- 6) 开始训练 -----
    # 3090 训练速度较快，这里先跑 1e6 步；如果你想跑更长，可以改成 2e6 / 5e6
    model.learn(
        total_timesteps=int(1e6),
        callback=eval_callback
    )

    # ----- 7) 保存模型 -----
    model.save("sac_3090_pathfinder")
    print("训练完成，模型保存在 sac_3090_pathfinder.zip")
