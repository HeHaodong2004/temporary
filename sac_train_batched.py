import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from domains_cc.footprint import createFootprint
from domains_cc.map_generator import generate_random_map
from domains_cc.worldCC import WorldCollisionChecker
from stable_baselines3.common.callbacks import EvalCallback  


class PathFindingEnvWithMap(gym.Env):
    """
    单实例 PathFinding 环境（与用户提供的版本一模一样）。
    这里省略 __init__, reset, _get_obs, step, render, close 方法的具体实现，
    假设已与用户代码完全一致，只展示类名以示替换。
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 grid_size=(20, 20),
                 map_obstacle_prob=0.2,
                 footprint_spec={'type': 'rectangle',
                                'width': 0.5,
                                'height': 1.5,
                                'resolution': 0.05},
                 max_steps: int = 200):
        super().__init__()

        self.grid = generate_random_map(grid_size, obstacle_prob=map_obstacle_prob).astype(np.float32)
        self.world_cc = WorldCollisionChecker(self.grid)

        self.footprint = createFootprint(
            footprint_spec['type'],
            {'width': footprint_spec['width'],
             'height': footprint_spec['height'],
             'resolution': footprint_spec['resolution']}
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
        return np.concatenate([map_flat,
                               self.current_state,
                               self.goal_state], axis=0)

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


'''if __name__ == "__main__":
    # ----- 1) 先测试一下单环境是否满足 Gym 接口 -----
    single_env = PathFindingEnvWithMap(
        grid_size=(20, 20),
        map_obstacle_prob=0.2,
        footprint_spec={'type': 'rectangle', 'width': 0.5, 'height': 1.5, 'resolution': 0.05},
        max_steps=200
    )
    # 如果你只检查单 env，可以保留下面一行；若检查向量化 env，就注释掉
    # from stable_baselines3.common.env_checker import check_env
    # check_env(single_env, warn=True)

    # ----- 2) 构造并行的 Vectorized Env -----
    num_parallel_envs = 64

    def make_env():
        return PathFindingEnvWithMap(
            grid_size=(20, 20),
            map_obstacle_prob=0.2,
            footprint_spec={'type': 'rectangle', 'width': 0.5, 'height': 1.5, 'resolution': 0.05},
            max_steps=200
        )

    # 把 4 个工厂函数放到列表里
    env_fns = [ (lambda: make_env()) for _ in range(num_parallel_envs) ]
    # DummyVecEnv 接受一个可调用对象列表，会各自创建独立的 env 实例
    vec_env = DummyVecEnv(env_fns)

    # ----- 3) 在向量化环境上训练 SAC -----
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        buffer_size=100_000,
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        device="auto"
    )
    model.learn(total_timesteps=200_000)
    model.save("sac_pathfinder_with_map_vecenv")
    print("训练完成，模型保存在 sac_pathfinder_with_map_vecenv.zip")'''
'''if __name__ == "__main__":
    # ----- 1) 构造 VecEnv 并套上 VecMonitor -----
    num_parallel_envs = 64

    def make_env():
        return PathFindingEnvWithMap(
            grid_size=(20, 20),
            map_obstacle_prob=0.2,
            footprint_spec={'type': 'rectangle', 'width': 0.5, 'height': 1.5, 'resolution': 0.05},
            max_steps=200
        )

    env_fns = [ (lambda: make_env()) for _ in range(num_parallel_envs) ]
    vec_env = DummyVecEnv(env_fns)
    # 用 VecMonitor 包装，让它自动记录 episode reward、episode length、is_success 等信息
    vec_env = VecMonitor(venv=vec_env)  

    # ----- 2) 配置 SAC 并开启 TensorBoard 日志 -----
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        verbose=0,
        buffer_size=100_000,
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        tensorboard_log="./tensorboard_logs/",  # ← 开启 TensorBoard 日志
        device="auto",
    )

    # ----- 3) 开始训练 ----- 
    model.learn(total_timesteps=200_000)
    model.save("sac_pathfinder_with_map_vecenv")
    print("训练完成，模型保存在 sac_pathfinder_with_map_vecenv.zip")'''
    
if __name__ == "__main__":
    # ----- 1) 构造 VecEnv 并套上 VecMonitor -----
    num_parallel_envs = 96

    def make_env():
        return PathFindingEnvWithMap(
            grid_size=(20, 20),
            map_obstacle_prob=0.2,
            footprint_spec={'type': 'rectangle', 'width': 0.5, 'height': 1.5, 'resolution': 0.05},
            max_steps=200
        )

    env_fns = [ (lambda: make_env()) for _ in range(num_parallel_envs) ]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(venv=vec_env)  # 自动记录每个 episode 的 reward, length, success

    # ----- 2) 构造专门的评估环境，套上 VecMonitor -----
    eval_env = DummyVecEnv([lambda: PathFindingEnvWithMap(
        grid_size=(20, 20),
        map_obstacle_prob=0.2,
        footprint_spec={'type': 'rectangle', 'width': 0.5, 'height': 1.5, 'resolution': 0.05},
        max_steps=200
    )])
    eval_env = VecMonitor(venv=eval_env)

    # ----- 3) 配置 EvalCallback -----
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=5000,        # 每 5000 环境步评估一次
        n_eval_episodes=20,    # 每次评估跑 20 个 episode
        deterministic=True,
        render=False
    )

    # ----- 4) 配置 SAC 并开启 TensorBoard 日志（verbose=0） -----
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        verbose=0,
        buffer_size=100_000,
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        tensorboard_log="./tensorboard_logs/",
        device="auto"
    )

    # ----- 5) 开始训练，并传入 eval_callback -----
    model.learn(total_timesteps=200_000, callback=eval_callback)
    model.save("sac_pathfinder_with_map_vecenv")
    print("训练完成，模型保存在 sac_pathfinder_with_map_vecenv.zip")

