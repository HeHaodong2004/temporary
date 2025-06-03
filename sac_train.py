import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from domains_cc.footprint import createFootprint
from domains_cc.map_generator import generate_random_map
from domains_cc.worldCC import WorldCollisionChecker  

from typing import Optional

class PathFindingEnvWithMap(gym.Env):
    """
    单智能体路径规划环境，观测包含全局地图、当前姿态和目标姿态，动作为连续 dx, dy, dtheta。
    现在继承自 gymnasium.Env，符合 Gymnasium API。
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 grid_size=(20, 20),
                 map_obstacle_prob=0.2,
                 footprint_spec={'type': 'rectangle', 'width': 0.5, 'height': 1.5, 'resolution': 0.05},
                 max_steps: int = 200):
        super().__init__()

        # 1) 随机生成地图并初始化碰撞检查器
        self.grid = generate_random_map(grid_size, obstacle_prob=map_obstacle_prob).astype(np.float32)
        self.world_cc = WorldCollisionChecker(self.grid)

        # 2) 生成机器人足迹
        self.footprint = createFootprint(
            footprint_spec['type'],
            {'width': footprint_spec['width'],
             'height': footprint_spec['height'],
             'resolution': footprint_spec['resolution']}
        )

        H, W = grid_size

        # 3) 连续动作空间：dx, dy ∈ [-0.5, 0.5], dtheta ∈ [-10°, 10°]
        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.5, -10.0], dtype=np.float32),
            high=np.array([ 0.5,  0.5,  10.0], dtype=np.float32),
            dtype=np.float32
        )

        # 4) 观测空间：map_flatten (H*W) + cur(x,y,theta) + goal(xg,yg,thetag)
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

        # 5) 其他属性
        self.max_steps = max_steps
        self.step_count = 0
        self.current_state = np.zeros(3, dtype=np.float32)
        self.goal_state = np.zeros(3, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Gymnasium 版 reset，需要接收 seed 和 options，并返回 (obs, info)。
        """
        if seed is not None:
            np.random.seed(seed)

        H, W = self.grid.shape

        # 随机采样合法起点
        while True:
            x0 = np.random.uniform(0, W)
            y0 = np.random.uniform(0, H)
            theta0 = np.random.uniform(0, 360)
            cand = np.array([x0, y0, theta0], dtype=np.float32)
            if self.world_cc.isValid(self.footprint, cand.reshape(1, 3))[0]:
                break

        # 随机采样合法目标
        while True:
            xg = np.random.uniform(0, W)
            yg = np.random.uniform(0, H)
            thetag = np.random.uniform(0, 360)
            cand_goal = np.array([xg, yg, thetag], dtype=np.float32)
            if self.world_cc.isValid(self.footprint, cand_goal.reshape(1, 3))[0]:
                if np.linalg.norm(cand_goal[:2] - cand[:2]) > 0.1:
                    break

        self.current_state = cand
        self.goal_state = cand_goal
        self.step_count = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self) -> np.ndarray:
        map_flat = self.grid.reshape(-1)
        return np.concatenate([map_flat, self.current_state, self.goal_state], axis=0)

    def step(self, action: np.ndarray):
        """
        Gymnasium 版 step，需要返回 (obs, reward, terminated, truncated, info)。
        """
        dx, dy, dtheta = action.astype(np.float32)
        next_state = self.current_state + np.array([dx, dy, dtheta], dtype=np.float32)

        valid_edge = self.world_cc.isValidEdge(
            self.footprint,
            self.current_state.reshape(1, 3),
            next_state.reshape(1, 3)
        )[0]

        # 如果发生碰撞
        if not valid_edge:
            reward = -100.0
            terminated = True
            truncated = False
            info = {"info": "collision"}
            obs = self._get_obs()
            return obs, reward, terminated, truncated, info

        # 更新状态
        self.current_state = next_state
        self.step_count += 1

        # 计算到目标的距离 和 角度差
        pos_dist = np.linalg.norm(self.current_state[:2] - self.goal_state[:2])
        ang_diff = (self.current_state[2] - self.goal_state[2] + 180.0) % 360.0 - 180.0
        ang_dist = abs(ang_diff)

        # 如果到达目标
        if (pos_dist < 0.2) and (ang_dist < 5.0):
            reward = 100.0
            terminated = True
            truncated = False
            info = {"info": "reached_goal"}
            obs = self._get_obs()
            return obs, reward, terminated, truncated, info

        # 正常行走的惩罚
        reward = -pos_dist
        terminated = False
        truncated = False
        info = {}

        # 如果超过最大步数，就算截断
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    # 创建环境实例
    env = PathFindingEnvWithMap(
        grid_size=(20, 20),
        map_obstacle_prob=0.2,
        footprint_spec={'type': 'rectangle', 'width': 0.5, 'height': 1.5, 'resolution': 0.05},
        max_steps=200
    )

    # 检查环境接口：check_env 会内部执行 reset(seed=0) 并检查 step() 返回值
    check_env(env, warn=True)

    # 创建 SAC 模型
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        buffer_size=100_000,
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        device="auto"
    )

    # 训练
    total_timesteps = 200_000
    model.learn(total_timesteps=total_timesteps)

    # 保存模型
    model.save("sac_pathfinder_with_map")
    print("训练完成，模型保存在 sac_pathfinder_with_map.zip")
