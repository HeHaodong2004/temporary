这是一段使用maskable ppo进行机器人路径规划的代码，向我解释它#!/usr/bin/env python3
"""
Maskable PPO for PathFindingEnvWithMap with SB3 v2.6.0 & sb3-contrib v2.6.0
- 离散动作
- Dict 观测 {"observation": Box, "action_mask": MultiBinary}
- 提供 action_masks() 接口，无需 ActionMasker
- VecNormalize 仅归一化 "observation"
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
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

        # 构造 observation_space
        obs_low = np.concatenate([np.zeros(H * W, np.float32),
                                  np.zeros(3, np.float32),
                                  np.zeros(3, np.float32)])
        obs_high = np.concatenate([np.ones(H * W, np.float32),
                                   np.array([W, H, 360], np.float32),
                                   np.array([W, H, 360], np.float32)])
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
            "action_mask": spaces.MultiBinary(len(self.discrete_actions)),
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

        # 合法起点
        while True:
            s = sample()
            if self.world_cc.isValid(self.footprint, s.reshape(1, 3))[0]:
                break
        # 合法目标
        while True:
            g = sample()
            if (self.world_cc.isValid(self.footprint, g.reshape(1, 3))[0]
                    and np.linalg.norm(g[:2] - s[:2]) > 0.1):
                break

        self.current_state, self.goal_state = s.copy(), g.copy()
        self.initial_dist = np.linalg.norm(s[:2] - g[:2])
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        map_flat = self.grid.reshape(-1)
        obs_vec = np.concatenate([map_flat,
                                  self.current_state,
                                  self.goal_state]).astype(np.float32)

        mask = np.zeros(len(self.discrete_actions), dtype=bool)
        for i, move in enumerate(self.discrete_actions):
            nxt = self.current_state + move
            mask[i] = self.world_cc.isValidEdge(
                self.footprint,
                self.current_state.reshape(1, 3),
                nxt.reshape(1, 3)
            )[0]
        return {"observation": obs_vec, "action_mask": mask}

    def step(self, action: int):
        move = self.discrete_actions[action]
        prev_d = np.linalg.norm(self.current_state[:2] - self.goal_state[:2])

        nxt = self.current_state + move
        if not self.world_cc.isValidEdge(
                self.footprint,
                self.current_state.reshape(1, 3),
                nxt.reshape(1, 3)
        )[0]:
            return self._get_obs(), -10.0, True, False, {"info": "collision"}

        self.current_state = nxt
        self.step_count += 1

        d = np.linalg.norm(self.current_state[:2] - self.goal_state[:2])
        ang_err = (self.current_state[2] - self.goal_state[2] + 180) % 360 - 180
        if d < 0.2 and abs(ang_err) < 5:
            return self._get_obs(), 100.0, True, False, {"info": "reached_goal"}

        delta = prev_d - d
        reward = float(np.clip((delta / self.initial_dist) * 5 if self.initial_dist > 1e-6 else 0, -5, 5))
        truncated = self.step_count >= self.max_steps
        return self._get_obs(), reward, False, truncated, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    # MaskablePPO 需要的接口：返回可行动作掩码
    def action_masks(self):
        return self._get_obs()["action_mask"]


def make_env():
    return PathFindingEnvWithMap(
        grid_size=(20, 20),
        map_obstacle_prob=0.2,
        max_steps=200
    )


if __name__ == "__main__":
    # 1) 32 并行 + 监控
    base_env = DummyVecEnv([make_env for _ in range(32)])
    mon_env = VecMonitor(base_env)

    # 2) 只对 "observation" 做归一化
    vec_env = VecNormalize(
        mon_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        norm_obs_keys=["observation"]
    )

    # 3) 训练 MaskablePPO
    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        batch_size=256,
        tensorboard_log="./tensorboard_logs/",
        device="auto",
    )
    model.learn(total_timesteps=200_000)
    model.save("maskableppo_pathfinder")
    print("训练完成，模型保存在 maskableppo_pathfinder.zip")
