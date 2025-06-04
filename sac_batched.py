import numpy as np
import torch
from torch import nn

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from domains_cc.footprint import createFootprint
from domains_cc.map_generator import generate_random_map
from domains_cc.worldCC import WorldCollisionChecker


# ================================================
# 1) BatchedPathFindingEnv：增加了 observation_space & action_space 属性
# ================================================
class BatchedPathFindingEnv(gym.Env):
    """
    一个“批量”版 PathFinding 环境，可同时处理 K 个实例。
    当 num_envs=1 时，也可以被 DummyVecEnv 包装为单环境使用。
    """

    def __init__(
        self,
        num_envs: int,
        grid_size=(20, 20),
        map_obstacle_prob=0.2,
        footprint_spec={'type': 'rectangle', 'width': 0.5, 'height': 1.5, 'resolution': 0.05},
        max_steps: int = 200
    ):
        super().__init__()
        self.num_envs = num_envs
        self.H, self.W = grid_size
        self.max_steps = max_steps

        # ------------------------
        # 新增：单个子环境的 observation_space 与 action_space
        # ------------------------
        # 对于单环境（num_envs=1），它的观测是：
        #   image_obs: shape=(6, H, W), 值域 [0,1]
        #   vector_obs: shape=(8,), 对应 [dx, dy, dist, dtheta, cos_t, sin_t, cos_dt, sin_dt]
        self.observation_space = spaces.Dict({
            "image_obs": spaces.Box(
                low=0.0, high=1.0,
                shape=(6, self.H, self.W),
                dtype=np.float32
            ),
            "vector_obs": spaces.Box(
                low=np.array([-self.W, -self.H, 0.0, -180.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([ self.W,  self.H, np.hypot(self.W, self.H),  180.0,  1.0,  1.0,  1.0,  1.0], dtype=np.float32),
                shape=(8,),
                dtype=np.float32
            )
        })
        # 单环境时的动作空间 shape = (3,)
        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.5, -10.0], dtype=np.float32),
            high=np.array([ 0.5,  0.5,  10.0], dtype=np.float32),
            shape=(3,), dtype=np.float32
        )
        # ------------------------
        # 下面保持原有“批量内部逻辑”
        # ------------------------

        # 1) 生成 K 张随机地图，并对应 K 个碰撞检测器
        self.grids = np.stack(
            [generate_random_map(grid_size, obstacle_prob=map_obstacle_prob).astype(np.float32)
             for _ in range(num_envs)],
            axis=0
        )  # shape = (K, H, W)
        self.world_ccs = [WorldCollisionChecker(self.grids[i]) for i in range(num_envs)]

        # 2) footprint（所有子环境共用一份）
        self.footprint = createFootprint(
            footprint_spec['type'],
            {
                'width': footprint_spec['width'],
                'height': footprint_spec['height'],
                'resolution': footprint_spec['resolution']
            }
        )

        # 3) K 个 env 的状态、目标、初始距离、步数
        self.current_state = np.zeros((num_envs, 3), dtype=np.float32)  # [[x,y,θ], ...]
        self.goal_state    = np.zeros((num_envs, 3), dtype=np.float32)
        self.initial_dist  = np.zeros((num_envs,), dtype=np.float32)
        self.step_count    = np.zeros((num_envs,), dtype=np.int32)

        # 4) K 个 env 的 distance_map (H × W)
        self.distance_maps = np.full((num_envs, self.H, self.W), np.inf, dtype=np.float32)

    def reset(self, *, seed: None = None, options=None):
        if seed is not None:
            np.random.seed(seed)

        for i in range(self.num_envs):
            # 采样合法起点
            while True:
                x0 = np.random.uniform(0, self.W)
                y0 = np.random.uniform(0, self.H)
                θ0 = np.random.uniform(0, 360)
                cand = np.array([x0, y0, θ0], dtype=np.float32)
                if self.world_ccs[i].isValid(self.footprint, cand.reshape(1, 3))[0]:
                    break
            self.current_state[i] = cand

            # 采样合法目标（且距离 >0.1）
            while True:
                xg = np.random.uniform(0, self.W)
                yg = np.random.uniform(0, self.H)
                θg = np.random.uniform(0, 360)
                cand_goal = np.array([xg, yg, θg], dtype=np.float32)
                if self.world_ccs[i].isValid(self.footprint, cand_goal.reshape(1, 3))[0]:
                    if np.linalg.norm(cand_goal[:2] - cand[:2]) > 0.1:
                        break
            self.goal_state[i] = cand_goal

            # 重置步数 & 记录初始距离
            self.step_count[i]    = 0
            self.initial_dist[i]  = np.linalg.norm(self.current_state[i, :2] - self.goal_state[i, :2])

            # BFS 生成 distance_map[i]
            dm = np.full((self.H, self.W), np.inf, dtype=np.float32)
            gi = int(np.clip(np.floor(self.goal_state[i, 1]), 0, self.H - 1))
            gj = int(np.clip(np.floor(self.goal_state[i, 0]), 0, self.W - 1))
            if self.grids[i, gi, gj] == 0.0:
                from collections import deque
                queue = deque()
                dm[gi, gj] = 0.0
                queue.append((gi, gj))
                while queue:
                    ci, cj = queue.popleft()
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < self.H and 0 <= nj < self.W:
                            if self.grids[i, ni, nj] == 0.0 and dm[ni, nj] == np.inf:
                                dm[ni, nj] = dm[ci, cj] + 1.0
                                queue.append((ni, nj))
            self.distance_maps[i] = dm

        obs = self._get_obs()
        infos = [{} for _ in range(self.num_envs)]
        return obs, infos

    def _get_obs(self):
        K, H, W = self.num_envs, self.H, self.W

        image_obs  = np.zeros((K, 6, H, W), dtype=np.float32)
        vector_obs = np.zeros((K, 8), dtype=np.float32)

        def make_gaussian_map(center, sigma=1.0):
            cx, cy = center
            xs = np.arange(0.5, W + 0.5, 1.0, dtype=np.float32)
            ys = np.arange(0.5, H + 0.5, 1.0, dtype=np.float32)
            xv, yv = np.meshgrid(xs, ys)
            g = np.exp(-((xv - cx) ** 2 + (yv - cy) ** 2) / (2 * sigma ** 2))
            g = g / (g.max() + 1e-8)
            return g.astype(np.float32)

        for i in range(K):
            grid = self.grids[i]
            curr = self.current_state[i]
            goal = self.goal_state[i]
            dm   = self.distance_maps[i]

            # 通道 0: 障碍图
            image_obs[i, 0] = grid

            # 通道 1: 当前高斯
            image_obs[i, 1] = make_gaussian_map((curr[0], curr[1]), sigma=1.0)

            # 通道 2: 目标高斯
            image_obs[i, 2] = make_gaussian_map((goal[0], goal[1]), sigma=1.0)

            # 通道 3: 全局距离场 (归一化)
            dm_copy = dm.copy()
            mask_inf = ~np.isfinite(dm_copy)
            dm_copy[mask_inf] = float(H + W)
            image_obs[i, 3] = (dm_copy / float(H + W)).astype(np.float32)

            # 通道 4: 局部 11×11 障碍
            local_full = np.zeros((H, W), dtype=np.float32)
            ci = int(np.clip(np.floor(curr[1]), 0, H - 1))
            cj = int(np.clip(np.floor(curr[0]), 0, W - 1))
            half = 5
            i0, i1 = max(ci - half, 0), min(ci + half + 1, H)
            j0, j1 = max(cj - half, 0), min(cj + half + 1, W)
            local_patch = grid[i0:i1, j0:j1]
            local_full[i0:i1, j0:j1] = local_patch
            image_obs[i, 4] = local_full

            # 通道 5: 当前余弦
            cos_val = np.cos(np.deg2rad(curr[2])).astype(np.float32)
            image_obs[i, 5] = np.full((H, W), cos_val, dtype=np.float32)

            # 8 维向量
            dx = goal[0] - curr[0]
            dy = goal[1] - curr[1]
            euclid_dist = np.linalg.norm([dx, dy], ord=2).astype(np.float32)
            dtheta = float(((curr[2] - goal[2] + 180.0) % 360.0) - 180.0)
            cos_t   = np.cos(np.deg2rad(curr[2])).astype(np.float32)
            sin_t   = np.sin(np.deg2rad(curr[2])).astype(np.float32)
            cos_dt  = np.cos(np.deg2rad(dtheta)).astype(np.float32)
            sin_dt  = np.sin(np.deg2rad(dtheta)).astype(np.float32)

            vector_obs[i] = np.array([
                dx, dy, euclid_dist, dtheta,
                cos_t, sin_t, cos_dt, sin_dt
            ], dtype=np.float32)

        return {"image_obs": image_obs, "vector_obs": vector_obs}

    def step(self, actions: np.ndarray):
        """
        接收批量动作 actions，shape = (K, 3)，
        返回 (obs_dict, rewards, terminated, truncated, infos)。
        """
        K, H, W = self.num_envs, self.H, self.W

        rewards    = np.zeros((K,), dtype=np.float32)
        terminated = np.zeros((K,), dtype=bool)
        truncated  = np.zeros((K,), dtype=bool)
        infos      = [{} for _ in range(K)]

        prev_states = self.current_state.copy()
        prev_dists  = np.linalg.norm(prev_states[:, :2] - self.goal_state[:, :2], axis=1)
        prev_idx_i  = np.clip(np.floor(prev_states[:, 1]).astype(int), 0, H - 1)
        prev_idx_j  = np.clip(np.floor(prev_states[:, 0]).astype(int), 0, W - 1)
        prev_pots   = self.distance_maps[np.arange(K), prev_idx_i, prev_idx_j]
        prev_pots[~np.isfinite(prev_pots)] = float(H + W)

        for i in range(K):
            dx, dy, dtheta = actions[i].astype(np.float32)
            proposed = self.current_state[i] + np.array([dx, dy, dtheta], dtype=np.float32)

            valid = self.world_ccs[i].isValidEdge(
                self.footprint,
                self.current_state[i].reshape(1, 3),
                proposed.reshape(1, 3)
            )[0]

            if not valid:
                # 碰撞惩罚 + 时间惩罚
                rewards[i] = -10.0 - 0.01
                self.step_count[i] += 1
                if self.step_count[i] >= self.max_steps:
                    truncated[i] = True
                infos[i] = {"info": "collision"}
                continue

            # 接受新状态
            self.current_state[i] = proposed.copy()
            self.step_count[i] += 1

            new_dist = np.linalg.norm(self.current_state[i, :2] - self.goal_state[i, :2])
            ni = int(np.clip(np.floor(self.current_state[i, 1]), 0, H - 1))
            nj = int(np.clip(np.floor(self.current_state[i, 0]), 0, W - 1))
            new_pot = self.distance_maps[i, ni, nj]
            if not np.isfinite(new_pot):
                new_pot = float(H + W)

            # 检查是否到达目标
            ang_diff = float(((self.current_state[i, 2] - self.goal_state[i, 2] + 180.0) % 360.0) - 180.0)
            if (new_dist < 0.2) and (abs(ang_diff) < 5.0):
                rewards[i] = 150.0
                terminated[i] = True
                infos[i] = {"info": "reached_goal"}
                continue

            # 中间奖励：距离 + 势场 + 时间
            if self.initial_dist[i] > 1e-6:
                r_dist = ((prev_dists[i] - new_dist) / self.initial_dist[i]) * 5.0
            else:
                r_dist = 0.0
            r_shape = (prev_pots[i] - new_pot) / float(H + W)
            r_time  = -0.01
            r_total = r_dist + r_shape + r_time
            rewards[i] = float(np.clip(r_total, -10.0, 10.0))

            if self.step_count[i] >= self.max_steps:
                truncated[i] = True

        obs = self._get_obs()
        return obs, rewards, terminated, truncated, infos


# ================================================
# 2) MyBatchedVecEnv：封装上面 BatchedPathFindingEnv，补齐 VecEnv 接口
# ================================================
class MyBatchedVecEnv(VecEnv):
    """
    继承自 Stable-Baselines3 的 VecEnv 接口，内部只含一个 BatchedPathFindingEnv 实例。
    """

    def __init__(self, num_envs, grid_size, map_obstacle_prob, footprint_spec, max_steps):
        # 定义“单个子环境”的 observation_space / action_space
        single_obs_space = spaces.Dict({
            "image_obs": spaces.Box(
                low=0.0, high=1.0,
                shape=(6, grid_size[0], grid_size[1]),
                dtype=np.float32
            ),
            "vector_obs": spaces.Box(
                low=np.array([-grid_size[1], -grid_size[0], 0.0, -180.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([ grid_size[1],  grid_size[0], np.hypot(grid_size[1], grid_size[0]), 180.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                shape=(8,),
                dtype=np.float32
            )
        })
        single_act_space = spaces.Box(
            low=np.array([-0.5, -0.5, -10.0], dtype=np.float32),
            high=np.array([ 0.5,  0.5,  10.0], dtype=np.float32),
            shape=(3,), dtype=np.float32
        )

        super().__init__(
            num_envs=num_envs,
            observation_space=single_obs_space,
            action_space=single_act_space
        )

        # 真正的内部环境：批量版 PathFindingEnv
        self.env = BatchedPathFindingEnv(num_envs, grid_size, map_obstacle_prob, footprint_spec, max_steps)
        self.actions = None

    def reset(self):
        obs, _ = self.env.reset()
        return obs  # 以字典形式返回 batch 观测

    def step_async(self, actions):
        # actions.shape = (K, 3)
        self.actions = actions

    def step_wait(self):
        obs, rewards, terminated, truncated, infos = self.env.step(self.actions)
        dones = np.logical_or(terminated, truncated)
        return obs, rewards, dones, infos

    def close(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)

    # 以下四个方法是 VecEnv 的抽象方法，必须实现

    def env_is_wrapped(self, wrapper_class):
        return False

    def env_method(self, method_name, *method_args, **method_kwargs):
        fn = getattr(self.env, method_name)
        return [fn(*method_args, **method_kwargs)]

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)


# ================================================
# 3) HybridExtractor：同之前示例
# ================================================
class HybridExtractor(BaseFeaturesExtractor):
    """
    将 Dict 观测分为两条分支：
      - CNN 处理 image_obs (shape=(K,6,H,W))
      - MLP 处理 vector_obs (shape=(K,8))
    最终输出 (K, feature_dim)。
    """
    def __init__(self, observation_space: gym.spaces.Dict, feature_dim: int = 512):
        super().__init__(observation_space, feature_dim)

        n_channels, H, W = observation_space["image_obs"].shape

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, H, W)
            conv_flat = self.cnn(dummy)
            conv_flat_dim = conv_flat.shape[1]

        feat_cnn_dim = feature_dim // 2
        self.fc_cnn = nn.Sequential(
            nn.Linear(conv_flat_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, feat_cnn_dim),
            nn.ReLU()
        )

        feat_vec_dim = feature_dim - feat_cnn_dim
        self.mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, feat_vec_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        x_img = observations["image_obs"]
        cnn_feat = self.cnn(x_img)
        cnn_feat = self.fc_cnn(cnn_feat)

        x_vec = observations["vector_obs"]
        vec_feat = self.mlp(x_vec)

        return torch.cat([cnn_feat, vec_feat], dim=1)


# ================================================
# 4) 主训练脚本：直接跑 SAC
# ================================================
if __name__ == "__main__":
    K = 8
    grid_size = (20, 20)
    map_obstacle_prob = 0.2
    footprint_spec = {'type': 'rectangle', 'width': 0.5, 'height': 1.5, 'resolution': 0.05}
    max_steps = 200

    # (1) 创建并行 VecEnv
    vec_env = MyBatchedVecEnv(
        num_envs=K,
        grid_size=grid_size,
        map_obstacle_prob=map_obstacle_prob,
        footprint_spec=footprint_spec,
        max_steps=max_steps
    )

    # (2) 创建评估环境：num_envs=1 的 BatchedPathFindingEnv -> DummyVecEnv + Normalize
    eval_base = BatchedPathFindingEnv(
        num_envs=1,
        grid_size=grid_size,
        map_obstacle_prob=map_obstacle_prob,
        footprint_spec=footprint_spec,
        max_steps=max_steps
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
    eval_env = DummyVecEnv([lambda: eval_base])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # (3) 提前停止回调
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=20,
        verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=5_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        verbose=1
    )

    # (4) policy_kwargs
    policy_kwargs = dict(
        features_extractor_class=HybridExtractor,
        features_extractor_kwargs=dict(feature_dim=512),
        net_arch=[512, 256],
        activation_fn=nn.ReLU
    )

    # (5) 创建 SAC 模型
    model = SAC(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,

        buffer_size=int(1e6),
        batch_size=512,

        tau=0.005,
        gamma=0.995,
        ent_coef="auto",

        train_freq=(1, "step"),
        gradient_steps=32,

        learning_rate=3e-4,

        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard_logs/",
        device="cuda"
    )

    # (6) 训练
    model.learn(
        total_timesteps=int(1e6),
        callback=eval_callback
    )

    # (7) 保存
    model.save("sac_batched_pathfinding")
    print("训练完成，模型保存在 sac_batched_pathfinding.zip")
