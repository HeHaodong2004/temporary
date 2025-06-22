#!/usr/bin/env python3
"""
Maskable PPO for PathFindingEnvWithMap with SB3 v2.6.0 & sb3-contrib v2.6.0

- Discrete actions
- Dict observation {"vec_obs", "next_obs", "occ_patch", "prev_actions", "action_mask"}
- Provides action_masks() interface
- VecNormalize only normalizes "vec_obs" and "next_obs"
- SubprocVecEnv for parallel sampling
- Implements Action Chunking + Temporal Ensemble via external Wrapper
"""

import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
from collections import deque
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import heapq

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
from sb3_contrib import MaskablePPO

from domains_cc.path_env import PathFindingEnvWithMap


def make_env():
    return PathFindingEnvWithMap(prev_action_len=4)

class ChunkTemporalWrapper(gym.Wrapper):
    """Action Chunking + Multi-Step Prediction + Temporal Ensemble Wrapper"""
    def __init__(self, env, model, chunk_size=5, ensemble_depth=4, m=0.5):
        super().__init__(env)
        self.model = model
        self.chunk_size = chunk_size
        self.ensemble_depth = ensemble_depth
        self.m = m
        self.chunks = deque(maxlen=ensemble_depth)
        self.ptr = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.chunks.clear()
        for _ in range(self.ensemble_depth):
            self.chunks.append([0] * self.chunk_size)
        self.ptr = 0
        return obs, info

    def step(self, action=None):
        if self.ptr == 0:
            pred_env = copy.deepcopy(self.env)
            chunk_actions = []
            obs_pred, _ = pred_env._get_obs()
            for _ in range(self.chunk_size):
                a, _ = self.model.predict(obs_pred, deterministic=True)
                chunk_actions.append(int(a))
                obs_pred, _, _, _, _ = pred_env.step(int(a))
            self.chunks.appendleft(chunk_actions)

        votes = {}
        for idx, ch in enumerate(self.chunks):
            a = ch[self.ptr]
            w = np.exp(-self.m * idx)
            votes[a] = votes.get(a, 0.0) + w
        chosen = max(votes.items(), key=lambda x: x[1])[0]

        obs, rew, done, term, info = self.env.step(chosen)
        self.ptr = (self.ptr + 1) % self.chunk_size
        return obs, rew, done, term, info


class PatchAndVecExtractor(BaseFeaturesExtractor):
    """Custom extractor for occ_patch + vector observations"""
    def __init__(self, observation_space, patch_size: int = 7, vec_size: int = 4 + 4 + 4 * 6):
        super().__init__(observation_space, features_dim=1)
        # CNN for 7x7 patch -> (7-2*2)=3x3
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_dim = 32 * (patch_size - 2*2) * (patch_size - 2*2)
        self._features_dim = conv_out_dim + vec_size

    def forward(self, observations: dict) -> th.Tensor:
        # occ_patch branch
        patch = observations['occ_patch'].float()       # (B,1,7,7)
        h = self.cnn(patch)
        h = h.view(h.size(0), -1)
        # vector branch
        vec_obs = observations['vec_obs'].float()       # (B,4)
        next_obs = observations['next_obs'].float()     # (B,4)
        prev = observations['prev_actions'].long().squeeze(-1)  # (B,4)
        # mask invalid (-1) indices before one_hot
        mask_valid = (prev >= 0)
        prev_clamped = prev.clamp(min=0)
        prev_onehot = nn.functional.one_hot(prev_clamped, num_classes=6).float()  # (B,4,6)
        prev_onehot = prev_onehot * mask_valid.unsqueeze(-1).float()
        prev_onehot = prev_onehot.view(prev_onehot.size(0), -1)  # (B,24)
        vec = th.cat([vec_obs, next_obs, prev_onehot], dim=1)    # (B,32)
        # concat features
        return th.cat([h, vec], dim=1)


if __name__ == "__main__":
    # --- training ---
    train_env = SubprocVecEnv([make_env] * 32)
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        norm_obs_keys=["vec_obs", "next_obs"],
    )

    policy_kwargs = dict(
        features_extractor_class=PatchAndVecExtractor,
        features_extractor_kwargs=dict(patch_size=7, vec_size=4 + 4 + 4 * 6),
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=nn.ReLU,
    )

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        batch_size=256,
        learning_rate=3e-4,
        tensorboard_log="./tensorboard_logs/",
        device="auto",
    )

    # --- evaluation callback ---
    eval_env = DummyVecEnv([make_env])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        norm_obs_keys=["vec_obs", "next_obs"],
    )
    early_stop = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=5,
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/eval/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
        callback_on_new_best=early_stop,
    )
    model.learn(total_timesteps=10_000_000, callback=CallbackList([eval_cb]))
    model.save("maskppo_pathfinder_final")
    print("Training complete.")

    # --- evaluation with Action Chunking + Temporal Ensemble ---
    eval_env0 = make_env()
    wrapped = ChunkTemporalWrapper(
        eval_env0,
        model,
        chunk_size=5,
        ensemble_depth=4,
        m=0.5,
    )
    obs, _ = wrapped.reset()
    done = False
    total_r = 0.0
    while not done:
        obs, r, done, term, info = wrapped.step()
        total_r += r
    print(f"Eval episodic reward: {total_r}")
