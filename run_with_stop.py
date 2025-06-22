#!/usr/bin/env python3
"""
Maskable PPO for PathFindingEnvWithMap with SB3 v2.6.0 & sb3-contrib v2.6.0

- Discrete actions
- Dict observation {"vec_obs", "next_obs", "occ_patch", "prev_actions", "action_mask"}
- Provides action_masks() interface
- VecNormalize only normalizes "vec_obs" and "next_obs"
- SubprocVecEnv for parallel sampling
- Implements Action Chunking + Temporal Ensemble via external Wrapper
- Added continue_learning option to resume training from checkpoint
"""

import os
import copy
import numpy as np
import gymnasium as gym
from collections import deque
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CallbackList
from sb3_contrib import MaskablePPO
from domains_cc.path_env import PathFindingEnvWithMap

# === CONFIGURATION ===
continue_learning = True  # set to False to start fresh training
total_timesteps = 10_000_000
log_dir = "./tensorboard_logs/"
models_dir = "./models/"
# ======================

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
                obs_pred, *_ = pred_env.step(int(a))
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

if __name__ == "__main__":
    # --- setup directories ---
    os.makedirs(os.path.join(models_dir, "best"), exist_ok=True)
    os.makedirs(os.path.join(models_dir, "last"), exist_ok=True)

    # --- environment setup ---
    train_env = SubprocVecEnv([make_env] * 96)
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        norm_obs_keys=["vec_obs", "next_obs"],
    )

    # paths
    best_path = os.path.join(models_dir, "best", "best_model.zip")
    last_path = os.path.join(models_dir, "last", "last_model.zip")
    vec_path = os.path.join(models_dir, "VecNormalize.pkl")

    initial_lr = 3e-4

    # --- model loading/creation ---
    if continue_learning and os.path.exists(last_path) and os.path.exists(vec_path):
        print(f"Resuming training from {last_path}")
        model = MaskablePPO.load(last_path, env=train_env, device="auto", tensorboard_log=log_dir)
        train_env.load(vec_path)
    else:
        print("Starting fresh training")
        model = MaskablePPO(
            policy="MultiInputPolicy",
            env=train_env,
            verbose=1,
            n_steps=4096,
            batch_size=512,
            learning_rate=lambda p: initial_lr * p,
            tensorboard_log=log_dir,
            device="auto"
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
    early_stop = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=15, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(models_dir, "best"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=40000,
        n_eval_episodes=25,
        deterministic=True,
        verbose=1,
        callback_on_new_best=early_stop,
    )

    # --- training ---
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([eval_cb]),
        reset_num_timesteps=not continue_learning
    )

    # --- save latest model and normalizer ---
    model.save(last_path)
    train_env.save(vec_path)
    print("Training complete.")

    # --- evaluation example ---
    eval_env0 = make_env()
    wrapped = ChunkTemporalWrapper(eval_env0, model, chunk_size=5, ensemble_depth=4, m=0.5)
    obs, _ = wrapped.reset()
    done = False
    total_r = 0.0
    while not done:
        obs, r, done, term, info = wrapped.step()
        total_r += r
    print(f"Eval episodic reward: {total_r}")
