#!/usr/bin/env python3
# coding: utf-8
"""
Quick-n-dirty: drop the last captured frame before saving the GIF.
Test–time exploration with:
  - Translation actions (0–3): de-duplicate after 3 repeats per cell
  - Rotation actions (4–7): de-duplicate after 10 repeats per cell
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from datetime import datetime
import numpy.core.numeric as _np_num
from collections import Counter

sys.modules['numpy._core.numeric'] = _np_num
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from domains_cc.path_env import PathFindingEnvWithMap

BEST_MODEL_PATH = "./models/best/best_model.zip"
GIF_NAME        = f"evaluation_{datetime.now():%Y%m%d_%H%M%S}.gif"

# Action index sets
TRANSLATION_ACTIONS = {0, 1, 2, 3}
ROTATION_ACTIONS    = {4, 5, 6, 7}

# De-dup thresholds
TRANSLATION_THRESHOLD = 3
ROTATION_THRESHOLD    = 10

def make_env():
    return PathFindingEnvWithMap(prev_action_len=4)

def fig2rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)

def main():
    vec_env = DummyVecEnv([make_env])
    base_env = vec_env.envs[0]
    model = MaskablePPO.load(BEST_MODEL_PATH, env=vec_env, device="auto")

    obs = vec_env.reset()
    frames = []

    # initial frame
    fig, _ = base_env.render()
    frames.append(fig2rgb(fig))
    plt.close(fig)

    done = False
    # per-cell counters for translations and rotations
    trans_counters = {}  # { (qj,qi): Counter() }
    rot_counters   = {}  # { (qj,qi): Counter() }

    while not done:
        # 1) model's preferred action
        action_arr, _ = model.predict(
            obs, deterministic=True, action_masks=obs["action_mask"][0]
        )
        action = int(action_arr[0])

        # 2) quantize position at 0.25-unit resolution
        step = 0.25
        x, y = base_env.current_state[:2]
        qi, qj = int(x/step), int(y/step)
        cell = (qj, qi)

        # init counters for this cell
        trans_ctr = trans_counters.setdefault(cell, Counter())
        rot_ctr   = rot_counters.setdefault(cell,   Counter())

        mask = obs["action_mask"][0]

        # 3) de-dup logic
        if action in TRANSLATION_ACTIONS:
            if trans_ctr[action] >= TRANSLATION_THRESHOLD:
                # choose among translations tried < threshold
                cands = [a for a in TRANSLATION_ACTIONS
                         if mask[a] and trans_ctr[a] < TRANSLATION_THRESHOLD]
                if cands:
                    action = int(np.random.choice(cands))

        elif action in ROTATION_ACTIONS:
            if rot_ctr[action] >= ROTATION_THRESHOLD:
                # choose among rotations tried < threshold
                cands = [a for a in ROTATION_ACTIONS
                         if mask[a] and rot_ctr[a] < ROTATION_THRESHOLD]
                if cands:
                    action = int(np.random.choice(cands))

        # 4) record this attempt
        if action in TRANSLATION_ACTIONS:
            trans_ctr[action] += 1
        else:
            rot_ctr[action] += 1

        # 5) step environment (batch of size 1)
        obs, reward, done_vec, _ = vec_env.step([action])
        done = done_vec[0]

        # 6) render and save frame
        fig, _ = base_env.render()
        frames.append(fig2rgb(fig))
        plt.close(fig)

    # drop final "done" frame
    if len(frames) > 1:
        frames = frames[:-1]

    # save GIF
    imageio.mimsave(GIF_NAME, frames, duration=0.12)
    print("GIF saved to", GIF_NAME)

if __name__ == "__main__":
    main()
