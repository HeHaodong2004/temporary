#!/usr/bin/env python3
# coding: utf-8
"""
plot_dist_map.py

重置 PathFindingEnvWithMap 后，绘制并保存
  1) dist_map (粗网格)
  2) hires_dist (放大后高分辨率距离图)

用法：
    python plot_dist_map.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from domains_cc.path_env import PathFindingEnvWithMap

def main():
    # 1) 创建环境并 reset
    env = PathFindingEnvWithMap(grid_size=(20,20), obstacle_prob=0.2,
                                prev_action_len=5, patch_size=9, scale=4)
    # reset 会内部 compute dist_map & hires_dist
    _obs, _info = env.reset()

    # 2) 取出两张图
    dist = env.dist_map            # 形状 (H, W)
    hires = env.hires_dist         # 形状 (H*scale, W*scale)

    # 3) 保存目录
    os.makedirs("plots", exist_ok=True)

    # 4) 绘制并保存粗网格距离图
    plt.figure(figsize=(6,5))
    plt.imshow(dist, cmap="viridis", origin="lower")
    plt.colorbar(label="Grid distance")
    plt.title("Coarse Grid Distance Map")
    plt.axis("off")
    plt.savefig("plots/dist_map.png", bbox_inches="tight", dpi=150)
    plt.close()

    # 5) 绘制并保存高分辨率距离图
    plt.figure(figsize=(6,5))
    plt.imshow(hires, cmap="viridis", origin="lower")
    plt.colorbar(label="High-res distance")
    plt.title("High-Resolution Distance Map")
    plt.axis("off")
    plt.savefig("plots/hires_dist_map.png", bbox_inches="tight", dpi=150)
    plt.close()

    print("✅ Saved dist_map.png and hires_dist_map.png under ./plots/")

if __name__ == "__main__":
    main()
