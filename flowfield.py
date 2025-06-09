#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# --------- 假设 dist_map 已经存在 ---------
# 这里我们直接从上一步保存的 distance_map.png 对应的 dist_map.npy 读入
# 如果你没有保存成 .npy，可以直接替换下面这行：
# dist_map = np.load("dist_map.npy")
#
# 下面为了示例，重用上次随机生成的 env 逻辑来算一个 dist_map
import gymnasium as gym
from collections import deque
from domains_cc.map_generator import generate_random_map

# 1) 随机地图
grid = generate_random_map((20,20), 0.1).astype(np.float32)
H, W = grid.shape

# 2) 计算 dist_map
def compute_distance_map(grid, goal_i, goal_j):
    H,W = grid.shape
    INF = np.inf
    dist = np.full((H,W), INF, np.float32)
    free = (grid < 0.5)
    queue = deque()
    if free[goal_i,goal_j]:
        dist[goal_i,goal_j] = 0
        queue.append((goal_i,goal_j))
    neighs = [(-1,0),(1,0),(0,-1),(0,1)]
    while queue:
        i,j = queue.popleft()
        d0 = dist[i,j]
        for di,dj in neighs:
            ni, nj = i+di, j+dj
            if 0<=ni<H and 0<=nj<W and free[ni,nj] and dist[ni,nj]==INF:
                dist[ni,nj] = d0+1
                queue.append((ni,nj))
    return dist

# 随机选一个目标格
goal_i, goal_j = np.random.randint(0,H), np.random.randint(0,W)
dist_map = compute_distance_map(grid, goal_i, goal_j)

# --------- 构造矢量场 ---------
flow = np.zeros((H, W, 2), np.float32)
neighs = [(-1,0),(1,0),(0,-1),(0,1)]
for i in range(H):
    for j in range(W):
        if dist_map[i,j] == np.inf: 
            continue
        best = dist_map[i,j]
        vec = None
        for di,dj in neighs:
            ni, nj = i+di, j+dj
            if 0<=ni<H and 0<=nj<W and dist_map[ni,nj] < best:
                best = dist_map[ni,nj]
                vec = (dj, di)   # x=j方向是列，y=i方向是行
        if vec is not None:
            dx, dy = vec
            norm = np.hypot(dx, dy)
            flow[i,j,0] = dx / (norm + 1e-8)
            flow[i,j,1] = dy / (norm + 1e-8)

# --------- 绘图并保存 ---------
fig, ax = plt.subplots(figsize=(6,6), dpi=100)
# 背景格子
ax.imshow(grid, cmap='Greys', origin='lower', extent=(0,W,0,H))
# 箭头
ys, xs = np.mgrid[0:H, 0:W]
U = flow[:,:,0]
V = flow[:,:,1]
ax.quiver(xs+0.5, ys+0.5, U, V, 
          pivot='mid', scale=20, width=0.003, alpha=0.8)

# 标出目标格中心
ax.plot(goal_j+0.5, goal_i+0.5, 'xr', markersize=12, label='Goal')
ax.legend(loc='upper right')

ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_aspect('equal')

plt.savefig("flow_field.png", bbox_inches='tight')
plt.close(fig)
print("✅ Saved flow_field.png")
