import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
from collections import deque
import heapq
import cv2   # for egocentric occupancy patch rotation

from domains_cc.footprint import createFootprint
from domains_cc.map_generator import generate_random_map
from domains_cc.worldCC import WorldCollisionChecker

class PathFindingEnvWithMap(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_size=(20,20),
        obstacle_prob=0.1,
        footprint_spec=None,
        max_steps=300,
        prev_action_len: int = 4
    ):
        super().__init__()
        # map + collision checker
        self.grid = generate_random_map(grid_size, obstacle_prob).astype(np.float32)
        self.world_cc = WorldCollisionChecker(self.grid)
        self.H, self.W = self.grid.shape

        # robot footprint
        if footprint_spec is None:
            footprint_spec = dict(type="rectangle", width=0.5,
                                  height=1.5, resolution=0.1)
        self.footprint = createFootprint(
            footprint_spec["type"],
            dict(width=footprint_spec["width"],
                 height=footprint_spec["height"],
                 resolution=footprint_spec["resolution"])
        )

        # discrete actions: dx, dy, dtheta
        self.discrete_actions = [
            np.array([0.5, 0.0, 0.0], np.float32),
            np.array([-0.5, 0.0, 0.0], np.float32),
            np.array([0.0, 0.5, 0.0], np.float32),
            np.array([0.0, -0.5, 0.0], np.float32),
            np.array([0.0, 0.0, 3.5], np.float32),
            np.array([0.0, 0.0, -3.5], np.float32),
        ]
        self.action_space = spaces.Discrete(len(self.discrete_actions))

        # previous actions buffer length
        self.prev_action_len = prev_action_len
        self.prev_actions = deque(maxlen=self.prev_action_len)

        # egocentric occupancy patch size (odd)
        self.patch_size = 7

        # observation: goal vec + flow vec + occ_patch + prev_actions + action_mask
        self.observation_space = spaces.Dict({
            "vec_obs":    spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
            "next_obs":   spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
            "occ_patch":  spaces.Box(0.0, 1.0, shape=(1, self.patch_size, self.patch_size), dtype=np.float32),
            # prev_actions encoded as past dx,dy,dtheta vectors, flattened
            "prev_actions": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.prev_action_len * 3,),
                dtype=np.float32
            ),
            "action_mask": spaces.MultiBinary(len(self.discrete_actions)),
        })

        # episode limits
        self.max_steps = max_steps
        self.current_state = np.zeros(3, np.float32)
        self.goal_state = np.zeros(3, np.float32)
        self.step_count = 0

        # placeholders for maps
        self.dist_map = np.full((self.H, self.W), np.inf, np.float32)
        self.flow_field = np.zeros((self.H, self.W, 2), np.float32)

    def reset(self, *, seed: Optional[int]=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        # sample start & goal
        def sample_pose():
            return np.array([
                np.random.uniform(0, self.W),
                np.random.uniform(0, self.H),
                np.random.uniform(0, 360)
            ], np.float32)
        # ensure valid start
        while True:
            s = sample_pose()
            if self.world_cc.isValid(self.footprint, s.reshape(1,3))[0]:
                break
        # ensure valid goal
        while True:
            gi = np.random.randint(0, self.H)
            gj = np.random.randint(0, self.W)
            theta = np.random.uniform(0, 360)
            g = np.array([gj+0.5, gi+0.5, theta], np.float32)
            if self.world_cc.isValid(self.footprint, g.reshape(1,3))[0] and \ 
               np.linalg.norm(g[:2]-s[:2])>0.1:
                break

        self.current_state = s.copy()
        self.goal_state = g.copy()
        self.step_count = 0

        # compute maps
        gi_idx, gj_idx = int(self.goal_state[1]-0.5), int(self.goal_state[0]-0.5)
        self.dist_map = self._bfs_distance(self.grid, gi_idx, gj_idx)
        self.flow_field = self._build_flow_field(self.dist_map)

        # init prev_actions
        self.prev_actions.clear()
        for _ in range(self.prev_action_len):
            self.prev_actions.append(-1)

        obs, _ = self._get_obs()
        return obs, {}

    @staticmethod
    def _bfs_distance(grid: np.ndarray, gi: int, gj: int) -> np.ndarray:
        H, W = grid.shape
        INF = np.inf
        dist = np.full((H, W), INF, np.float32)
        free = (grid < 0.5)
        dq = deque()
        if free[gi, gj]:
            dist[gi, gj] = 0
            dq.append((gi, gj))
        neighs = [(-1,0),(1,0),(0,-1),(0,1)]
        while dq:
            i, j = dq.popleft()
            for di, dj in neighs:
                ni, nj = i+di, j+dj
                if 0<=ni<H and 0<=nj<W and free[ni, nj] and dist[ni, nj]==INF:
                    dist[ni, nj] = dist[i, j]+1
                    dq.append((ni, nj))
        return dist

    @staticmethod
    def _build_flow_field(dist: np.ndarray) -> np.ndarray:
        H, W = dist.shape
        flow = np.zeros((H, W, 2), np.float32)
        neighs = [(-1,0),(1,0),(0,-1),(0,1)]
        for i in range(H):
            for j in range(W):
                d0 = dist[i, j]
                if not np.isfinite(d0):
                    continue
                best, vec = d0, None
                for di, dj in neighs:
                    ni, nj = i+di, j+dj
                    if 0<=ni<H and 0<=nj<W and dist[ni, nj]<best:
                        best = dist[ni, nj]
                        vec = (dj, di)
                if vec is not None:
                    dx, dy = vec
                    norm = np.hypot(dx, dy) + 1e-8
                    flow[i, j, 0] = dx/norm
                    flow[i, j, 1] = dy/norm
        return flow

    def _grid_pos(self, state):
        return int(state[1]), int(state[0])

    def _occ_patch(self):
        k = self.patch_size
        pad = k//2 + 1
        cx, cy = self.current_state[:2]
        gi0, gj0 = int(cy)-pad, int(cx)-pad
        raw = np.ones((k+2*pad, k+2*pad), np.float32)
        for r in range(k+2*pad):
            for c in range(k+2*pad):
                gi, gj = gi0+r, gj0+c
                if 0 <= gi < self.H and 0 <= gj < self.W:
                    raw[r, c] = self.grid[gi, gj]
        angle = -self.current_state[2]
        M = cv2.getRotationMatrix2D(((k+2*pad)/2, (k+2*pad)/2), angle, 1.0)
        rot = cv2.warpAffine(raw, M, (k+2*pad, k+2*pad), flags=cv2.INTER_NEAREST, borderValue=1.0)
        center = rot[pad:pad+k, pad:pad+k]
        return (1.0 - center)[None, ...]

    def _get_obs(self):
        # goal vec
        dx = (self.goal_state[0]-self.current_state[0]) / self.W
        dy = (self.goal_state[1]-self.current_state[1]) / self.H
        err = (self.goal_state[2]-self.current_state[2] + 180) % 360 - 180
        sin_g = np.sin(np.deg2rad(err))
        cos_g = np.cos(np.deg2rad(err))
        vec_obs = np.array([dx, dy, sin_g, cos_g], np.float32)
        # flow vec
        gi, gj = self._grid_pos(self.current_state)
        fx, fy = self.flow_field[gi, gj]
        angle = np.arctan2(fy, fx)
        sin_n = np.sin(angle)
        cos_n = np.cos(angle)
        next_obs = np.array([fx, fy, sin_n, cos_n], np.float32)
        # occupancy patch
        patch = self._occ_patch()
        # action mask
        mask = np.array([
            self.world_cc.isValidEdge(
                self.footprint,
                self.current_state.reshape(1,3),
                (self.current_state+mv).reshape(1,3)
            )[0] for mv in self.discrete_actions
        ], dtype=bool)
        # prev_actions as dx,dy,dtheta
        prev_arr = np.zeros((self.prev_action_len, 3), np.float32)
        for i, a in enumerate(self.prev_actions):
            if a >= 0:
                prev_arr[i] = self.discrete_actions[a]
        prev_flat = prev_arr.ravel()
        return {
            "vec_obs": vec_obs,
            "next_obs": next_obs,
            "occ_patch": patch,
            "prev_actions": prev_flat,
            "action_mask": mask,
        }, None

    def step(self, action: int):
        prev = self.current_state.copy()
        self.current_state = prev + self.discrete_actions[action]
        self.step_count += 1
        # reward: A* path length change
        pd = len(self._astar_path(prev, self.goal_state))
        nd = len(self._astar_path(self.current_state, self.goal_state))
        reward = -0.1 + float(pd - nd)
        # done
        d = np.linalg.norm(self.current_state[:2]-self.goal_state[:2])
        err = (self.current_state[2]-self.goal_state[2] + 180) % 360 - 180
        done = False
        info = {}
        if d < 0.5 and abs(err) < 10:
            done = True
            reward += 100.0
            info["info"] = "reached_goal"
        elif self.step_count >= self.max_steps:
            done = True
        self.prev_actions.append(action)
        obs, _ = self._get_obs()
        return obs, reward, done, False, info

    def action_masks(self):
        obs, _ = self._get_obs()
        return obs["action_mask"]

    def render(self, save_path="flow_field.png"):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6,6), dpi=100)
        ax.imshow(self.grid, cmap='Greys', origin='lower', extent=(-0.5, self.W-0.5, -0.5, self.H-0.5))
        ys, xs = np.arange(0, self.H), np.arange(0, self.W)
        X, Y = np.meshgrid(xs, ys)
        U = self.flow_field[Y, X, 0]
        V = self.flow_field[Y, X, 1]
        ax.quiver(X+0.5, Y+0.5, U, V, angles='xy', scale_units='xy', scale=1, width=0.003, alpha=0.7)
        self.world_cc.addXYThetaToPlot(ax, self.footprint, self.current_state, True)
        self.world_cc.addXYThetaToPlot(ax, self.footprint, self.goal_state, True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, self.W)
        ax.set_ylim(0, self.H)

    def _astar_path(self, start, goal):
        start_c, goal_c = self._grid_pos(start), self._grid_pos(goal)
        open_set = [(abs(start_c[0]-goal_c[0]) + abs(start_c[1]-goal_c[1]), 0, start_c)]
        came, g_score, visited = {}, {start_c:0}, set()
        while open_set:
            _, cost, cur = heapq.heappop(open_set)
            if cur in visited:
                continue
            visited.add(cur)
            if cur == goal_c:
                break
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nb = (cur[0]+dx, cur[1]+dy)
                if not (0<=nb[1]<self.H and 0<=nb[0]<self.W):
                    continue
                if self.grid[nb[1], nb[0]] > 0.5:
                    continue
                nc = cost + 1
                if nc < g_score.get(nb, 1e9):
                    came[nb] = cur
                    g_score[nb] = nc
                    heapq.heappush(open_set, (nc + abs(nb[0]-goal_c[0]) + abs(nb[1]-goal_c[1]), nc, nb))
        # reconstruct
        path = []
        node = goal_c
        while node != start_c and node in came:
            path.append(node)
            node = came[node]
        if node == start_c:
            path.append(start_c)
        return list(reversed(path))
