#!/usr/bin/env python3
"""
Maskable PPO for PathFindingEnvWithMap with SB3 v2.6.0 & sb3-contrib v2.6.0
- Discrete actions
- Dict observation {"grid_patch", "dist_patch", "vec_obs", "action_mask"}
- Provides action_masks() interface
- VecNormalize only normalizes "vec_obs"
- SubprocVecEnv for parallel sampling
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import heapq
import matplotlib
matplotlib.use('Agg')

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CallbackList
from sb3_contrib import MaskablePPO

# Your project dependencies
from domains_cc.footprint import createFootprint
from domains_cc.map_generator import generate_random_map
from domains_cc.worldCC import WorldCollisionChecker

class PathFindingEnvWithMap(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 grid_size=(20, 20),
                 map_obstacle_prob=0.1,
                 footprint_spec=None,
                 max_steps=200,
                 patch_size=3):
        super().__init__()
        if footprint_spec is None:
            footprint_spec = dict(type="rectangle", width=0.5,
                                  height=1.5, resolution=0.05)
        # Global map and collision checker
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
        self.max_astar = H + W

        # Discrete action set
        self.discrete_actions = [
            np.array([0.5,  0.0,  0.0], np.float32),
            np.array([-0.5, 0.0,  0.0], np.float32),
            np.array([0.0,  0.5,  0.0], np.float32),
            np.array([0.0, -0.5,  0.0], np.float32),
            np.array([0.0,  0.0, 10.0], np.float32),
            np.array([0.0,  0.0,-10.0], np.float32),
        ]
        self.action_space = spaces.Discrete(len(self.discrete_actions))

        # Patch parameters
        self.patch_size = patch_size
        half = patch_size // 2

        # Observation space
        self.observation_space = spaces.Dict({
            "grid_patch": spaces.Box(0, 1, shape=(1, patch_size, patch_size), dtype=np.float32),
            "dist_patch": spaces.Box(0, 1, shape=(1, patch_size, patch_size), dtype=np.float32),
            "vec_obs":    spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(len(self.discrete_actions)),
        })

        self.max_steps = max_steps
        self.current_state = np.zeros(3, np.float32)
        self.goal_state    = np.zeros(3, np.float32)
        self.global_dist   = np.zeros_like(self.grid)
        self.step_count    = 0
        self.half_patch    = half

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            np.random.seed(seed)
        H, W = self.grid.shape

        # Sample valid start and goal
        def sample():
            return np.array([np.random.uniform(0, W),
                             np.random.uniform(0, H),
                             np.random.uniform(0, 360)], np.float32)
        while True:
            s = sample()
            if self.world_cc.isValid(self.footprint, s.reshape(1,3))[0]: break
        while True:
            g = sample()
            if self.world_cc.isValid(self.footprint, g.reshape(1,3))[0] and \
               np.linalg.norm(g[:2]-s[:2])>0.1: break

        self.current_state, self.goal_state = s.copy(), g.copy()
        self.step_count = 0

        # Compute global distance field from goal
        goal_pos = self._grid_pos(self.goal_state)
        self.global_dist = self._compute_global_dist(goal_pos)

        return self._get_obs(), {}

    def _grid_pos(self, state):
        x, y, _ = state
        return (int(x), int(y))

    def _compute_global_dist(self, goal):
        H, W = self.grid.shape
        dist = np.full((H, W), np.inf, dtype=np.float32)
        visited = np.zeros((H, W), bool)
        heap = [(0, goal)]
        dist[goal[1], goal[0]] = 0

        while heap:
            d, (gx, gy) = heapq.heappop(heap)
            if visited[gy, gx]: continue
            visited[gy, gx] = True
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = gx+dx, gy+dy
                if not(0<=nx<W and 0<=ny<H): continue
                if self.grid[ny, nx]>0.5: continue
                nd = d+1
                if nd<dist[ny, nx]:
                    dist[ny, nx]=nd
                    heapq.heappush(heap,(nd,(nx,ny)))
        return np.minimum(dist, self.max_astar)/self.max_astar

    def _astar_distance(self, start, goal):
        start_c, goal_c = self._grid_pos(start), self._grid_pos(goal)
        H,W = self.grid.shape
        def h(p): return abs(p[0]-goal_c[0])+abs(p[1]-goal_c[1])
        open_set = [(h(start_c),0,start_c)]
        g_score={start_c:0}; visited=set()
        while open_set:
            _,g,pos=heapq.heappop(open_set)
            if pos in visited: continue
            visited.add(pos)
            if pos==goal_c: return g
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nb=(pos[0]+dx,pos[1]+dy)
                if not(0<=nb[0]<W and 0<=nb[1]<H): continue
                if self.grid[nb[1], nb[0]]>0.5: continue
                tg=g+1
                if tg<g_score.get(nb,float('inf')):
                    g_score[nb]=tg
                    heapq.heappush(open_set,(tg+h(nb),tg,nb))
        return self.max_astar

    def _get_obs(self):
        x,y,_ = self.current_state
        cx, cy = int(x), int(y)
        h = self.half_patch

        # Local occupancy patch
        occ = self.grid[max(0, cy-h): cy+h+1, max(0, cx-h): cx+h+1]
        pad_y = (max(0, h-cy), max(0, cy+h+1-self.grid.shape[0]))
        pad_x = (max(0, h-cx), max(0, cx+h+1-self.grid.shape[1]))
        occ_patch = np.pad(occ, (pad_y, pad_x), constant_values=1.0)
        occ_patch = (occ_patch>0.5).astype(np.float32)[None,...]

        # Local distance patch
        dp = self.global_dist[max(0, cy-h): cy+h+1, max(0, cx-h): cx+h+1]
        dist_patch = np.pad(dp, (pad_y, pad_x), constant_values=1.0)[None,...]

        # Vector obs
        dx, dy = (self.goal_state[:2]-self.current_state[:2])
        dx /= self.grid.shape[1]; dy /= self.grid.shape[0]
        err = (self.goal_state[2]-self.current_state[2]+180)%360-180
        sin_e, cos_e = np.sin(np.deg2rad(err)), np.cos(np.deg2rad(err))
        vec_obs = np.array([dx,dy,sin_e,cos_e], dtype=np.float32)

        # Action mask
        mask = np.array([
            self.world_cc.isValidEdge(
                self.footprint,
                self.current_state.reshape(1,3),
                (self.current_state+move).reshape(1,3)
            )[0]
            for move in self.discrete_actions
        ], bool)

        return {"grid_patch": occ_patch,
                "dist_patch": dist_patch,
                "vec_obs":    vec_obs,
                "action_mask": mask}

    def step(self, action: int):
        prev = self.current_state.copy()
        self.current_state = prev + self.discrete_actions[action]
        self.step_count += 1

        # Reward: time penalty + ASTAR progress
        tp = -0.1
        pd = self._astar_distance(prev, self.goal_state)
        nd = self._astar_distance(self.current_state, self.goal_state)
        ar = float(pd - nd)

        # Check terminal
        d = np.linalg.norm(self.current_state[:2]-self.goal_state[:2])
        err = (self.current_state[2]-self.goal_state[2]+180)%360-180
        if d<0.2 and abs(err)<5:
            return self._get_obs(), 100.0, True, False, {"info": "reached_goal"}

        obs = self._get_obs()
        done = self.step_count>=self.max_steps
        return obs, tp+ar, False, done, {}

    def action_masks(self):
        return self._get_obs()["action_mask"]

    def render(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6,6), dpi=100)
        ax.imshow(self.grid, cmap='Greys', origin='lower',
                  extent=(0,self.grid.shape[1],0,self.grid.shape[0]), interpolation='nearest')
        self.world_cc.addXYThetaToPlot(ax, self.footprint, self.current_state, computeValidity=True)
        self.world_cc.addXYThetaToPlot(ax, self.footprint, self.goal_state, computeValidity=False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0,self.grid.shape[1]); ax.set_ylim(0,self.grid.shape[0])
        ax.set_aspect('equal')

        canvas = FigureCanvas(fig)
        canvas.draw()
        w,h = fig.canvas.get_width_height()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
        plt.close(fig)
        return img

def make_env(): return PathFindingEnvWithMap()


if __name__ == "__main__":
    # Linear lr schedule
    def linear_schedule(init): return lambda prog: init * prog

    # Training envs
    train_env = SubprocVecEnv([make_env]*32)
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                              clip_obs=10.0, norm_obs_keys=["vec_obs"])

    # Eval env
    eval_env = DummyVecEnv([make_env])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True,
                             clip_obs=10.0, norm_obs_keys=["vec_obs"])

    # Early stop callback
    early_stop = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10, min_evals=5, verbose=1
    )
    eval_callback = EvalCallback(
        eval_env, best_model_save_path="./models/best_maskppo/",
        log_path="./logs/eval/", eval_freq=10000, n_eval_episodes=5,
        deterministic=True, render=False, verbose=1,
        callback_on_new_best=early_stop
    )
    callbacks = CallbackList([eval_callback])

    # Create and train
    model = MaskablePPO(
        policy="MultiInputPolicy", env=train_env, verbose=1,
        batch_size=256, learning_rate=linear_schedule(3e-4),
        tensorboard_log="./tensorboard_logs/", device="auto"
    )
    model.learn(total_timesteps=2_000_000, callback=callbacks)
    model.save("maskableppo_pathfinder_final")
    print("✅ Training complete, model saved to maskableppo_pathfinder_final.zip")
