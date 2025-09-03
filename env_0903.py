import os
import random
import numpy as np
from copy import deepcopy
from skimage import io
from skimage.measure import block_reduce

from sensor import sensor_work
from utils import *
from parameter import *

class Env:
    def __init__(self, episode_index, plot=False, test=False):
        self.episode_index = episode_index
        self.plot = plot
        self.test = test

        # 读取地图（保持你单机的数据集结构）
        self.ground_truth, initial_cell, self.map_path = self.import_ground_truth(episode_index)
        self.cell_size = CELL_SIZE
        self.ground_truth_size = np.shape(self.ground_truth)

        # —— 多机信念图 —— #
        self.global_belief  = np.ones(self.ground_truth_size) * UNKNOWN
        self.agent_beliefs  = [np.ones_like(self.global_belief) * UNKNOWN for _ in range(N_AGENTS)]

        # 坐标原点（与单机一致）
        self.belief_origin_x = -np.round(initial_cell[0] * self.cell_size, 1)
        self.belief_origin_y = -np.round(initial_cell[1] * self.cell_size, 1)

        # 传感器与指标
        self.sensor_range = SENSOR_RANGE
        self.explored_rate = 0.0
        self.done = False
        self.total_travel_dist = 0.0
        # —— 新增：逐机器人累计路程与最大路程 —— #
        self.agent_travel_dists = np.zeros(N_AGENTS, dtype=float)
        self.max_travel_dist = 0.0

        # 初次观测：先让起点更新一次
        self.global_belief = sensor_work(initial_cell,
                                         round(self.sensor_range / self.cell_size),
                                         self.global_belief,
                                         self.ground_truth)
        for i in range(N_AGENTS):
            self.agent_beliefs[i] = deepcopy(self.global_belief)

        # 产生多机器人初始位姿（从 FREE 的更新窗口里采样 N_AGENTS 个）
        tmp_info = MapInfo(self.agent_beliefs[0], self.belief_origin_x, self.belief_origin_y, self.cell_size)
        free_nodes, _ = get_updating_node_coords(np.array([0.0, 0.0]), tmp_info)
        choice = np.random.choice(free_nodes.shape[0], N_AGENTS, replace=True)
        self.robot_locations = np.array(free_nodes[choice])

        # 起点也做一次感知
        for cell in get_cell_position_from_coords(self.robot_locations, tmp_info).reshape(-1, 2):
            self.global_belief = sensor_work(cell, round(self.sensor_range / self.cell_size),
                                             self.global_belief, self.ground_truth)
        for i in range(N_AGENTS):
            self.agent_beliefs[i] = deepcopy(self.global_belief)

        # belief_info（默认给 agent0，用 get_agent_map(i) 拿其它 agent 的）
        self.belief_info = MapInfo(self.agent_beliefs[0], self.belief_origin_x, self.belief_origin_y, self.cell_size)
        # ground truth for critic
        self.ground_truth_info = MapInfo(self.ground_truth, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        self.global_frontiers = get_frontier_in_map(self.belief_info)
        if self.plot:
            self.frame_files = []

        H, W = self.ground_truth_size

        # —— 归因：首次发现者（-1 表示尚未归属）—— #
        self.ownership_map = -np.ones((H, W), dtype=np.int16)

        # —— 归因：每步每 agent 的“真实发现掩码”（由本体 sensor 引起的写图）—— #
        self._discover_free_masks = [np.zeros((H, W), dtype=bool) for _ in range(N_AGENTS)]
        self._discover_occ_masks  = [np.zeros((H, W), dtype=bool) for _ in range(N_AGENTS)]

        # —— 统计：每个 agent 累计“自己发现的 FREE 面积（m²）/ OCC 面积（m²）”—— #
        self.discovered_area_free_m2 = np.zeros(N_AGENTS, dtype=float)
        self.discovered_area_occ_m2  = np.zeros(N_AGENTS, dtype=float)
        self._cell_area = float(self.cell_size) ** 2

        # ----- 覆盖率相关的“上一步”快照，用于计算 per-agent 密集增益 -----
        self._known_cells_prev_per_agent = [
            int(np.count_nonzero(self.agent_beliefs[i] != UNKNOWN)) for i in range(N_AGENTS)
        ]
        H, W = self.ground_truth_size
        self._obs_total_cells = int(H * W)

        # 里程碑：是否已跨过阈值（避免重复加奖）
        self._obs_rate_thr = 0.995
        self._milestone_hit_prev = [
            (self._known_cells_prev_per_agent[i] / max(1, self._obs_total_cells)) >= self._obs_rate_thr
            for i in range(N_AGENTS)
        ]
        self.last_personal_obs_gain = [0.0 for _ in range(N_AGENTS)]


    # ---------- dataset ----------
    def import_ground_truth(self, episode_index):
        map_dir = f'dataset/maps_eval' if not self.test else f'dataset/maps_eval'
        map_list = []
        for root, _, files in os.walk(map_dir):
            for f in files:
                map_list.append(os.path.join(root, f))
        if not self.test:
            rng = random.Random(1)
            rng.shuffle(map_list)

        idx = episode_index % len(map_list)
        gt = (io.imread(map_list[idx], 1)).astype(int)

        # 起点cell（与单机一致）
        robot_cell = np.nonzero(gt == 208)
        robot_cell = np.array([np.array(robot_cell)[1, 10], np.array(robot_cell)[0, 10]])

        gt = (gt > 150) | ((gt <= 80) & (gt >= 50))
        gt = gt * 254 + 1
        return gt, robot_cell, map_list[idx]

    # ---------- comms ----------
    def _compute_comm_groups(self):
        """按 COMMS_RANGE 求连通分量（多跳可连）。"""
        adj = np.zeros((N_AGENTS, N_AGENTS), dtype=bool)
        for i in range(N_AGENTS):
            for j in range(i+1, N_AGENTS):
                if np.linalg.norm(self.robot_locations[i] - self.robot_locations[j]) <= COMMS_RANGE:
                    adj[i, j] = adj[j, i] = True
        groups, unseen = [], set(range(N_AGENTS))
        while unseen:
            r = unseen.pop()
            stack = [r]; comp = {r}
            while stack:
                u = stack.pop()
                for v in range(N_AGENTS):
                    if v in unseen and adj[u, v]:
                        unseen.remove(v)
                        comp.add(v)
                        stack.append(v)
            groups.append(comp)
        return groups

    def _merge_agent_beliefs(self, groups):
        """组内像素级合并：Known 覆盖 Unknown。"""
        for g in groups:
            merged = np.ones_like(self.global_belief) * UNKNOWN
            for i in g:
                known = (self.agent_beliefs[i] != UNKNOWN)
                merged[known] = self.agent_beliefs[i][known]
            for i in g:
                self.agent_beliefs[i] = merged.copy()

    # ---------- step ----------
    def step(self, next_waypoint, agent_id):
        old = self.robot_locations[agent_id]
        dist = np.linalg.norm(next_waypoint - old)
        self.total_travel_dist += dist
        self.agent_travel_dists[agent_id] += dist

        # 1) 位置更新
        self.robot_locations[agent_id] = next_waypoint

        # 2) 观测写回 global + own —— 在写入前做快照
        cell = get_cell_position_from_coords(next_waypoint, self.belief_info)

        # ===== 关键：写入前快照（只对本体写入导致的变化做归因） =====
        old_global = self.global_belief.copy()
        old_own    = self.agent_beliefs[agent_id].copy()

        # 传感器写入（与你原逻辑一致）
        self.global_belief = sensor_work(
            cell, round(self.sensor_range / self.cell_size),
            self.global_belief, self.ground_truth
        )
        self.agent_beliefs[agent_id] = sensor_work(
            cell, round(self.sensor_range / self.cell_size),
            self.agent_beliefs[agent_id], self.ground_truth
        )

        # ===== 关键：对比 “写入前后” 得到由本体传感器造成的真实增量 =====
        # 只看 global_belief 的变化即可（更严格：要求 own 也同步变化）
        newly_free_global = (old_global == UNKNOWN) & (self.global_belief == FREE)
        newly_occ_global  = (old_global == UNKNOWN) & (self.global_belief == OCCUPIED)

        # （可选更强约束）如果你想保证“本体自己也观到了”，再与 own 的变化相与：
        # newly_free_global &= (old_own == UNKNOWN) & (self.agent_beliefs[agent_id] == FREE)
        # newly_occ_global  &= (old_own == UNKNOWN) & (self.agent_beliefs[agent_id] == OCCUPIED)

        # 本步掩码先清空再写入（仅当前 agent）
        self._discover_free_masks[agent_id][:] = False
        self._discover_occ_masks[agent_id][:]  = False
        if newly_free_global.any():
            self._discover_free_masks[agent_id][newly_free_global] = True
            self.discovered_area_free_m2[agent_id] += float(newly_free_global.sum()) * self._cell_area
        if newly_occ_global.any():
            self._discover_occ_masks[agent_id][newly_occ_global] = True
            self.discovered_area_occ_m2[agent_id]  += float(newly_occ_global.sum()) * self._cell_area

        # 首次归属：仅对还未归属过的像素设 owner
        # 这样保证“谁先发现”在通信阶段不会被覆盖
        if newly_free_global.any():
            idx = newly_free_global & (self.ownership_map < 0)
            self.ownership_map[idx] = agent_id
        if newly_occ_global.any():
            idx = newly_occ_global & (self.ownership_map < 0)
            self.ownership_map[idx] = agent_id

        # 3) 组内合并（通信发生在归因之后，不会污染“谁先发现”的记账）
        groups = self._compute_comm_groups()
        self._merge_agent_beliefs(groups)

        # 4) 刷新该 agent 的 belief_info
        self.belief_info = MapInfo(
            self.agent_beliefs[agent_id],
            self.belief_origin_x, self.belief_origin_y, self.cell_size
        )


    # ---------- metrics ----------
    def evaluate_exploration_rate(self):
        self.explored_rate = np.sum(self.global_belief == FREE) / np.sum(self.ground_truth == FREE)

    def calculate_reward(self):
        """
        新版团队 + 个体奖励（与“个人belief也能涨就给奖”的需求对齐）

        返回:
            team_reward: float
                团队层面的奖励（仍然包含全局前沿推进 + 少量整体覆盖率增长 + 里程碑加成）
            per_agent_rewards: list[float] 长度 N_AGENTS
                每个体本步自己的已知像素增量（包括通信合并带来的增量）所对应的奖励，
                已做与 team_reward 同尺度的归一化，便于直接与其它项线性相加。
        """
        # ---- 归一化尺度，与旧版保持一致，量级稳定 ----
        denom = float(max(1, (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)))

        # ---- 1) 团队：全局前沿减少（与旧版一致）----
        self.evaluate_exploration_rate()
        binfo = MapInfo(self.global_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        global_frontiers = get_frontier_in_map(binfo)
        if len(global_frontiers) == 0:
            delta_frontier = len(self.global_frontiers)
        else:
            observed = self.global_frontiers - global_frontiers
            delta_frontier = len(observed)
        self.global_frontiers = global_frontiers
        R_frontier = float(delta_frontier) / denom  # 团队尺度

        # ---- 2) 个体：每个体“本步新增已知像素”（FREE 或 OCC 都算“已知”）----
        known_now = [int(np.count_nonzero(self.agent_beliefs[i] != UNKNOWN)) for i in range(N_AGENTS)]
        deltas = [max(0, known_now[i] - self._known_cells_prev_per_agent[i]) for i in range(N_AGENTS)]
        self._known_cells_prev_per_agent = known_now  # 刷新快照

        # 个体奖励：归一化到与 R_frontier 同量纲
        per_agent_rewards = [float(d) / denom for d in deltas]
        # 团队也取个“人均增量”作为温和的全局项，避免个体奖励过离散
        R_obs_mean = (float(np.mean(deltas)) if len(deltas) > 0 else 0.0) / denom

        # ---- 3) 里程碑：首次跨过 99.5% 覆盖率的小幅团队加成 ----
        rates_now = [known_now[i] / max(1, self._obs_total_cells) for i in range(N_AGENTS)]
        newly_hit = 0
        for i in range(N_AGENTS):
            hit_now = (rates_now[i] >= self._obs_rate_thr)
            if (not self._milestone_hit_prev[i]) and hit_now:
                newly_hit += 1
            self._milestone_hit_prev[i] = hit_now
        R_milestone = 0.5 * float(newly_hit)

        # ---- 4) 汇总：团队奖励 = 0.6*前沿推进 + 0.2*人均覆盖增量 + 里程碑 ----
        team_reward = 0.4 * R_frontier + 0.2 * R_obs_mean + R_milestone

        # 记录“最近一次个体覆盖增量奖励”，便于外部读取
        self.last_personal_obs_gain = per_agent_rewards

        return team_reward, per_agent_rewards


    # ---------- helpers ----------
    def get_agent_map(self, agent_id):
        return MapInfo(self.agent_beliefs[agent_id],
                       self.belief_origin_x, self.belief_origin_y, self.cell_size)

    def get_total_travel(self):
        return self.total_travel_dist

    # —— 新增：暴露 per-agent 和 max 路程 —— #
    def get_agent_travel(self):
        return self.agent_travel_dists.copy()

    def get_max_travel(self):
        return float(self.max_travel_dist)

    def pop_discovery_masks(self):
        """
        返回并清空本步的“由各自传感器造成”的增量掩码（UNKNOWN->FREE / UNKNOWN->OCC）。
        形状：list[N_AGENTS]，每个元素是 (H,W) 的 bool 数组。
        """
        free = [m.copy() for m in self._discover_free_masks]
        occ  = [m.copy() for m in self._discover_occ_masks]
        for m in self._discover_free_masks: m[:] = False
        for m in self._discover_occ_masks:  m[:] = False
        return free, occ

    def get_ownership_map(self):
        """返回首次归属图（-1 表示尚未被任何人首次发现）。"""
        return self.ownership_map.copy()

    def get_discovered_area(self):
        """
        返回每个 agent 的累计发现面积（m²）：FREE 与 OCC。
        """
        return (self.discovered_area_free_m2.copy(),
                self.discovered_area_occ_m2.copy())

    def get_map_balance_stats(self, which="free"):
        """
        多智能体探索均衡度指标。
        which: "free" | "occ" | "both"
        返回: dict(mean, std, cv, per_agent)
        """
        if which == "free":
            arr = self.discovered_area_free_m2
        elif which == "occ":
            arr = self.discovered_area_occ_m2
        else:  # both
            arr = self.discovered_area_free_m2 + self.discovered_area_occ_m2

        per_agent = arr.copy()
        mean = float(per_agent.mean())
        std  = float(per_agent.std(ddof=0))
        cv   = float(std / (mean + 1e-9))  # 变异系数：越小越均衡
        return dict(mean=mean, std=std, cv=cv, per_agent=per_agent)

    def get_last_personal_obs_gain(self):
        if hasattr(self, 'last_personal_obs_gain'):
            return list(self.last_personal_obs_gain)
        else:
            return [0.0 for _ in range(N_AGENTS)]
