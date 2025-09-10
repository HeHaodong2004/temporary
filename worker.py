# worker.py
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy
from collections import deque
import heapq
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import math

from env import Env
from agent import Agent
from utils import *
from node_manager import NodeManager
from ground_truth_node_manager import GroundTruthNodeManager
from parameter import *

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

def make_gif_safe(frame_paths, out_path, duration_ms=120):
    frame_paths = [p for p in frame_paths if os.path.exists(p)]
    frame_paths.sort()
    if not frame_paths: return
    frames = []
    base_size = None
    for p in frame_paths:
        try:
            im = Image.open(p).convert("RGB")
            if base_size is None: base_size = im.size
            elif im.size != base_size: im = im.resize(base_size, Image.BILINEAR)
            frames.append(im)
        except Exception:
            continue
    if not frames: return
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)


# ===================== 合同结构（双点，不扩圈） =====================
class Contract:
    """
    Rendezvous 合同（“协商-待激活(armed)” -> “执行(active)” -> done/failed）
    - P_list: [P_primary, P_backup] 至多两个点，备选可以为空
    - r: 区域半径（米）
    - t_min_list / t_max_list: 每个点对应的时间窗（协商时估计；真正执行时会重算调度）
    - participants: 参与 agent id
    - status: 'armed' | 'active' | 'done' | 'failed'
    - target_idx: 当前执行的目标索引（0=主点，1=备选）
    - meta: 执行阶段的调度信息：{'T_tar','t_dep'[N_AGENTS],'q_i'[N_AGENTS]} 针对当前 target_idx
    """
    def __init__(self, P_list, r, t_min_list, t_max_list, participants, created_t):
        self.P_list = [np.array(P, dtype=float) for P in P_list]
        self.r = float(r)
        self.t_min_list = [int(x) for x in t_min_list]
        self.t_max_list = [int(x) for x in t_max_list]
        self.participants = set(participants)
        self.created_t = int(created_t)
        self.status = 'armed'
        self.target_idx = 0  # 默认先用主点
        self.meta = {}       # 执行时填充

    @property
    def P(self):
        return self.P_list[self.target_idx]

    @property
    def t_min(self):
        return self.t_min_list[self.target_idx]

    @property
    def t_max(self):
        return self.t_max_list[self.target_idx]

    def within_region(self, pos_xy):
        return np.linalg.norm(np.asarray(pos_xy, dtype=float) - self.P) <= self.r

    def has_backup(self):
        return len(self.P_list) >= 2

    def switch_to_backup(self):
        if self.has_backup() and self.target_idx == 0:
            self.target_idx = 1
            self.meta = {}  # 切换目标需要重新计算调度
            return True
        return False


# ===================== D* Lite（4邻） =====================
class PriorityQueue:
    def __init__(self): self.data = []
    def push(self, k, s): heapq.heappush(self.data, (k, s))
    def pop(self): return heapq.heappop(self.data)
    def top_key(self): return (float('inf'), float('inf')) if not self.data else self.data[0][0]
    def empty(self): return len(self.data) == 0
    def remove(self, s):
        self.data = [(k, x) for (k, x) in self.data if x != s]
        heapq.heapify(self.data)

class DStarLite:
    def __init__(self, cost_map, start_rc, goal_rc, heuristic=lambda a,b:(abs(a[0]-b[0])+abs(a[1]-b[1]))):
        self.H, self.W = cost_map.shape
        self.cmap = cost_map
        self.s_start = tuple(start_rc)
        self.s_goal = tuple(goal_rc)
        self.rhs, self.g = {}, {}
        self.U = PriorityQueue()
        self.km = 0.0
        self.h = heuristic
        self.s_last = self.s_start
        for r in range(self.H):
            for c in range(self.W):
                self.g[(r,c)] = float('inf')
                self.rhs[(r,c)] = float('inf')
        self.rhs[self.s_goal] = 0.0
        self.U.push(self._calc_key(self.s_goal), self.s_goal)

    def _calc_key(self, s):
        g_rhs = min(self.g[s], self.rhs[s])
        return (g_rhs + self.h(self.s_start, s) + self.km, g_rhs)

    def _neighbors(self, s):
        r, c = s
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.H and 0 <= nc < self.W and np.isfinite(self.cmap[nr, nc]):
                yield (nr, nc)

    def _cost(self, a, b):
        return self.cmap[b[0], b[1]]

    def update_vertex(self, s):
        if s != self.s_goal:
            self.rhs[s] = min([self.g[n] + self._cost(s, n) for n in self._neighbors(s)] + [float('inf')])
        in_U = False
        for k, x in self.U.data:
            if x == s:
                in_U = True; break
        if self.g[s] != self.rhs[s]:
            if in_U: self.U.remove(s)
            self.U.push(self._calc_key(s), s)
        else:
            if in_U: self.U.remove(s)

    def compute_shortest_path(self, max_expand=100000):
        cnt = 0
        while (self.U.top_key() < self._calc_key(self.s_start)) or (self.rhs[self.s_start] != self.g[self.s_start]):
            if cnt >= max_expand: break
            cnt += 1
            k_old, u = self.U.pop()
            k_new = self._calc_key(u)
            if k_old < k_new:
                self.U.push(k_new, u)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for n in self._neighbors(u):
                    self.update_vertex(n)
            else:
                self.g[u] = float('inf')
                for n in list(self._neighbors(u)) + [u]:
                    self.update_vertex(n)

    def update_start(self, new_start):
        hdiff = self.h(self.s_last, new_start)
        self.km += hdiff
        self.s_start = tuple(new_start)
        self.s_last = tuple(new_start)

    def update_cost_map(self, new_cost_map, changed_cells=None):
        self.cmap = new_cost_map
        if changed_cells is None:
            iters = [(r,c) for r in range(self.H) for c in range(self.W) if np.isfinite(self.cmap[r,c])]
        else:
            iters = set()
            for (r,c) in changed_cells:
                iters.add((r,c))
                for n in [(r+1,c),(r-1,c),(r,c+1),(r,c-1)]:
                    if 0 <= n[0] < self.H and 0 <= n[1] < self.W and np.isfinite(self.cmap[n[0],n[1]]):
                        iters.add(n)
        for s in iters:
            self.update_vertex(s)

    def next_step_on_policy(self):
        s = self.s_start
        best_n, best_v = s, float('inf')
        for n in self._neighbors(s):
            v = self.g[n] + self._cost(s, n)
            if v < best_v:
                best_v, best_n = v, n
        return best_n if best_n != s else s

    def extract_path(self, max_len=100000):
        path = [self.s_start]
        cur = self.s_start
        seen = set([cur])
        for _ in range(max_len):
            nxt = self.next_step_on_policy()
            if nxt == cur: break
            path.append(nxt)
            if nxt in seen: break
            seen.add(nxt)
            cur = nxt
            if cur == self.s_goal: break
        return path


# ===================== Worker =====================
class Worker:
    def __init__(self, meta_agent_id, policy_net, predictor, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image)
        self.node_manager = NodeManager(plot=self.save_image)
        self.robots = [Agent(i, policy_net, predictor, self.node_manager, device=device, plot=save_image) for i in range(N_AGENTS)]
        self.gtnm = GroundTruthNodeManager(self.node_manager, self.env.ground_truth_info, device=device, plot=save_image)

        self.episode_buffer = [[] for _ in range(27)]
        self.perf_metrics = dict()

        self.last_known_locations = [self.env.robot_locations.copy() for _ in range(N_AGENTS)]
        self.last_known_intents = [{aid: [] for aid in range(N_AGENTS)} for _ in range(N_AGENTS)]

        self.run_dir = os.path.join(gifs_path, f"run_g{self.global_step}_w{self.meta_agent_id}_{os.getpid()}_{int(time.time()*1000)}")
        if self.save_image: os.makedirs(self.run_dir, exist_ok=True)
        self.env.frame_files = []

        # Rendezvous
        self.contract: Contract = None
        self.candidate_buffer = []  # [{'P','score','etas','risk','ig_total','t_min','t_max'}...]
        self.cand_last_update_t = -1

        # D* Lite（仅在出发后使用）
        self._planners = [None] * N_AGENTS
        self._planner_goal = [None] * N_AGENTS

        self.was_fully_connected = False

        gt_map = self.env.ground_truth_info.map
        self._gt_free_total = int(np.count_nonzero(gt_map == FREE))

    # critic 通道对齐
    def _match_intent_channels(self, obs_pack):
        n, m, e, ci, ce, ep = obs_pack
        need, got = NODE_INPUT_DIM, n.shape[-1]
        if got < need:
            pad = torch.zeros((n.shape[0], n.shape[1], need - got), dtype=n.dtype, device=n.device)
            n = torch.cat([n, pad], dim=-1)
        return [n, m, e, ci, ce, ep]

    # ========================= 主循环 =========================
    def run_episode(self):
        done = False

        # 初始化：图、预测、意图
        for i, r in enumerate(self.robots):
            r.update_graph(self.env.get_agent_map(i), self.env.robot_locations[i])
        for r in self.robots:
            r.update_predict_map()
        for i, r in enumerate(self.robots):
            r.update_planning_state(self.env.robot_locations)
            self.last_known_intents[r.id][r.id] = deepcopy(r.intent_seq)

        groups0 = self._compute_groups_from_positions(self.env.robot_locations)
        self.was_fully_connected = (len(groups0) == 1 and len(groups0[0]) == N_AGENTS)

        if self.save_image:
            self.plot_env(step=0)

        # ================= 主循环 =================
        for t in range(MAX_EPISODE_STEP):
            # 地图/掩膜
            global_map_info = MapInfo(self.env.global_belief,
                                      self.env.belief_origin_x,
                                      self.env.belief_origin_y,
                                      self.env.cell_size)
            belief_map = global_map_info.map
            p_free = (self.robots[0].pred_mean_map_info.map.astype(np.float32) / 255.0
                      if self.robots[0].pred_mean_map_info is not None
                      else (belief_map == FREE).astype(np.float32))
            trav_mask = (((p_free) >= RDV_TAU_FREE) | (belief_map == FREE)) & (belief_map != OCCUPIED)
            unknown_mask = (belief_map == UNKNOWN)

            # 连通状态
            groups = self._compute_groups_from_positions(self.env.robot_locations)
            is_fully_connected = (len(groups) == 1 and len(groups[0]) == N_AGENTS)

            # -------- 协商阶段：仅在“全连通且无合同”时，低频构建候选并立“武装合同”(armed) --------
            if is_fully_connected and self.contract is None and (t % RDV_CAND_UPDATE_EVERY == 0):
                try:
                    self._update_candidate_buffer(global_map_info, trav_mask, unknown_mask, p_free, t)
                    best = self._select_best_candidate_from_buffer(idx=0)
                    backup = self._select_best_candidate_from_buffer(idx=1, min_dist=RDV_BACKUP_MIN_DIST, ref=best)
                    if best is not None:
                        P_list = [best['P']]
                        tmins = [best['t_min']]
                        tmaxs = [best['t_max']]
                        if backup is not None:
                            P_list.append(backup['P'])
                            tmins.append(backup['t_min'])
                            tmaxs.append(backup['t_max'])
                        r_meet = RDV_REGION_FRAC * COMMS_RANGE
                        self.contract = Contract(P_list=P_list,
                                                 r=r_meet,
                                                 t_min_list=tmins,
                                                 t_max_list=tmaxs,
                                                 participants=set(range(N_AGENTS)),
                                                 created_t=t)
                        if RDV_VERBOSE:
                            if backup is not None:
                                print(f"[RDV] Negotiated@t={t} primary={best['P']}, backup={backup['P']}, r={r_meet:.1f}")
                            else:
                                print(f"[RDV] Negotiated@t={t} primary={best['P']}, r={r_meet:.1f}")
                except Exception as e:
                    print(f"[RDV] negotiation failed at t={t}: {e}")

            # -------- 断联触发：从 armed -> active，并在触发时“重算调度”（按当前 t） --------
            if (not is_fully_connected) and self.was_fully_connected and (self.contract is not None) and (self.contract.status == 'armed'):
                # 尝试对“当前目标（主点）”做机会约束调度；若失败，改用备选点
                ok = self._activate_contract_with_schedule(self.contract, self.contract.target_idx,
                                                           t, global_map_info, belief_map, p_free)
                if not ok and self.contract.switch_to_backup():
                    ok2 = self._activate_contract_with_schedule(self.contract, self.contract.target_idx,
                                                                t, global_map_info, belief_map, p_free)
                    if not ok2:
                        # 两个点全失败：合同失败
                        self.contract.status = 'failed'
                        if RDV_VERBOSE:
                            print(f"[RDV] Activate failed@t={t} (both targets infeasible).")
                elif ok and RDV_VERBOSE:
                    print(f"[RDV] Contract ACTIVATED@t={t} target={self.contract.target_idx}, "
                          f"T_tar={self.contract.meta['T_tar']}, t_dep={self.contract.meta['t_dep']}")

            # ---------- 自由探索：下一步候选（用原策略；用于非干预期 & 出发前期） ----------
            picks_raw, dists = [], []
            for i, r in enumerate(self.robots):
                obs = r.get_observation(self.last_known_locations[i], self.last_known_intents[i])
                c_obs, _ = self.gtnm.get_ground_truth_observation(
                    r.location, r.pred_mean_map_info, self.env.robot_locations
                )
                r.save_observation(obs, self._match_intent_channels(c_obs))
                nxt, _, act = r.select_next_waypoint(obs)
                r.save_action(act)
                picks_raw.append(nxt)
                dists.append(np.linalg.norm(nxt - r.location))

            # ---------- 执行（active）：出发前不干预；出发后用 D* Lite ----------
            picks = []
            for i, r in enumerate(self.robots):
                if self.contract is None or self.contract.status != 'active':
                    picks.append(picks_raw[i])
                    continue

                # 若主点不可达，尝试切换到备选点（仅切换一次）
                goal_rc = self._nearest_reachable_in_region(self.contract.P, self.contract.r, trav_mask, global_map_info)
                if goal_rc is None and self.contract.target_idx == 0 and self.contract.switch_to_backup():
                    if RDV_VERBOSE:
                        print(f"[RDV] Switch to BACKUP target@t={t}")
                    # 切换后立刻重算调度
                    ok = self._activate_contract_with_schedule(self.contract, self.contract.target_idx,
                                                               t, global_map_info, belief_map, p_free)
                    if not ok:
                        # 备选也不可行，合同失败，回到自由探索
                        self.contract.status = 'failed'
                        picks.append(picks_raw[i])
                        continue
                    goal_rc = self._nearest_reachable_in_region(self.contract.P, self.contract.r, trav_mask, global_map_info)

                # 仍不可达：合同失败，回到自由探索
                if goal_rc is None:
                    self.contract.status = 'failed'
                    picks.append(picks_raw[i])
                    continue

                # 圈内：巡航但不离圈
                if self.contract.within_region(r.location):
                    picks.append(self._in_zone_patrol_step(i, r, global_map_info))
                    continue

                # 出发时刻
                t_dep_i = int(self.contract.meta['t_dep'][i])
                if t < t_dep_i:
                    picks.append(picks_raw[i])  # 出发前完全不干预
                    continue

                # 出发后：D* Lite 强制导航
                cost_map = self._build_cost_map(belief_map, p_free)
                start_rc = self._world_to_cell_rc(r.location, global_map_info)

                if self._planners[i] is None or self._planner_goal[i] != tuple(goal_rc):
                    self._planners[i] = DStarLite(cost_map, start_rc, goal_rc)
                    self._planner_goal[i] = tuple(goal_rc)
                else:
                    if (t % RDV_PLAN_REPLAN_EVERY) == 0:
                        self._planners[i].update_start(start_rc)
                        if (t % RDV_COSTMAP_UPDATE_EVERY) == 0:
                            self._planners[i].update_cost_map(cost_map, changed_cells=None)

                # 预算
                T_tar = int(self.contract.meta['T_tar'])
                budget = max(1, int(T_tar - t))
                self._planners[i].compute_shortest_path(max_expand=RDV_DSTAR_MAX_EXPAND)
                path = self._planners[i].extract_path(max_len=budget+5)
                if len(path) <= 1:
                    picks.append(picks_raw[i])
                else:
                    nxt_rc = path[1]
                    nxt_xy = np.array([
                        global_map_info.map_origin_x + nxt_rc[1] * global_map_info.cell_size,
                        global_map_info.map_origin_y + nxt_rc[0] * global_map_info.cell_size
                    ], dtype=float)
                    picks.append(nxt_xy)

            # ---------- 冲突消解 & 推进 ----------
            picks = self.resolve_conflicts(picks, dists)
            prev_max = self.env.get_agent_travel().max()
            prev_total = self.env.get_total_travel()
            for r, loc in zip(self.robots, picks):
                self.env.step(loc, r.id)
            self.env.max_travel_dist = self.env.get_agent_travel().max()
            delta_max = self.env.max_travel_dist - prev_max
            delta_total = self.env.get_total_travel() - prev_total

            # 通信同步
            groups_after_move = self._compute_groups_from_positions(self.env.robot_locations)
            for g in groups_after_move:
                for i in g:
                    for j in g:
                        self.last_known_locations[i][j] = self.env.robot_locations[j].copy()
                        self.last_known_intents[i][j] = deepcopy(self.robots[j].intent_seq)

            # 图/预测/意图更新
            for i, r in enumerate(self.robots):
                r.update_graph(self.env.get_agent_map(i), self.env.robot_locations[i])
                r.update_predict_map()
                r.update_planning_state(self.last_known_locations[i], self.last_known_intents[i])
                self.last_known_intents[i][r.id] = deepcopy(r.intent_seq)

            # 合同完成判定（全体进圈即 done）；不再扩圈，不再 fallback
            if self.contract is not None and self.contract.status == 'active':
                all_in = all(self.contract.within_region(self.robots[aid].location) for aid in self.contract.participants)
                if all_in:
                    self.contract.status = 'done'
                    if RDV_VERBOSE:
                        print(f"[RDV] DONE@t={t}")
                    # 完成后清空合同，允许下一个阶段再协商
                    self.contract = None
                    # 清空规划器
                    self._planners = [None] * N_AGENTS
                    self._planner_goal = [None] * N_AGENTS

            # 奖励与终止
            team_reward_env, per_agent_obs_rewards = self.env.calculate_reward()
            team_reward = (
                team_reward_env
                - ((delta_max / UPDATING_MAP_SIZE) * MAX_TRAVEL_COEF)
                - ((delta_total / UPDATING_MAP_SIZE) * TOTAL_TRAVEL_COEF)
            )

            utilities_empty = all((r.utility <= 3).all() for r in self.robots)
            if self._gt_free_total > 0:
                per_agent_free_counts = [int(np.count_nonzero(r.map_info.map == FREE)) for r in self.robots]
                per_agent_cov = [c / self._gt_free_total for c in per_agent_free_counts]
                coverage_ok = all(c >= 0.995 for c in per_agent_cov)
            else:
                coverage_ok = False

            done = utilities_empty or coverage_ok
            if done:
                team_reward += 10.0

            for i, r in enumerate(self.robots):
                indiv_total = team_reward + per_agent_obs_rewards[i]
                r.save_reward(indiv_total)
                r.save_done(done)
                next_obs = r.get_observation(self.last_known_locations[i], self.last_known_intents[i])
                c_next_obs, _ = self.gtnm.get_ground_truth_observation(
                    r.location, r.pred_mean_map_info, self.env.robot_locations
                )
                r.save_next_observations(next_obs, self._match_intent_channels(c_next_obs))

            if self.save_image:
                self.plot_env(step=t + 1)

            if done:
                break

            self.was_fully_connected = is_fully_connected

        # 总结
        self.perf_metrics.update({
            'travel_dist': self.env.get_total_travel(),
            'max_travel': self.env.get_max_travel(),
            'explored_rate': self.env.explored_rate,
            'success_rate': done
        })

        if self.save_image:
            make_gif_safe(self.env.frame_files,
                          os.path.join(self.run_dir, f"ep_{self.global_step}.gif"))

        for r in self.robots:
            for k in range(len(self.episode_buffer)):
                self.episode_buffer[k] += r.episode_buffer[k]

    # ---------------- 冲突消解（保留你的实现） ----------------
    def resolve_conflicts(self, picks, dists):
        """
        改进版冲突消解：
        - 优先让“正在执行 RDV 的机器人”（有 D* 规划器，且不在圈内）先挑邻居
        - 其次才是自由探索者
        其它逻辑保持与你版本一致。
        """
        picks = np.array(picks).reshape(-1, 2)

        # 识别“强制前往 RDV”的 agent（无需当前 t）：
        # 有激活合同 + 该 agent 存在 D* 规划器 + 还不在圈内
        forced_idx, free_idx = [], []
        for i in range(len(self.robots)):
            in_forced = (
                self.contract is not None and self.contract.status == 'active' and
                (self._planners[i] is not None) and
                (not self.contract.within_region(self.robots[i].location))
            )
            (forced_idx if in_forced else free_idx).append(i)

        # 强制者先消解，自由者后消解
        order = forced_idx + free_idx

        chosen_complex, resolved = set(), [None] * len(self.robots)
        for rid in order:
            robot = self.robots[rid]
            # 尽量从“离目标点更近”的邻居开始（这里依旧按与 picks[rid] 的距离近-先）
            try:
                neighbor_coords = sorted(
                    list(robot.node_manager.nodes_dict.find(robot.location.tolist()).data.neighbor_set),
                    key=lambda c: np.linalg.norm(np.array(c) - picks[rid])
                )
            except Exception:
                neighbor_coords = [robot.location.copy()]

            picked = None
            for cand in neighbor_coords:
                key = complex(cand[0], cand[1])
                if key not in chosen_complex:
                    picked = np.array(cand); break
            resolved[rid] = picked if picked is not None else robot.location.copy()
            chosen_complex.add(complex(resolved[rid][0], resolved[rid][1]))

        return np.array(resolved).reshape(-1, 2)

    # ======================= 候选池（选点） =======================
    def _update_candidate_buffer(self, map_info: MapInfo, trav_mask, unknown_mask, p_free, t_now: int):
        H, W = trav_mask.shape
        cell_size = float(map_info.cell_size)

        r_idx, c_idx = np.where(unknown_mask & trav_mask)
        if r_idx.size == 0:
            self.candidate_buffer = []
            self.cand_last_update_t = t_now
            return

        # 下采样 & 稀疏
        n_cand = min(r_idx.size, RDV_CAND_K)
        sel = np.random.choice(r_idx.size, n_cand, replace=False)
        rc = np.stack([r_idx[sel], c_idx[sel]], axis=1)
        if RDV_CAND_STRIDE > 1:
            keep, seen = [], set()
            for r, c in rc:
                key = (int(r // RDV_CAND_STRIDE), int(c // RDV_CAND_STRIDE))
                if key in seen: continue
                seen.add(key); keep.append((r, c))
            rc = np.array(keep, dtype=int)

        # BFS 距离图
        dist_maps = []
        for agent in self.robots:
            start_rc = self._world_to_cell_rc(agent.location, map_info)
            if not trav_mask[start_rc[0], start_rc[1]]:
                start_rc = self._find_nearest_valid_cell(trav_mask, np.array(start_rc))
            dist_maps.append(self._bfs_dist_map(trav_mask, tuple(start_rc)))

        candidates = []
        R_info_pix = int(RDV_INFO_RADIUS_M / cell_size)
        for (r_c, c_c) in rc:
            r0 = max(0, r_c - R_info_pix); r1 = min(H, r_c + R_info_pix + 1)
            c0 = max(0, c_c - R_info_pix); c1 = min(W, c_c + R_info_pix + 1)
            local_ig = int(unknown_mask[r0:r1, c0:c1].sum())

            etas, risks, path_ig = [], [], []
            feasible = True
            for j, agent in enumerate(self.robots):
                d_steps = dist_maps[j][r_c, c_c]
                if not np.isfinite(d_steps):
                    feasible = False; break
                eta_j = d_steps / max(NODE_RESOLUTION, 1e-6)
                etas.append(float(eta_j))
                r_s, c_s = self._world_to_cell_rc(agent.location, map_info)
                line = self._bresenham_line_rc(r_s, c_s, r_c, c_c)
                line_risk, line_ig = 0.0, 0
                for (rr, cc) in line:
                    if 0 <= rr < H and 0 <= cc < W:
                        line_risk += float(1.0 - p_free[rr, cc])
                        if unknown_mask[rr, cc]: line_ig += 1
                risks.append(line_risk / max(len(line), 1))
                path_ig.append(line_ig)
            if not feasible: continue

            ig_total = float(RDV_ALPHA * (sum(path_ig) + local_ig))
            disp = float(max(etas) - min(etas))
            risk_total = float(sum(risks))
            score = ig_total - RDV_BETA * disp - RDV_GAMMA * risk_total

            eta_max = max(etas)
            t_mid = t_now + int(round(eta_max))
            t_min = t_mid - int(round(RDV_WINDOW_ALPHA_EARLY * eta_max + RDV_WINDOW_BETA_EARLY))
            t_max = t_mid + int(round(RDV_WINDOW_ALPHA_LATE  * eta_max + RDV_WINDOW_BETA_LATE))

            P_world = np.array([map_info.map_origin_x + c_c * cell_size,
                                map_info.map_origin_y + r_c * cell_size], dtype=float)
            candidates.append({
                'P': P_world, 'score': score, 'etas': etas, 'risk': risk_total, 'ig_total': ig_total,
                't_min': t_min, 't_max': t_max
            })

        candidates.sort(key=lambda d: d['score'], reverse=True)
        self.candidate_buffer = candidates[:RDV_TOP_M]
        self.cand_last_update_t = t_now

    def _select_best_candidate_from_buffer(self, idx=0, min_dist=0.0, ref=None):
        if not self.candidate_buffer: return None
        if idx == 0: return self.candidate_buffer[0]
        # 选第 idx 个且与 ref 距离大于 min_dist
        taken = 0
        for c in self.candidate_buffer:
            if ref is not None:
                if np.linalg.norm(np.asarray(c['P']) - np.asarray(ref['P'])) < max(1e-6, min_dist):
                    continue
            if taken == idx - 1:
                return c
            taken += 1
        return None

    # ======================= 激活合约时重算调度 =======================
    def _activate_contract_with_schedule(self, contract: Contract, target_idx, t_now, map_info, belief_map, p_free):
        P = contract.P_list[target_idx]
        r = contract.r
        sched = self._chance_constrained_schedule(P, r, t_now, map_info, belief_map, p_free)
        if sched is None:
            return False
        T_tar, dep_times, q_i, eps_used = sched
        contract.status = 'active'
        contract.target_idx = target_idx
        contract.meta = {'T_tar': int(T_tar), 't_dep': [int(x) for x in dep_times],
                         'q_i': [float(x) for x in q_i], 'eps': float(eps_used)}
        # 清空规划器（目标变更/首次激活）
        self._planners = [None] * N_AGENTS
        self._planner_goal = [None] * N_AGENTS
        return True

    # ======================= CC-ODS（机会约束调度） =======================
    def _chance_constrained_schedule(self, P, r, t_now, map_info, belief_map, p_free):
        H, W = belief_map.shape
        cell_size = float(map_info.cell_size)

        def nearest_goal_rc(trav_mask_s):
            r_pix = int(max(1, round(r / cell_size)))
            c_rc = self._world_to_cell_rc(P, map_info)
            best, bestd = None, float('inf')
            r0 = max(0, c_rc[0]-r_pix); r1 = min(H, c_rc[0]+r_pix+1)
            c0 = max(0, c_rc[1]-r_pix); c1 = min(W, c_rc[1]+r_pix+1)
            for rr in range(r0, r1):
                for cc in range(c0, c1):
                    if trav_mask_s[rr, cc]:
                        d = (rr - c_rc[0])**2 + (cc - c_rc[1])**2
                        if d < bestd: bestd, best = d, (rr, cc)
            return best

        # 采样可通行图（保留已知 FREE，排除 OCCUPIED，其他按 p_free 独立伯努利）
        samples = []
        for s in range(RDV_TT_N_SAMPLES):
            rand = np.random.rand(H, W).astype(np.float32)
            trav_s = (((rand < p_free) | (belief_map == FREE)) & (belief_map != OCCUPIED))
            samples.append(trav_s)

        # 为每个样本、每个 agent 计算到“区域内最近可达点”的步数 -> 时间
        T_samples = {aid: [] for aid in range(N_AGENTS)}
        for s in range(RDV_TT_N_SAMPLES):
            trav_s = samples[s]
            goal_rc = nearest_goal_rc(trav_s)
            if goal_rc is None:
                for aid in range(N_AGENTS): T_samples[aid].append(float('inf'))
                continue
            dist_maps = []
            for aid in range(N_AGENTS):
                start_rc = self._world_to_cell_rc(self.robots[aid].location, map_info)
                if not trav_s[start_rc[0], start_rc[1]]:
                    start_rc = self._find_nearest_valid_cell(trav_s, np.array(start_rc))
                dist_maps.append(self._bfs_dist_map(trav_s, tuple(start_rc)))
            for aid in range(N_AGENTS):
                d = dist_maps[aid][goal_rc[0], goal_rc[1]]
                T_samples[aid].append(float(d / max(NODE_RESOLUTION, 1e-6)) if np.isfinite(d) else float('inf'))

        # 取 (1-eps) 分位数
        q_i = []
        for aid in range(N_AGENTS):
            arr = np.array(T_samples[aid], dtype=np.float32)
            feasible = np.isfinite(arr)
            if feasible.sum() < max(1, int((1.0 - RDV_EPSILON) * RDV_TT_N_SAMPLES)):
                return None
            vals = np.sort(arr[feasible])
            k = max(0, int(math.ceil((1.0 - RDV_EPSILON) * len(vals))) - 1)
            q_i.append(float(vals[k]))

        # 共同到达时刻与出发时刻
        T_tar = max([t_now + qi for qi in q_i])
        dep_times = [int(max(t_now, math.floor(T_tar - qi))) for qi in q_i]
        return int(T_tar), dep_times, q_i, RDV_EPSILON

    # ======================= D* Lite 相关辅助 =======================
    def _build_cost_map(self, belief_map, p_free):
        """
        cost = 1 + λ_risk*(1 - p_free) + ε_known
        - OCCUPIED -> inf
        - 若 RDV_ALLOW_UNKNOWN_IN_DSTAR=False：UNKNOWN -> inf（与 Node 图一致）
        否则 UNKNOWN 根据 p_free 与阈值决定是否可走
        """
        H, W = belief_map.shape
        cost = np.ones((H, W), dtype=np.float32)

        # 风险项：鼓励走高 p_free 的格
        cost += RDV_RISK_LAMBDA * (1.0 - p_free)

        # 已知区域略加惩罚（未知略便宜，沿途还能探索）
        known = (belief_map != UNKNOWN) & (belief_map != OCCUPIED)
        cost += RDV_INFO_EPS * known.astype(np.float32)

        # 基础阻塞
        cost[belief_map == OCCUPIED] = np.inf

        if not RDV_ALLOW_UNKNOWN_IN_DSTAR:
            # 与 Node 图一致：UNKNOWN 直接不可走
            cost[belief_map == UNKNOWN] = np.inf
        else:
            # 允许高置信未知，低置信未知仍不可走
            mask_bad_unknown = (belief_map == UNKNOWN) & (p_free < RDV_TAU_FREE)
            cost[mask_bad_unknown] = np.inf

        return cost



    def _nearest_reachable_in_region(self, P, r, trav_mask, map_info):
        H, W = trav_mask.shape
        r_pix = int(max(1, round(r / map_info.cell_size)))
        c_rc = self._world_to_cell_rc(P, map_info)
        best, bestd = None, float('inf')
        r0 = max(0, c_rc[0]-r_pix); r1 = min(H, c_rc[0]+r_pix+1)
        c0 = max(0, c_rc[1]-r_pix); c1 = min(W, c_rc[1]+r_pix+1)
        for rr in range(r0, r1):
            for cc in range(c0, c1):
                if not trav_mask[rr, cc]: continue
                d = (rr - c_rc[0])**2 + (cc - c_rc[1])**2
                if d < bestd:
                    bestd, best = d, (rr, cc)
        return best

    def _in_zone_patrol_step(self, aid, agent, map_info):
        try:
            node = self.node_manager.nodes_dict.find(agent.location.tolist()).data
            neighbor_coords = list(node.neighbor_set)
        except Exception:
            neighbor_coords = [agent.location.copy()]

        def inside(xy): return self.contract.within_region(xy)
        best, best_score = agent.location.copy(), -1e18
        for nb in neighbor_coords:
            if not inside(nb): continue
            s_frontier = 0.0
            try:
                nd = self.node_manager.nodes_dict.find(np.around(nb, 1).tolist()).data
                s_frontier = float(max(nd.utility, 0.0))
            except Exception:
                pass
            rr, cc = self._world_to_cell_rc(nb, map_info)
            risk = 1.0
            pmap = (self.robots[0].pred_mean_map_info.map.astype(np.float32) / 255.0
                    if self.robots[0].pred_mean_map_info is not None else 1.0)
            if isinstance(pmap, np.ndarray):
                if 0 <= rr < pmap.shape[0] and 0 <= cc < pmap.shape[1]:
                    risk = 1.0 - float(pmap[rr, cc])
            score = s_frontier - RDV_RISK_LAMBDA * risk
            if score > best_score:
                best_score, best = score, np.array(nb, dtype=float)
        return best

    # ======================= 小工具 & 图/几何 =======================
    def _compute_groups_from_positions(self, positions):
        n = len(positions)
        if n == 0: return []
        used = [False] * n
        groups = []
        for i in range(n):
            if used[i]: continue
            comp, q = [], [i]
            used[i] = True
            while q:
                u = q.pop()
                comp.append(u)
                for v in range(n):
                    if used[v]: continue
                    if np.linalg.norm(np.asarray(positions[u]) - np.asarray(positions[v])) <= COMMS_RANGE + 1e-6:
                        used[v] = True; q.append(v)
            groups.append(tuple(sorted(comp)))
        return groups

    def _world_to_cell_rc(self, world_xy, map_info):
        cell = get_cell_position_from_coords(np.array(world_xy, dtype=float), map_info)
        return int(cell[1]), int(cell[0])

    def _find_nearest_valid_cell(self, mask, start_rc):
        q = deque([tuple(start_rc)]); visited = {tuple(start_rc)}
        H, W = mask.shape
        while q:
            r, c = q.popleft()
            if 0 <= r < H and 0 <= c < W and mask[r, c]:
                return np.array([r, c])
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                    q.append((nr, nc)); visited.add((nr, nc))
        return np.asarray(start_rc)

    def _bfs_dist_map(self, trav_mask, start_rc):
        H, W = trav_mask.shape
        dist_map = np.full((H, W), np.inf, dtype=np.float32)
        q = deque([start_rc])
        if 0 <= start_rc[0] < H and 0 <= start_rc[1] < W and trav_mask[start_rc]:
            dist_map[start_rc[0], start_rc[1]] = 0.0
        while q:
            r, c = q.popleft()
            base = dist_map[r, c]
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and trav_mask[nr, nc] and np.isinf(dist_map[nr, nc]):
                    dist_map[nr, nc] = base + 1.0
                    q.append((nr, nc))
        return dist_map

    def _bfs_path_rc(self, trav_mask, start_rc, goal_rc):
        H, W = trav_mask.shape
        parent = {tuple(start_rc): None}
        q = deque([tuple(start_rc)])
        if not (0 <= start_rc[0] < H and 0 <= start_rc[1] < W and trav_mask[start_rc[0], start_rc[1]]):
            return []
        while q:
            r, c = q.popleft()
            if (r, c) == tuple(goal_rc):
                path = []
                cur = (r, c)
                while cur is not None:
                    path.append(cur); cur = parent[cur]
                path.reverse(); return path
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and trav_mask[nr, nc] and (nr, nc) not in parent:
                    parent[(nr, nc)] = (r, c); q.append((nr, nc))
        return []

    def _bresenham_line_rc(self, r0, c0, r1, c1):
        points = []
        dr, dc = abs(r1-r0), abs(c1-c0)
        sr, sc = (1 if r1>=r0 else -1), (1 if c1>=c0 else -1)
        if dc > dr:
            err, r = dc//2, r0
            for c in range(c0, c1+sc, sc):
                points.append((r, c)); err -= dr
                if err < 0: r += sr; err += dc
        else:
            err, c = dr//2, c0
            for r in range(r0, r1+sr, sr):
                points.append((r, c)); err -= dc
                if err < 0: c += sc; err += dr
        return points
    
    def _pick_graph_neighbor_toward(self, agent, target_xy, map_info):
        """
        在当前节点的 neighbor_set 里选一个“能严格拉近到 target_xy 的邻居”；
        若没有更近的邻居，返回 agent.location（调用方再决定 fallback）。
        """
        try:
            node = self.node_manager.nodes_dict.find(agent.location.tolist()).data
            neighbor_coords = list(node.neighbor_set)
        except Exception:
            neighbor_coords = [agent.location.copy()]

        cur = np.asarray(agent.location, dtype=float)
        best = cur.copy()
        best_d = np.linalg.norm(cur - np.asarray(target_xy, dtype=float)) - 1e-9  # 需要“严格更近”
        for nb in neighbor_coords:
            nb = np.asarray(nb, dtype=float)
            d = np.linalg.norm(nb - np.asarray(target_xy, dtype=float))
            if d < best_d:   # 严格更近
                best_d = d
                best = nb
        return np.asarray(best, dtype=float)


    def _forced_dstar_next_xy(self, aid, agent, t, global_map_info, belief_map, p_free, trav_mask, picks_raw_i):
        """
        强制导航到 rendezvous 区域（图对齐版）：
        1) D* 在栅格上解；2) 取下一格；3) snap 到 Node 图的邻居，必须严格更近；
        4) 若仍是原地，朝合同中心再 snap 一次；5) 仍不行 -> 回退探索步（避免站桩）。
        """
        # 已在圈内：圈内巡航，不离开区域
        if self.contract is not None and self.contract.within_region(agent.location):
            return self._in_zone_patrol_step(aid, agent, global_map_info)

        # 计算圈内最近可达目标格
        goal_rc = self._nearest_reachable_in_region(self.contract.P, self.contract.r, trav_mask, global_map_info)
        if goal_rc is None:
            return picks_raw_i   # 本步放弃强制，避免站桩

        # 代价图（见下方 _build_cost_map，建议 UNKNOWN 视为不可走，保证与 Node 图一致）
        cost_map = self._build_cost_map(belief_map, p_free)
        start_rc = self._world_to_cell_rc(agent.location, global_map_info)

        # 规划器初始化 / 复用
        if self._planners[aid] is None or self._planner_goal[aid] != tuple(goal_rc):
            self._planners[aid] = DStarLite(cost_map, start_rc, goal_rc)
            self._planner_goal[aid] = tuple(goal_rc)
        else:
            self._planners[aid].update_start(start_rc)
            if (t % RDV_COSTMAP_UPDATE_EVERY) == 0:
                self._planners[aid].update_cost_map(cost_map, changed_cells=None)

        # 预算
        T_tar = int(self.contract.meta.get('T_tar', t + 1))
        budget = max(1, T_tar - t)

        # D* 最短路
        self._planners[aid].compute_shortest_path(max_expand=RDV_DSTAR_MAX_EXPAND)
        path = self._planners[aid].extract_path(max_len=budget + 5)

        # 路径退化：硬重建 + BFS 兜底
        if len(path) <= 1:
            self._planners[aid] = DStarLite(cost_map, start_rc, goal_rc)
            self._planner_goal[aid] = tuple(goal_rc)
            self._planners[aid].compute_shortest_path(max_expand=RDV_DSTAR_MAX_EXPAND)
            path = self._planners[aid].extract_path(max_len=256)
            if len(path) <= 1:
                bfs_path = self._bfs_path_rc(trav_mask, start_rc, goal_rc)
                if bfs_path and len(bfs_path) > 1:
                    nxt_rc = bfs_path[1]
                else:
                    return picks_raw_i
        else:
            nxt_rc = path[1]

        # rc -> world（D* 的下一格）
        nxt_xy_grid = np.array([
            global_map_info.map_origin_x + nxt_rc[1] * global_map_info.cell_size,
            global_map_info.map_origin_y + nxt_rc[0] * global_map_info.cell_size
        ], dtype=float)

        # 关键：把“栅格下一格”snap 到 Node 图的一个邻居（必须严格更近）
        snap1 = self._pick_graph_neighbor_toward(agent, nxt_xy_grid, global_map_info)

        # 若仍是原地，再朝合同中心 P snap 一次
        if np.allclose(snap1, agent.location):
            snap2 = self._pick_graph_neighbor_toward(agent, self.contract.P, global_map_info)
            if not np.allclose(snap2, agent.location):
                return snap2
            else:
                # 仍然没有一个严格更近的邻居（可能在死胡同里需要探索开路）→ 回退探索步
                return picks_raw_i

        return snap1


    # ========================= 可视化 =========================
    def plot_env(self, step):
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 10), dpi=110)

        gs = GridSpec(N_AGENTS, 3, figure=fig, width_ratios=[2.5, 1.2, 1.2], wspace=0.15, hspace=0.1)
        ax_global = fig.add_subplot(gs[:, 0])
        ax_locals_obs = [fig.add_subplot(gs[i, 1]) for i in range(N_AGENTS)]
        ax_locals_pred = [fig.add_subplot(gs[i, 2]) for i in range(N_AGENTS)]
        agent_colors = plt.cm.get_cmap('cool', N_AGENTS)

        global_info = MapInfo(self.env.global_belief, self.env.belief_origin_x, self.env.belief_origin_y, self.env.cell_size)
        ax_global.set_title(f"Global View | Step {step}/{MAX_EPISODE_STEP}", fontsize=14, pad=10)
        ax_global.imshow(global_info.map, cmap='gray', origin='lower')
        ax_global.set_aspect('equal', adjustable='box')
        ax_global.set_axis_off()

        if self.robots and self.robots[0].pred_mean_map_info is not None:
            pred_mean = self.robots[0].pred_mean_map_info.map.astype(np.float32) / 255.0
            belief = global_info.map
            unknown_mask = (belief == UNKNOWN)
            prob = np.zeros_like(pred_mean)
            prob[unknown_mask] = pred_mean[unknown_mask]
            ax_global.imshow(prob, cmap='magma', origin='lower', alpha=0.35)

        # 通信边
        groups = self._compute_groups_from_positions(self.env.robot_locations)
        for group in groups:
            for i_idx, i in enumerate(list(group)):
                for j in list(group)[i_idx+1:]:
                    p1 = self._world_to_cell_rc(self.robots[i].location, global_info)
                    p2 = self._world_to_cell_rc(self.robots[j].location, global_info)
                    ax_global.plot([p1[1], p2[1]], [p1[0], p2[0]], color="#33ff88", lw=2, alpha=0.8, zorder=5)

        # 机器人、轨迹、通讯圈
        for i, r in enumerate(self.robots):
            pos_cell = self._world_to_cell_rc(r.location, global_info)
            if hasattr(r, 'trajectory_x') and r.trajectory_x:
                traj_cells = [self._world_to_cell_rc(np.array([x, y]), global_info)
                              for x, y in zip(r.trajectory_x, r.trajectory_y)]
                ax_global.plot([c for rr, c in traj_cells], [rr for rr, c in traj_cells],
                               color=agent_colors(i), lw=1.5, zorder=3)
            comms_radius = patches.Circle((pos_cell[1], pos_cell[0]), COMMS_RANGE / CELL_SIZE,
                                          fc=(0, 1, 0, 0.05), ec=(0, 1, 0, 0.4), ls='--', lw=1.5, zorder=4)
            ax_global.add_patch(comms_radius)
            ax_global.plot(pos_cell[1], pos_cell[0], 'o', ms=10, mfc=agent_colors(i),
                           mec='white', mew=1.5, zorder=10)

        # 合同可视化
        if self.contract is not None:
            # 画主点/备选点
            for k, Pk in enumerate(self.contract.P_list):
                p_cell = self._world_to_cell_rc(Pk, global_info)
                if k == self.contract.target_idx and self.contract.status == 'active':
                    ax_global.plot(p_cell[1], p_cell[0], '*', ms=22, mfc='yellow', mec='white', mew=2, zorder=12)
                else:
                    ax_global.plot(p_cell[1], p_cell[0], '+', ms=20, c=('yellow' if k==0 else 'orange'), mew=2, zorder=11)
                radius = patches.Circle((p_cell[1], p_cell[0]), self.contract.r / CELL_SIZE,
                                        fc=(1, 1, 0, 0.07 if k==self.contract.target_idx else 0.04),
                                        ec=('yellow' if k==0 else 'orange'),
                                        ls='--', lw=1.8, zorder=11)
                ax_global.add_patch(radius)

            if self.contract.status == 'active':
                ax_global.text(5, 5,
                               f"RDV(active) target={self.contract.target_idx} "
                               f"T_tar: {self.contract.meta.get('T_tar','-')}",
                               fontsize=10, color='yellow', ha='left', va='top')
            elif self.contract.status == 'armed':
                ax_global.text(5, 5, f"RDV(armed) waiting for disconnect",
                               fontsize=10, color='yellow', ha='left', va='top')

        # 本地视角
        for i, r in enumerate(self.robots):
            ax_obs = ax_locals_obs[i]
            local_map_info = r.map_info
            ax_obs.set_title(f"Agent {i} View", fontsize=10, pad=5)
            ax_obs.imshow(local_map_info.map, cmap='gray', origin='lower')
            ax_obs.set_aspect('equal', adjustable='box')
            pos_cell_local = self._world_to_cell_rc(r.location, local_map_info)
            ax_obs.plot(pos_cell_local[1], pos_cell_local[0], 'o', ms=8, mfc=agent_colors(i),
                        mec='white', mew=1.5, zorder=10)
            if r.intent_seq:
                intent_world = [r.location] + r.intent_seq
                intent_cells = [self._world_to_cell_rc(pos, local_map_info) for pos in intent_world]
                ax_obs.plot([c for rr, c in intent_cells], [rr for rr, c in intent_cells],
                            'x--', c=agent_colors(i), lw=2, ms=6, zorder=8)
            ax_obs.set_axis_off()

            ax_pred = ax_locals_pred[i]
            ax_pred.set_title(f"Agent {i} Predicted (local)", fontsize=10, pad=5)
            ax_pred.set_aspect('equal', adjustable='box')
            ax_pred.set_axis_off()

            try:
                if r.pred_mean_map_info is not None or r.pred_max_map_info is not None:
                    pred_info = r.pred_mean_map_info if r.pred_mean_map_info is not None else r.pred_max_map_info
                    pred_local = r.get_updating_map(r.location, base=pred_info)
                    belief_local = r.get_updating_map(r.location, base=r.map_info)

                    ax_pred.imshow(pred_local.map, cmap='gray', origin='lower', vmin=0, vmax=255)
                    alpha_mask = (belief_local.map == FREE) * 0.45
                    ax_pred.imshow(belief_local.map, cmap='Blues', origin='lower', alpha=alpha_mask)

                    rc = get_cell_position_from_coords(r.location, pred_local)
                    ax_pred.plot(rc[0], rc[1], 'mo', markersize=8, zorder=6)
                else:
                    ax_pred.text(0.5, 0.5, 'No prediction', ha='center', va='center', fontsize=9)
            except Exception as e:
                ax_pred.text(0.5, 0.5, f'Pred plot err:\n{e}', ha='center', va='center', fontsize=8)

        out_path = os.path.join(self.run_dir, f"t{step:04d}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        self.env.frame_files.append(out_path)
