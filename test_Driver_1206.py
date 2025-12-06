import ray
import torch
import yaml

from .model import PolicyNet
from .env import Env
from .agent import Agent
from .ground_truth_node_manager import GroundTruthNodeManager
from .utils import *
from .parameter import *
from .sensor import sensor_work
from mapinpaint.networks import Generator
from mapinpaint.evaluator import Evaluator

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

# Other configuration settings in parameter.py
NUM_TEST = 100
NUM_META_AGENT = 4  # number of parallel tests, NUM_TEST % NUM_META_AGENT should be 0
SAFE_MODE = True
SAVE_GIFS = True

# ===== Multi-robot settings =====
NUM_ROBOTS = 3          # 每个环境里的机器人数量
COMM_RANGE = 32.0       # 通信半径（米），可以先设成 SENSOR_RANGE 或 2*SENSOR_RANGE

if SAVE_GIFS:
    os.makedirs(gif_path, exist_ok=True)


def run_test():
    device = torch.device('cpu')
    global_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)

    print(f"Testing on {device}, model: {model_path}, num of tests: {NUM_TEST}, num of samples: {N_GEN_SAMPLE}")
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(f'{model_path}/checkpoint.pth', weights_only=True, map_location=device)

    global_network.load_state_dict(checkpoint['policy_model'])

    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    weights = global_network.state_dict()
    curr_test = 0

    travel_dist = []
    explored_rate = []
    success_rate = []
    sr_room = []
    sr_tunnel = []
    sr_outdoor = []
    td_room = []
    td_tunnel = []
    td_outdoor = []

    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(weights, curr_test))
        curr_test += 1

    try:
        while len(travel_dist) < curr_test:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                metrics, info = job
                travel_dist.append(metrics['travel_dist'])
                explored_rate.append(metrics['explored_rate'])
                success_rate.append(metrics['success_rate'])
                if 'room' in info['map_path']:
                    sr_room.append(metrics['success_rate'])
                    td_room.append(metrics['travel_dist'])
                elif 'tunnel' in info['map_path']:
                    sr_tunnel.append(metrics['success_rate'])
                    td_tunnel.append(metrics['travel_dist'])
                elif 'outdoor' in info['map_path']:
                    sr_outdoor.append(metrics['success_rate'])
                    td_outdoor.append(metrics['travel_dist'])

                if curr_test < NUM_TEST:
                    job_list.append(meta_agents[info['id']].job.remote(weights, curr_test))
                    curr_test += 1

        print('=====================================')
        print('| Test:', FOLDER_NAME)
        print('| Total test: {} with {} predictions'.format(NUM_TEST, N_GEN_SAMPLE))
        print('| Average success rate:', np.array(success_rate).mean())
        print('| Average travel distance:', np.array(travel_dist).mean())
        print('| Average explored rate:', np.array(explored_rate).mean())
        print('| Room success rate: {}, travel distance: {:.2f} ± {:.2f}'.format(
            np.mean(sr_room) if len(sr_room) > 0 else np.nan,
            np.mean(td_room) if len(td_room) > 0 else np.nan,
            np.std(td_room) if len(td_room) > 0 else np.nan))
        print('| Tunnel success rate: {}, travel distance: {:.2f} ± {:.2f}'.format(
            np.mean(sr_tunnel) if len(sr_tunnel) > 0 else np.nan,
            np.mean(td_tunnel) if len(td_tunnel) > 0 else np.nan,
            np.std(td_tunnel) if len(td_tunnel) > 0 else np.nan))
        print('| Outdoor success rate: {}, travel distance: {:.2f} ± {:.2f}'.format(
            np.mean(sr_outdoor) if len(sr_outdoor) > 0 else np.nan,
            np.mean(td_outdoor) if len(td_outdoor) > 0 else np.nan,
            np.std(td_outdoor) if len(td_outdoor) > 0 else np.nan))

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


class TestWorker:
    """
    原始单机器人评估 worker（保持不变）
    """
    def __init__(self, meta_agent_id, policy_net, predictor, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image, test=True)

        self.robot = Agent(policy_net, predictor, self.device, self.save_image)

        self.ground_truth_node_manager = GroundTruthNodeManager(
            self.robot.node_manager, self.env.ground_truth_info,
            device=self.device, plot=self.save_image
        )

        self.perf_metrics = dict()
        self.location_history = []
        self.loop_detected = False
        self.path_to_nearest_frontier = None

    def run_episode(self):
        done = False
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        observation = self.robot.get_observation()
        self.ground_truth_node_manager.get_ground_truth_observation(
            self.env.robot_location, self.robot.pred_mean_map_info
        )

        if self.save_image:
            self.robot.plot_env()
            self.robot.pred_node_manager.plot_predicted_env(self.env.robot_location, self.robot.map_info.map)
            self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location)
            self.env.plot_env(0)

        for i in range(MAX_EPISODE_STEP):
            self.location_history.append(self.env.robot_location)
            if SAFE_MODE and len(self.location_history) >= 4:
                if (np.array_equal(self.location_history[-1], self.location_history[-3]) and
                        np.array_equal(self.location_history[-2], self.location_history[-4])):
                    print("Loop detected, go to nearest frontier")
                    self.loop_detected = True
                    self.path_to_nearest_frontier = self.robot.pred_node_manager.path_to_nearest_frontier.copy()
            if self.loop_detected:
                next_location = np.array(self.path_to_nearest_frontier.pop(0))
                if len(self.path_to_nearest_frontier) == 0:
                    self.loop_detected = False
                node_exist = self.robot.node_manager.nodes_dict.find((next_location[0], next_location[1]))
                if node_exist is None:
                    next_location, action_index = self.robot.select_next_waypoint(observation, greedy=True)
                    self.loop_detected = False
            else:
                next_location, action_index = self.robot.select_next_waypoint(observation, greedy=True)

            self.env.step(next_location)
            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
            if not (self.robot.utility > 0).any() or self.env.explored_rate > 0.9999:
                done = True
            observation = self.robot.get_observation()
            self.ground_truth_node_manager.get_ground_truth_observation(
                self.env.robot_location, self.robot.pred_mean_map_info
            )

            if self.save_image:
                self.robot.plot_env()
                self.robot.pred_node_manager.plot_predicted_env(self.env.robot_location, self.robot.map_info.map)
                self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location)
                self.env.plot_env(i + 1)

            if done:
                break

        if not done:
            print(f"Exploration not completed in env: {self.env.map_path}")

        # save metrics
        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        # save gif
        if self.save_image:
            make_gif(gif_path, self.global_step, self.env.frame_files, self.env.explored_rate)

class MultiTestWorker:
    """
    多机器人评估：
    - 复用单机器人 Env / Agent / GroundTruthNodeManager
    - 每个 robot 有自己的 Env & belief，但 ground_truth 是同一张地图
    - 通信半径 COMM_RANGE 内进行 belief 合并
    - global planner 在预测地图节点上做区域划分（不动机器人的位置）
    - local planner 只在自己区域内的节点上保留 utility，降低重叠探索
    """
    def __init__(self, meta_agent_id, policy_net, predictor, global_step,
                 num_robots=2, comm_range=16.0,
                 device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.num_robots = num_robots
        self.comm_range = comm_range

        # === 1. 创建 base_env & 多个 env（共享同一张地图） ===
        base_env = Env(global_step, plot=save_image, test=True)
        base_start = base_env.robot_location.copy()

        # 在整张地图上生成节点网格（按 NODE_RESOLUTION），只取与起点连通的 free 区域
        node_coords, _ = get_updating_node_coords(
            base_start,
            base_env.ground_truth_info,
            check_connectivity=True
        )  # shape: (N_nodes, 2)

        # 选择 robot0 的起点：离原始 base_start 最近的一个节点
        dists_to_base = np.linalg.norm(node_coords - base_start, axis=1)
        idx0 = np.argmin(dists_to_base)
        start_nodes = [node_coords[idx0]]

        rng = np.random.default_rng(seed=global_step + meta_agent_id)

        # 其他机器人在 robot0 附近“稍微散开一点”
        r_min = 0.5 * NODE_RESOLUTION   # 离 robot0 至少这么远
        r_max = 2.0 * NODE_RESOLUTION   # 最多离这么远
        sep_min = 0.5 * NODE_RESOLUTION # 机器人之间最小间距

        for k in range(1, num_robots):
            chosen = None
            for _ in range(2000):
                cand = node_coords[rng.integers(len(node_coords))]
                d0 = np.linalg.norm(cand - start_nodes[0])  # 距离 robot0
                if not (r_min <= d0 <= r_max):
                    continue
                # 和已选起点保持一定间隔
                if any(np.linalg.norm(cand - p) <= sep_min for p in start_nodes):
                    continue
                chosen = cand
                break
            if chosen is None:
                # 实在选不到，就退一步：再选一次离 robot0 最近的
                chosen = node_coords[np.argmin(np.linalg.norm(node_coords - start_nodes[0], axis=1))]
            start_nodes.append(chosen)

        # === 2. 根据 start_nodes 创建每个 env ===
        self.envs = []
        for k in range(num_robots):
            if k == 0:
                env = base_env
            else:
                env = Env(global_step, plot=save_image, test=True)
                # 共享同一张 ground truth 和 map_path
                env.ground_truth = base_env.ground_truth
                env.map_path = base_env.map_path

            # 统一 origin，保证世界坐标一致
            env.belief_origin_x = base_env.belief_origin_x
            env.belief_origin_y = base_env.belief_origin_y
            env.ground_truth_info = base_env.ground_truth_info

            # 起点用我们选好的节点（世界坐标 = node 坐标）
            env.robot_location = start_nodes[k]
            env.robot_cell = get_cell_position_from_coords(
                env.robot_location,
                base_env.ground_truth_info
            )

            # belief 全 UNKNOWN，然后用 sensor_work 做一次初始观测
            env.robot_belief = np.ones_like(base_env.ground_truth, dtype=np.uint8) * UNKNOWN
            env.robot_belief = sensor_work(
                env.robot_cell,
                env.sensor_range / env.cell_size,
                env.robot_belief,
                env.ground_truth
            )
            env.belief_info = MapInfo(
                env.robot_belief,
                env.belief_origin_x,
                env.belief_origin_y,
                env.cell_size
            )

            if save_image:
                env.trajectory_x = [env.robot_location[0]]
                env.trajectory_y = [env.robot_location[1]]

            self.envs.append(env)

        # 兼容 run_test 里 self.worker.env.map_path 的用法
        self.env = self.envs[0]

        # === 3. 为每个机器人创建 Agent & GroundTruthNodeManager ===
        self.robots = [
            Agent(policy_net, predictor, self.device, self.save_image)
            for _ in range(num_robots)
        ]
        self.ground_truth_node_managers = [
            GroundTruthNodeManager(
                self.robots[i].node_manager,
                self.envs[i].ground_truth_info,
                device=self.device,
                plot=self.save_image
            )
            for i in range(num_robots)
        ]

        # 每个机器人的历史轨迹
        self.location_histories = [[] for _ in range(num_robots)]

        # 指标
        self.perf_metrics = dict()
        self.explored_rate_global = 0.0

        # global planner 结果：每个机器人负责的节点集合 & rendezvous 节点
        self.region_assignments = None   # list[set[(x,y)]]
        self.rendezvous_points = None    # list[np.array or None]

        # GIF 帧
        self.frame_files = [] if self.save_image else None

    # --------- 基础工具函数：保持不变 ---------
    def _update_planning_states(self):
        """根据当前 belief 和位置，更新所有机器人的规划状态"""
        for i in range(self.num_robots):
            env_i = self.envs[i]
            robot_i = self.robots[i]
            env_i.belief_info.map = env_i.robot_belief.astype(np.uint8)
            robot_i.update_planning_state(env_i.belief_info, env_i.robot_location)

    def _apply_communication(self):
        """通信范围内的机器人之间合并 belief"""
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                pi = self.envs[i].robot_location
                pj = self.envs[j].robot_location
                dist = np.linalg.norm(pi - pj)
                if dist <= self.comm_range:
                    bi = self.envs[i].robot_belief
                    bj = self.envs[j].robot_belief

                    merged = np.ones_like(bi) * UNKNOWN

                    occ = (bi == OCCUPIED) | (bj == OCCUPIED)
                    free = (bi == FREE) | (bj == FREE)

                    merged[occ] = OCCUPIED
                    merged[~occ & free] = FREE

                    self.envs[i].robot_belief = merged.copy()
                    self.envs[j].robot_belief = merged.copy()
                    self.envs[i].belief_info.map = self.envs[i].robot_belief
                    self.envs[j].belief_info.map = self.envs[j].robot_belief

                    self.envs[i].evaluate_exploration_rate()
                    self.envs[j].evaluate_exploration_rate()

    def _evaluate_global_explored_rate(self):
        """全局覆盖率 = 各机器人 free cells 的并集 / ground truth free cells"""
        gt = self.envs[0].ground_truth
        gt_free = (gt == FREE)
        union_free = np.zeros_like(gt_free, dtype=bool)
        for env in self.envs:
            union_free |= (env.robot_belief == FREE)
        explored = np.sum(union_free & gt_free)
        total_free = np.sum(gt_free)
        self.explored_rate_global = explored / (total_free + 1e-8)
        return self.explored_rate_global

    # --------- 新增：global planner，在预测地图节点上划分区域 ---------
    def _global_plan_regions_and_rendezvous(self):
        """
        基于当前预测地图做一次全局区域划分：
        - 使用机器人 0 的预测图作为“全局”近似
        - 从其中选出高置信度 free 节点
        - 通过最近机器人分配节点 → 每个机器人得到一个节点集合
        - 每个机器人再从自己节点集合里选一个质心附近的 rendezvous 节点
        """
        # 依赖于 pred_node_manager 已经初始化（即至少跑过一次 get_observation）
        ref_robot = self.robots[0]
        pred_mgr = getattr(ref_robot, "pred_node_manager", None)
        if pred_mgr is None or getattr(pred_mgr, "ground_truth_node_coords", None) is None:
            # 预测图还没准备好，先给个空分配
            self.region_assignments = [set() for _ in range(self.num_robots)]
            self.rendezvous_points = [None for _ in range(self.num_robots)]
            return

        all_nodes = np.array(pred_mgr.ground_truth_node_coords).reshape(-1, 2)  # (N,2)
        pred_prob = np.array(pred_mgr.pred_prob).reshape(-1)                    # (N,)

        if all_nodes.shape[0] == 0:
            self.region_assignments = [set() for _ in range(self.num_robots)]
            self.rendezvous_points = [None for _ in range(self.num_robots)]
            return

        # 节点概率归一化到 [0,1]，选出高置信度的 free 节点
        prob_norm = pred_prob / float(FREE)
        mask = prob_norm >= 0.6
        if not mask.any():
            # 如果全都不太确定，就先全部拿来分配
            mask[:] = True

        candidate_nodes = all_nodes[mask]  # (M,2)
        if candidate_nodes.shape[0] == 0:
            self.region_assignments = [set() for _ in range(self.num_robots)]
            self.rendezvous_points = [None for _ in range(self.num_robots)]
            return

        # 机器人当前位姿（world frame）
        robot_positions = np.stack([env.robot_location for env in self.envs], axis=0)  # (R,2)

        # 为每个候选节点分配最近的机器人
        region_assignments = [set() for _ in range(self.num_robots)]
        for coord in candidate_nodes:
            dists = np.linalg.norm(robot_positions - coord[None, :], axis=1)
            ridx = int(np.argmin(dists))
            region_assignments[ridx].add((float(coord[0]), float(coord[1])))

        self.region_assignments = region_assignments

        # 为每个机器人选一个 rendezvous 点（区域质心附近的某个节点）
        rendezvous_points = []
        for r in range(self.num_robots):
            nodes_r = list(region_assignments[r])
            if len(nodes_r) == 0:
                rendezvous_points.append(None)
                continue
            nodes_r = np.array(nodes_r).reshape(-1, 2)
            centroid = nodes_r.mean(axis=0)
            idx = int(np.argmin(np.linalg.norm(nodes_r - centroid[None, :], axis=1)))
            rendezvous_points.append(nodes_r[idx])

        self.rendezvous_points = rendezvous_points

    def _apply_region_mask_to_observation(self, robot_idx, observation):
        """
        根据 global planner 的区域划分，对 observation 里的 utility 做 mask：
        - 不属于该机器人区域的节点 utility 置为 0
        - 区域内节点保持原样
        注意：只动 observation，不改 node graph 拓扑
        """
        if self.region_assignments is None:
            return observation

        assigned = self.region_assignments[robot_idx]
        if assigned is None or len(assigned) == 0:
            return observation  # 暂时没有区域限制

        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation

        # Agent 在 get_observation 里已经缓存了真实世界坐标的 node_coords
        coords = getattr(self.robots[robot_idx], "node_coords", None)
        if coords is None:
            return observation

        coords = np.array(coords).reshape(-1, 2)
        n_node = coords.shape[0]
        assigned_set = set(assigned)

        # 遍历前 n_node 个节点（后面是 padding）
        for j in range(n_node):
            c = tuple(float(x) for x in coords[j])
            if c not in assigned_set:
                # node_inputs 维度: [1, NODE_PADDING_SIZE, 6]
                # [.., 0:2] 是归一化坐标，[.., 2] 是 utility
                node_inputs[0, j, 2] = 0.0

        # 同步 Agent 内部 utility（可选）
        if hasattr(self.robots[robot_idx], "utility") and self.robots[robot_idx].utility is not None:
            util = np.array(self.robots[robot_idx].utility).copy()
            for j in range(n_node):
                c = tuple(float(x) for x in coords[j])
                if c not in assigned_set:
                    util[j] = 0
            self.robots[robot_idx].utility = util

        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask]

    # --------- 主循环 ---------
    def run_episode(self):
        done = False

        # 0. 先初始化每个机器人的图 / 预测地图 / 节点观测
        self._update_planning_states()
        for i in range(self.num_robots):
            _ = self.robots[i].get_observation()
            self.ground_truth_node_managers[i].get_ground_truth_observation(
                self.envs[i].robot_location,
                self.robots[i].pred_mean_map_info
            )

        # 1. 做一次全局区域划分（后面你可以在若干步后重新调这个函数）
        self._global_plan_regions_and_rendezvous()

        if self.save_image:
            self.plot_multi_env(step=0)

        # 2. 迭代执行
        for step in range(MAX_EPISODE_STEP):
            next_locations = []

            # 每个机器人基于自己的 observation + 区域 mask 来选下一节点
            for i in range(self.num_robots):
                env_i = self.envs[i]
                robot_i = self.robots[i]

                self.location_histories[i].append(env_i.robot_location)

                obs_i = robot_i.get_observation()
                obs_i = self._apply_region_mask_to_observation(i, obs_i)
                next_loc_i, _ = robot_i.select_next_waypoint(obs_i, greedy=True)
                next_locations.append(next_loc_i)

            # 冲突消解：禁止 vertex / edge swap 冲突
            next_locations = self._resolve_conflicts(next_locations)

            # 同步执行一步
            for i in range(self.num_robots):
                self.envs[i].step(next_locations[i])

            # 通信范围内合并 belief
            self._apply_communication()

            # belief 更新后，重新更新规划状态
            self._update_planning_states()

            # 更新全局覆盖率 & 终止条件
            explored_global = self._evaluate_global_explored_rate()

            no_utility = True
            for i in range(self.num_robots):
                _ = self.robots[i].get_observation()
                if (self.robots[i].utility > 0).any():
                    no_utility = False
                    break

            if explored_global > 0.999 or no_utility:
                done = True

            if self.save_image:
                self.plot_multi_env(step=step + 1)

            if done:
                break

        if not done:
            print(f"[Multi] Exploration not completed in env: {self.env.map_path}")

        # 记录指标
        total_dist = sum(env.travel_dist for env in self.envs)
        self.perf_metrics['travel_dist'] = total_dist
        self.perf_metrics['explored_rate'] = self.explored_rate_global
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['travel_dist_each'] = [env.travel_dist for env in self.envs]

        # 如果要保存 gif
        if self.save_image and self.frame_files:
            make_gif(gif_path, self.global_step, self.frame_files, self.explored_rate_global)

    # --------- 冲突消解（沿用你之前的版本） ---------
    def _resolve_conflicts(self, next_locations):
        """
        简单冲突消解：
        - vertex conflict：两个或以上机器人目标 cell 相同 → 只有编号更小的机器人前进，其余原地不动
        - swap conflict：i: A->B, j: B->A → 只让编号更小的机器人前进，另一个原地
        输入、输出都是 world 坐标 list[np.array]
        """
        gt_info = self.envs[0].ground_truth_info

        curr_cells = [
            tuple(get_cell_position_from_coords(env.robot_location, gt_info))
            for env in self.envs
        ]
        next_cells = [
            tuple(get_cell_position_from_coords(pos, gt_info))
            for pos in next_locations
        ]

        # 1) vertex conflict
        occupied = {}
        for i in range(self.num_robots):
            n_cell = next_cells[i]
            c_cell = curr_cells[i]

            if n_cell in occupied:
                next_cells[i] = c_cell
                next_locations[i] = self.envs[i].robot_location.copy()
            else:
                occupied[n_cell] = i

        # 2) edge swap conflict
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                ci, cj = curr_cells[i], curr_cells[j]
                ni, nj = next_cells[i], next_cells[j]

                if ci == nj and cj == ni and ni != ci:
                    loser = j
                    next_cells[loser] = curr_cells[loser]
                    next_locations[loser] = self.envs[loser].robot_location.copy()

        return next_locations

    '''# --------- 多机器人可视化（保持你之前的风格） ---------
    def plot_multi_env(self, step):
        if not self.save_image:
            return

        plt.switch_backend('agg')
        plt.figure(figsize=(6, 6))

        gt = self.envs[0].ground_truth
        gt_free = (gt == FREE)
        gt_occ = (gt == OCCUPIED)

        union_free = np.zeros_like(gt_free, dtype=bool)
        for env in self.envs:
            union_free |= (env.robot_belief == FREE)

        vis_map = np.ones_like(gt, dtype=np.uint8) * 127
        vis_map[gt_occ] = 0
        vis_map[union_free] = 255

        plt.imshow(vis_map, cmap='gray', origin='upper')
        ax = plt.gca()

        colors = ['r', 'g', 'b', 'c', 'm', 'y']

        for i, env in enumerate(self.envs):
            x = (env.robot_location[0] - env.belief_origin_x) / env.cell_size
            y = (env.robot_location[1] - env.belief_origin_y) / env.cell_size

            color = colors[i % len(colors)]
            ax.plot(x, y, marker='o', markersize=6, linestyle='None',
                    color=color, label=f'R{i}')

            radius_cells = self.comm_range / env.cell_size
            circle = plt.Circle((x, y), radius_cells, fill=False,
                                linestyle=':', linewidth=1, alpha=0.3,
                                edgecolor=color)
            ax.add_patch(circle)

        # 通信边
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                pi = self.envs[i].robot_location
                pj = self.envs[j].robot_location
                dist = np.linalg.norm(pi - pj)
                if dist <= self.comm_range:
                    xi = (pi[0] - self.envs[i].belief_origin_x) / self.envs[i].cell_size
                    yi = (pi[1] - self.envs[i].belief_origin_y) / self.envs[i].cell_size
                    xj = (pj[0] - self.envs[j].belief_origin_x) / self.envs[j].cell_size
                    yj = (pj[1] - self.envs[j].belief_origin_y) / self.envs[j].cell_size

                    ax.plot([xi, xj], [yi, yj],
                            linestyle='--', linewidth=1, alpha=0.5, color='yellow')

        ax.set_title(f'Global explored: {self.explored_rate_global:.3f}')
        ax.set_axis_off()
        ax.legend(loc='lower right', fontsize=8)

        frame_path = f'{gif_path}/{self.global_step}_{step}_multi.png'
        plt.tight_layout()
        plt.savefig(frame_path, dpi=150)
        plt.close()

        self.frame_files.append(frame_path)'''
    '''def plot_multi_env(self, step):
        """
        更强可视化版本：
        - 背景：ground truth 障碍 + 多机器人 FREE 并集（灰度）
        - 区域划分：按 global planner 分配，把节点所在的 cell 直接染色（块状区域）
        - rendezvous：同色大星星
        - 机器人：同色粗点 + 编号
        - 通信范围：圆
        - 通信边：虚线
        """
        if not self.save_image:
            return

        plt.switch_backend('agg')
        plt.figure(figsize=(7, 7))

        # ---------- 1. 背景地图 ----------
        gt = self.envs[0].ground_truth
        gt_free = (gt == FREE)
        gt_occ = (gt == OCCUPIED)

        union_free = np.zeros_like(gt_free, dtype=bool)
        for env in self.envs:
            union_free |= (env.robot_belief == FREE)

        # 0=障碍(黑), 127=未知(灰), 255=已探索free(白)
        vis_map = np.ones_like(gt, dtype=np.uint8) * 127
        vis_map[gt_occ] = 0
        vis_map[union_free] = 255

        ax = plt.gca()
        ax.imshow(vis_map, cmap='gray', origin='upper')

        H, W = gt.shape
        env0 = self.envs[0]

        # 用 tab10 给不同机器人/区域配色
        cmap_regions = plt.get_cmap('tab10', self.num_robots)

        # ---------- 2. 区域染色 ----------
        if getattr(self, "region_assignments", None) is not None:
            # region_rgba: (H, W, 4)，初始全透明
            region_rgba = np.zeros((H, W, 4), dtype=float)

            for ridx, node_set in enumerate(self.region_assignments):
                if not node_set:
                    continue

                base_color = cmap_regions(ridx)  # (r,g,b,a)
                rgb = base_color[:3]

                for (wx, wy) in node_set:
                    cell = get_cell_position_from_coords(
                        np.array([wx, wy]),
                        env0.ground_truth_info,
                        check_negative=False  # 我们自己做边界检查
                    )
                    cx, cy = int(cell[0]), int(cell[1])
                    if 0 <= cy < H and 0 <= cx < W:
                        region_rgba[cy, cx, :3] = rgb
                        region_rgba[cy, cx, 3] = 0.7  # 区域透明度，颜色比较“实”

            ax.imshow(region_rgba, origin='upper')

        # ---------- 3. rendezvous 显示 ----------
        if getattr(self, "rendezvous_points", None) is not None:
            for ridx, rv in enumerate(self.rendezvous_points):
                if rv is None:
                    continue
                rv = np.array(rv)
                gx = (rv[0] - env0.belief_origin_x) / env0.cell_size
                gy = (rv[1] - env0.belief_origin_y) / env0.cell_size

                color = cmap_regions(ridx)
                ax.scatter(
                    [gx], [gy],
                    s=150,
                    marker='*',
                    facecolors=color,
                    edgecolors='k',
                    linewidths=1.0,
                    zorder=5,
                )
                ax.text(
                    gx + 2, gy + 2,
                    f'Rv{ridx}',
                    color='k',
                    fontsize=7,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                    zorder=6,
                )

        # ---------- 4. 机器人位置 + 通信圆 ----------
        handles = []
        labels = []
        for i, env in enumerate(self.envs):
            x = (env.robot_location[0] - env.belief_origin_x) / env.cell_size
            y = (env.robot_location[1] - env.belief_origin_y) / env.cell_size

            color = cmap_regions(i)
            h = ax.plot(
                x, y,
                marker='o',
                markersize=8,
                linestyle='None',
                markeredgecolor='k',
                markeredgewidth=1.0,
                color=color,
                label=f'R{i}',
                zorder=7,
            )[0]
            handles.append(h)
            labels.append(f'R{i}')

            radius_cells = self.comm_range / env.cell_size
            circle = plt.Circle(
                (x, y),
                radius_cells,
                fill=False,
                linestyle=':',
                linewidth=1.0,
                alpha=0.5,
                edgecolor=color,
            )
            ax.add_patch(circle)

            # 在机器人旁边再写个 R0/R1...
            ax.text(
                x + 1, y + 1,
                f'R{i}',
                color='k',
                fontsize=7,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                zorder=8,
            )

        # ---------- 5. 通信边 ----------
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                pi = self.envs[i].robot_location
                pj = self.envs[j].robot_location
                dist = np.linalg.norm(pi - pj)
                if dist <= self.comm_range:
                    xi = (pi[0] - self.envs[i].belief_origin_x) / self.envs[i].cell_size
                    yi = (pi[1] - self.envs[i].belief_origin_y) / self.envs[i].cell_size
                    xj = (pj[0] - self.envs[j].belief_origin_x) / self.envs[j].cell_size
                    yj = (pj[1] - self.envs[j].belief_origin_y) / self.envs[j].cell_size

                    ax.plot(
                        [xi, xj], [yi, yj],
                        linestyle='--',
                        linewidth=1.2,
                        alpha=0.9,
                        color='yellow',
                        zorder=6,
                    )

        ax.set_title(f'Global explored: {self.explored_rate_global:.3f}')
        ax.set_axis_off()

        if handles:
            ax.legend(handles, labels, loc='lower right', fontsize=8)

        frame_path = f'{gif_path}/{self.global_step}_{step}_multi.png'
        plt.tight_layout()
        plt.savefig(frame_path, dpi=150)
        plt.close()

        self.frame_files.append(frame_path)'''
    def plot_multi_env(self, step):
        """
        可视化（带预测 + 分区）：

        上半部分（全局）：
        - 背景：真实障碍 + 所有机器人探索到的 FREE 并集
        - 分区：global planner 分配的区域，用半透明色块 + 边界线标出来
        - 机器人：同色粗点 + 编号
        - rendezvous：同色大星星 + 文本 Rv#
        - 通信约束：通信圆 + 黄虚线连线

        下半部分（每行一个机器人）：
        - 显示该机器人的预测地图 pred_mean_map_info.map（或退化到 belief）
        - 同色半透明色块标出自己的区域
        - 机器人位置 + rendezvous
        """
        if not self.save_image:
            return

        plt.switch_backend('agg')

        # ---------- 准备基础数据 ----------
        env0 = self.envs[0]
        gt = env0.ground_truth
        H, W = gt.shape

        # 真实 free / occ
        gt_free = (gt == FREE)
        gt_occ = (gt == OCCUPIED)

        # 所有机器人探索到的 FREE 并集
        union_free = np.zeros_like(gt_free, dtype=bool)
        for env in self.envs:
            union_free |= (env.robot_belief == FREE)

        # 背景灰度图：0=障碍, 127=未知, 255=已探索free
        vis_map = np.ones_like(gt, dtype=np.uint8) * 127
        vis_map[gt_occ] = 0
        vis_map[union_free] = 255

        # ---------- 计算每个机器人的区域 mask（在栅格上） ----------
        region_masks = [None] * self.num_robots
        if getattr(self, "region_assignments", None) is not None:
            for ridx, node_set in enumerate(self.region_assignments):
                if not node_set:
                    continue
                mask = np.zeros((H, W), dtype=bool)
                for (wx, wy) in node_set:
                    cell = get_cell_position_from_coords(
                        np.array([wx, wy]),
                        env0.ground_truth_info,
                        check_negative=False
                    )
                    cx, cy = int(cell[0]), int(cell[1])
                    if 0 <= cx < W and 0 <= cy < H:
                        # 小范围扩展一圈，让区域更“块”一点而不是单点
                        y0 = max(0, cy - 1)
                        y1 = min(H, cy + 2)
                        x0 = max(0, cx - 1)
                        x1 = min(W, cx + 2)
                        mask[y0:y1, x0:x1] = True
                region_masks[ridx] = mask

        # ---------- 画布布局：上面全局，下面每个机器人一行预测图 ----------
        n_rows = 1 + self.num_robots
        fig = plt.figure(figsize=(8, 3 + 2 * self.num_robots))
        gs = gridspec.GridSpec(
            n_rows, 1,
            height_ratios=[3] + [2] * self.num_robots
        )

        # colormap：用 tab10 给不同机器人上色
        cmap_regions = plt.get_cmap('tab10', self.num_robots)

        # ===================== 上半部分：全局视图 =====================
        ax_global = fig.add_subplot(gs[0, 0])
        ax_global.imshow(vis_map, cmap='gray', origin='upper')

        # --- 分区：色块 + 边界线 ---
        for ridx in range(self.num_robots):
            mask = region_masks[ridx]
            if mask is None:
                continue
            base_color = cmap_regions(ridx)
            rgb = base_color[:3]

            region_rgba = np.zeros((H, W, 4), dtype=float)
            region_rgba[mask, 0] = rgb[0]
            region_rgba[mask, 1] = rgb[1]
            region_rgba[mask, 2] = rgb[2]
            region_rgba[mask, 3] = 0.3  # 区域透明度

            ax_global.imshow(region_rgba, origin='upper')
            # 边界线
            ax_global.contour(
                mask.astype(float),
                levels=[0.5],
                colors=[rgb],
                linewidths=1.0,
                alpha=0.9
            )

        # --- rendezvous 点（全局视图） ---
        if getattr(self, "rendezvous_points", None) is not None:
            for ridx, rv in enumerate(self.rendezvous_points):
                if rv is None:
                    continue
                rv = np.array(rv)
                gx = (rv[0] - env0.belief_origin_x) / env0.cell_size
                gy = (rv[1] - env0.belief_origin_y) / env0.cell_size

                color = cmap_regions(ridx)
                ax_global.scatter(
                    [gx], [gy],
                    s=150,
                    marker='*',
                    facecolors=color,
                    edgecolors='k',
                    linewidths=1.0,
                    zorder=5,
                )
                ax_global.text(
                    gx + 2, gy + 2,
                    f'Rv{ridx}',
                    color='k',
                    fontsize=7,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                    zorder=6,
                )

        # --- 机器人位置 + 通信圆 + 通信边（全局视图） ---
        handles = []
        labels = []

        for i, env in enumerate(self.envs):
            color = cmap_regions(i)

            # 机器人世界坐标 -> 栅格坐标
            x = (env.robot_location[0] - env.belief_origin_x) / env.cell_size
            y = (env.robot_location[1] - env.belief_origin_y) / env.cell_size

            h = ax_global.plot(
                x, y,
                marker='o',
                markersize=8,
                linestyle='None',
                markeredgecolor='k',
                markeredgewidth=1.0,
                color=color,
                label=f'R{i}',
                zorder=7,
            )[0]
            handles.append(h)
            labels.append(f'R{i}')

            # 通信圆（以 cell 为单位）
            radius_cells = self.comm_range / env.cell_size
            circle = plt.Circle(
                (x, y),
                radius_cells,
                fill=False,
                linestyle=':',
                linewidth=1.0,
                alpha=0.5,
                edgecolor=color,
            )
            ax_global.add_patch(circle)

            # 标注 R0/R1...
            ax_global.text(
                x + 1, y + 1,
                f'R{i}',
                color='k',
                fontsize=7,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                zorder=8,
            )

        # 通信边
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                pi = self.envs[i].robot_location
                pj = self.envs[j].robot_location
                dist = np.linalg.norm(pi - pj)
                if dist <= self.comm_range:
                    xi = (pi[0] - self.envs[i].belief_origin_x) / self.envs[i].cell_size
                    yi = (pi[1] - self.envs[i].belief_origin_y) / self.envs[i].cell_size
                    xj = (pj[0] - self.envs[j].belief_origin_x) / self.envs[j].cell_size
                    yj = (pj[1] - self.envs[j].belief_origin_y) / self.envs[j].cell_size

                    ax_global.plot(
                        [xi, xj], [yi, yj],
                        linestyle='--',
                        linewidth=1.2,
                        alpha=0.9,
                        color='yellow',
                        zorder=6,
                    )

        ax_global.set_title(f'Global explored: {self.explored_rate_global:.3f}')
        ax_global.set_axis_off()
        if handles:
            ax_global.legend(handles, labels, loc='lower right', fontsize=8)

        # ===================== 下半部分：每个机器人的预测地图 =====================
        for ridx in range(self.num_robots):
            ax_pred = fig.add_subplot(gs[1 + ridx, 0])

            # 该机器人的预测地图（如果还没生成，就退化到 belief）
            robot = self.robots[ridx]
            env = self.envs[ridx]

            if getattr(robot, "pred_mean_map_info", None) is not None and robot.pred_mean_map_info is not None:
                pred_map = robot.pred_mean_map_info.map
                pred_info = robot.pred_mean_map_info
            else:
                pred_map = env.robot_belief
                pred_info = env.ground_truth_info  # 用同一个 origin/cell_size

            ax_pred.imshow(pred_map, cmap='gray', origin='upper', vmin=0, vmax=255)

            # 自己区域的色块
            mask = region_masks[ridx]
            if mask is not None:
                base_color = cmap_regions(ridx)
                rgb = base_color[:3]
                region_rgba = np.zeros((H, W, 4), dtype=float)
                region_rgba[mask, 0] = rgb[0]
                region_rgba[mask, 1] = rgb[1]
                region_rgba[mask, 2] = rgb[2]
                region_rgba[mask, 3] = 0.25
                ax_pred.imshow(region_rgba, origin='upper')

            # 机器人在预测图中的位置（用 cell 画）
            r_cell = get_cell_position_from_coords(
                env.robot_location,
                pred_info,
                check_negative=False
            )
            ax_pred.plot(
                r_cell[0], r_cell[1],
                marker='o',
                markersize=6,
                markeredgecolor='k',
                markeredgewidth=1.0,
                color=cmap_regions(ridx),
                zorder=5,
            )

            # rendezvous 在预测图中的位置
            if getattr(self, "rendezvous_points", None) is not None:
                rv = self.rendezvous_points[ridx]
                if rv is not None:
                    rv_cell = get_cell_position_from_coords(
                        np.array(rv),
                        pred_info,
                        check_negative=False
                    )
                    ax_pred.scatter(
                        [rv_cell[0]], [rv_cell[1]],
                        s=80,
                        marker='*',
                        facecolors=cmap_regions(ridx),
                        edgecolors='k',
                        linewidths=0.8,
                        zorder=5,
                    )

            ax_pred.set_axis_off()
            ax_pred.set_title(f'Robot {ridx} predicted map', fontsize=8)

        # ---------- 保存帧 ----------
        frame_path = f'{gif_path}/{self.global_step}_{step}_multi.png'
        plt.tight_layout()
        plt.savefig(frame_path, dpi=150)
        plt.close()

        self.frame_files.append(frame_path)


@ray.remote(num_cpus=1)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.worker = None
        self.network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
        self.network.to(self.device)
        self.predictor = self.load_predictor()

    def load_predictor(self):
        config_path = f'{generator_path}/config.yaml'
        checkpoint_path = os.path.join(
            generator_path,
            [f for f in os.listdir(generator_path)
             if f.startswith('gen') and f.endswith('.pt')][0]
        )
        with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.SafeLoader)
        generator = Generator(config['netG'], USE_GPU)
        generator.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.predictor = Evaluator(config, generator, USE_GPU, N_GEN_SAMPLE)
        print("Map predictor loaded from {}".format(checkpoint_path))
        return self.predictor

    def set_weights(self, weights):
        self.network.load_state_dict(weights)

    def do_job(self, episode_number):
        # 单机器人：沿用原来的 TestWorker
        if NUM_ROBOTS == 1:
            self.worker = TestWorker(
                self.meta_agent_id,
                self.network,
                self.predictor,
                episode_number,
                device=self.device,
                save_image=SAVE_GIFS
            )
        else:
            # 多机器人：使用 MultiTestWorker
            self.worker = MultiTestWorker(
                self.meta_agent_id,
                self.network,
                self.predictor,
                episode_number,
                num_robots=NUM_ROBOTS,
                comm_range=COMM_RANGE,
                device=self.device,
                save_image=SAVE_GIFS
            )

        self.worker.run_episode()
        perf_metrics = self.worker.perf_metrics
        return perf_metrics

    def job(self, weights, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))

        self.set_weights(weights)

        metrics = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
            "map_path": self.worker.env.map_path,
        }

        return metrics, info


if __name__ == '__main__':
    ray.init()
    run_test()
