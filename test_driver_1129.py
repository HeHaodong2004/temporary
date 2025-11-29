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

# Other configuration settings in parameter.py
NUM_TEST = 1
NUM_META_AGENT = 1  # number of parallel tests, NUM_TEST % NUM_META_AGENT should be 0
SAFE_MODE = True
SAVE_GIFS = True

# ===== Multi-robot settings =====
NUM_ROBOTS = 3          # 每个环境里的机器人数量，先随便给个数
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
        print('| Room success rate: {}, travel distance: {:.2f} ± {:.2f}'.format(np.mean(sr_room), np.mean(td_room), np.std(td_room)))
        print('| Tunnel success rate: {}, travel distance: {:.2f} ± {:.2f}'.format(np.mean(sr_tunnel), np.mean(td_tunnel), np.std(td_tunnel)))
        print('| Outdoor success rate: {}, travel distance: {:.2f} ± {:.2f}'.format(np.mean(sr_outdoor), np.mean(td_outdoor), np.std(td_outdoor)))


    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


class TestWorker:
    def __init__(self, meta_agent_id, policy_net, predictor, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image, test=True)

        self.robot = Agent(policy_net, predictor, self.device, self.save_image)

        self.ground_truth_node_manager = GroundTruthNodeManager(self.robot.node_manager, self.env.ground_truth_info,
                                                                device=self.device, plot=self.save_image)

        self.perf_metrics = dict()
        self.location_history = []
        self.loop_detected = False
        self.path_to_nearest_frontier = None

    def run_episode(self):
        done = False
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        observation = self.robot.get_observation()
        self.ground_truth_node_manager.get_ground_truth_observation(self.env.robot_location, self.robot.pred_mean_map_info)

        if self.save_image:
            self.robot.plot_env()
            self.robot.pred_node_manager.plot_predicted_env(self.env.robot_location, self.robot.map_info.map)
            self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location)
            self.env.plot_env(0)

        for i in range(MAX_EPISODE_STEP):
            self.location_history.append(self.env.robot_location)
            if SAFE_MODE and len(self.location_history) >= 4:
                if np.array_equal(self.location_history[-1], self.location_history[-3]) and \
                   np.array_equal(self.location_history[-2], self.location_history[-4]):
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
            self.ground_truth_node_manager.get_ground_truth_observation(self.env.robot_location, self.robot.pred_mean_map_info)

            if self.save_image:
                self.robot.plot_env()
                self.robot.pred_node_manager.plot_predicted_env(self.env.robot_location, self.robot.map_info.map)
                self.ground_truth_node_manager.plot_ground_truth_env(self.env.robot_location)
                self.env.plot_env(i+1)

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
    - success 的判据：全局覆盖率 > 0.999 或所有 robot 没有可用的 frontier
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

        # --- 创建多个 Env：同一张地图 ---
        # 第一个 env 正常创建
        base_env = Env(global_step, plot=save_image, test=True)
        self.envs = [base_env]

        # 其他 env 强制共享同一张 ground_truth / map_path
        for k in range(1, num_robots):
            env_k = Env(global_step, plot=save_image, test=True)
            env_k.ground_truth = base_env.ground_truth
            env_k.ground_truth_info = base_env.ground_truth_info
            env_k.map_path = base_env.map_path

            # 重置 belief（全未知），并做一次传感器观测，保证初始状态一致
            env_k.robot_belief = np.ones_like(base_env.ground_truth) * UNKNOWN
            env_k.robot_belief = sensor_work(
                env_k.robot_cell,
                env_k.sensor_range / env_k.cell_size,
                env_k.robot_belief,
                env_k.ground_truth
            )
            env_k.belief_info = MapInfo(
                env_k.robot_belief,
                env_k.belief_origin_x,
                env_k.belief_origin_y,
                env_k.cell_size
            )
            self.envs.append(env_k)

        # 为了兼容 run_test 里用 self.worker.env.map_path 的写法，这里给一个别名
        self.env = self.envs[0]

        # --- 每个机器人一个 Agent & GroundTruthNodeManager ---
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

        # 每个机器人的位置历史（可以以后加 loop detect）
        self.location_histories = [[] for _ in range(num_robots)]

        # 指标
        self.perf_metrics = dict()
        self.explored_rate_global = 0.0

    # --------- 一些辅助函数 ---------
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
                    # UNKNOWN 保持 127 即可

                    # 更新两个 env 的 belief 和 MapInfo
                    self.envs[i].robot_belief = merged.copy()
                    self.envs[j].robot_belief = merged.copy()
                    self.envs[i].belief_info.map = self.envs[i].robot_belief
                    self.envs[j].belief_info.map = self.envs[j].robot_belief

                    # 单机 explored_rate 用原有函数更新一下（可选）
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

    # --------- 主循环 ---------
    def run_episode(self):
        done = False

        # 初始：为每个机器人建立图、预测地图等
        self._update_planning_states()
        # 先跑一遍 observation，这样 self.robots[i].utility 会被填充
        for i in range(self.num_robots):
            _ = self.robots[i].get_observation()
            self.ground_truth_node_managers[i].get_ground_truth_observation(
                self.envs[i].robot_location,
                self.robots[i].pred_mean_map_info
            )

        # 如果要画图，这里可以在每个 env 上画一下（先略）

        for step in range(MAX_EPISODE_STEP):
            next_locations = []

            # 1) 每个机器人基于自己的 belief 决策下一 waypoint
            for i in range(self.num_robots):
                env_i = self.envs[i]
                robot_i = self.robots[i]

                self.location_histories[i].append(env_i.robot_location)
                obs_i = robot_i.get_observation()
                # 暂时不做 loop detect，直接 greedy
                next_loc_i, _ = robot_i.select_next_waypoint(obs_i, greedy=True)
                next_locations.append(next_loc_i)

            # 2) 所有机器人同步执行一步 env.step
            for i in range(self.num_robots):
                self.envs[i].step(next_locations[i])

            # 3) 通信范围内合并 belief
            self._apply_communication()

            # 4) belief 更新后，重新更新规划状态
            self._update_planning_states()

            # 5) 计算全局覆盖率、终止条件
            explored_global = self._evaluate_global_explored_rate()

            # 所有 robot 都没有 utility（即没有 frontier）？
            no_utility = True
            for i in range(self.num_robots):
                # 注意：utility 是在 get_observation 里更新的，这里再调一次保证最新
                _ = self.robots[i].get_observation()
                if (self.robots[i].utility > 0).any():
                    no_utility = False
                    break

            if explored_global > 0.999 or no_utility:
                done = True

            # 6) 可选：画图，这里你以后可以改成“全局 belief + 多个机器人轨迹”的版本

            if done:
                break

        if not done:
            print(f"[Multi] Exploration not completed in env: {self.env.map_path}")

        # ------- 记录指标（保持接口兼容 run_test） -------
        # travel_dist 用“所有机器人路程之和”，explored_rate 用全局覆盖率
        total_dist = sum(env.travel_dist for env in self.envs)

        self.perf_metrics['travel_dist'] = total_dist
        self.perf_metrics['explored_rate'] = self.explored_rate_global
        self.perf_metrics['success_rate'] = done

        # 你也可以顺便存一下每个机器人的路程，方便后处理
        self.perf_metrics['travel_dist_each'] = [env.travel_dist for env in self.envs]

        # 如果要保存 gif，这里可以以后自己实现一个 multi-agent 版本
        if self.save_image:
            print("Multi-agent gif not implemented yet (you can add your own plotting here).")


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
        checkpoint_path = os.path.join(generator_path, [f for f in os.listdir(generator_path)
                                                        if f.startswith('gen') and f.endswith('.pt')][0])
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
        # 多机器人：使用 MultiTestWorker
        else:
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
