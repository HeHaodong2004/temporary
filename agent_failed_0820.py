import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from utils import *
from parameter import *
from node_manager import NodeManager
from ground_truth_node_manager import GroundTruthNodeManager  # 保留引用，当前实现未直接用它组装观测


class Agent:
    def __init__(self, id, policy_net, predictor, node_manager, device='cpu', plot=False):
        self.id = id
        self.device = device
        self.policy_net = policy_net
        self.predictor = predictor
        self.plot = plot

        # map-related
        self.location = None
        self.map_info = None
        self.cell_size = CELL_SIZE
        self.node_resolution = NODE_RESOLUTION
        self.updating_map_size = UPDATING_MAP_SIZE

        self.updating_map_info = None
        self.frontier = set()

        # 共享图：由外部传入的 NodeManager
        self.node_manager = node_manager

        # 预测输出
        self.pred_mean_map_info, self.pred_max_map_info = None, None

        # 观测缓存
        self.node_coords = None
        self.utility = None
        self.guidepost = None           # 这里用 explored_sign 来占位
        self.explored_sign = None
        self.adjacent_matrix = None
        self.neighbor_indices = None

        # 训练缓存
        self.episode_buffer = [[] for _ in range(27)]

        # 可视化轨迹
        if plot:
            self.trajectory_x, self.trajectory_y = [], []
            
        self.intent_seq = []

        self.rdv_path_nodes_set = set()
        

    # ---------- 状态更新 ----------
    def update_map(self, map_info):
        self.map_info = map_info

    def update_location(self, location):
        self.location = np.around(location, 1)
        if self.plot:
            self.trajectory_x.append(self.location[0])
            self.trajectory_y.append(self.location[1])
        if self.node_manager.nodes_dict.__len__() != 0:
            node = self.node_manager.nodes_dict.find(self.location.tolist())
            if node is not None:
                node.data.set_visited()

    def get_updating_map(self, location, base=None):
        """
        从 base(MapInfo) 上裁局部窗口；默认用 self.map_info。
        """
        base = base if base is not None else self.map_info
        ox = (location[0] - self.updating_map_size / 2)
        oy = (location[1] - self.updating_map_size / 2)
        tx = ox + self.updating_map_size
        ty = oy + self.updating_map_size

        min_x, min_y = base.map_origin_x, base.map_origin_y
        max_x = base.map_origin_x + self.cell_size * (base.map.shape[1] - 1)
        max_y = base.map_origin_y + self.cell_size * (base.map.shape[0] - 1)
        ox = max(ox, min_x); oy = max(oy, min_y)
        tx = min(tx, max_x); ty = min(ty, max_y)

        ox = (ox // self.cell_size + 1) * self.cell_size
        oy = (oy // self.cell_size + 1) * self.cell_size
        tx = (tx // self.cell_size) * self.cell_size
        ty = (ty // self.cell_size) * self.cell_size

        ox = np.round(ox, 1); oy = np.round(oy, 1); tx = np.round(tx, 1); ty = np.round(ty, 1)
        o = np.array([ox, oy]); t = np.array([tx, ty])
        oidx = get_cell_position_from_coords(o, base)
        tidx = get_cell_position_from_coords(t, base)

        sub = base.map[oidx[1]:tidx[1] + 1, oidx[0]:tidx[0] + 1]
        return MapInfo(sub, ox, oy, self.cell_size)

    def update_frontiers(self):
        self.frontier = get_frontier_in_map(self.updating_map_info)

    def update_graph(self, map_info, location):
        """
        用 belief 更新前沿与节点属性；连边/碰撞仍由 NodeManager 内部依据 updating_map_info 判定。
        """
        self.update_map(map_info)
        self.update_location(location)
        self.updating_map_info = self.get_updating_map(self.location, base=self.map_info)
        self.update_frontiers()
        self.node_manager.update_graph(self.location, self.frontier, self.updating_map_info, self.map_info)

    def update_predict_map(self):
        """
        基于当前 agent 的 belief 进行补全预测，得到 pred_mean_map_info / pred_max_map_info。
        """
        x_belief, mask, x_raw = self.pre_process_input()
        onehots = torch.tensor(
            [[0.333, 0.333, 0.333], [1, 0, 0], [0, 1, 0], [0, 0, 1],
             [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]],
            device=x_belief.device
        ).unsqueeze(1).float()

        preds = []
        for i in range(self.predictor.nsample):
            _, x_inpaint = self.predictor.eval_step(x_belief, mask, onehots[i], self.map_info.map.shape)
            x_proc = self.predictor.post_process(x_inpaint, x_raw, kernel_size=5)
            x_proc = np.where(x_proc > 0, FREE, OCCUPIED)
            preds.append(x_proc)

        self.pred_mean_map_info = MapInfo(
            np.mean(preds, axis=0), self.map_info.map_origin_x, self.map_info.map_origin_y, self.cell_size
        )
        self.pred_max_map_info = MapInfo(
            np.max(preds, axis=0), self.map_info.map_origin_x, self.map_info.map_origin_y, self.cell_size
        )

    def update_planning_state(self, robot_locations, intents_view=None):
        """
        与 planner 状态同步；这里用“viewer 的视角”生成自己的 intent_seq
        """
        try:
            self.intent_seq = self.predict_intent_path(
                k=INTENT_HORIZON,
                robot_locations_view=robot_locations,
                global_intents_view=intents_view
            )
        except Exception:
            self.intent_seq = []

    def predict_intent_path(self, k, robot_locations_view=None, global_intents_view=None):
        """
        用和 actor 一样的观测做 k 步 rollout；但将自己的 intent 通道置空避免自循环。
        """
        device = next(self.policy_net.parameters()).device

        # 拷贝 viewer 的 intents，并清空自己的通道
        intents_for_rollout = {}
        if isinstance(global_intents_view, dict):
            intents_for_rollout = {aid: list(path) for aid, path in global_intents_view.items()}
            intents_for_rollout[self.id] = []  # 自己的未来不要喂进去
        else:
            intents_for_rollout = {}

        # 观测与 actor 一致（含他人的 intents + viewer 自己看到的位置）
        obs = self.get_observation(
            robot_locations=robot_locations_view,
            global_intents=intents_for_rollout
        )
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = [
            t.to(device) for t in obs
        ]

        intent = []
        for _ in range(k):
            with torch.no_grad():
                logp = self.policy_net(node_inputs, node_padding_mask, edge_mask,
                                    current_index, current_edge, edge_padding_mask)
            a = int(torch.argmax(logp, dim=1).item())
            next_idx = int(current_edge[0, a, 0].item())
            intent.append(self.node_coords[next_idx].tolist())

            # 游标前移
            current_index = torch.tensor([[[next_idx]]], device=device, dtype=torch.long)
            # 根据邻接矩阵快速刷新 current_edge（注意：node_inputs 没有重新中心化，相当于近似）
            neigh = np.argwhere(self.adjacent_matrix[next_idx] == 0).reshape(-1)
            ce = torch.tensor(neigh, dtype=torch.long, device=device).view(1, -1, 1)
            pad = K_SIZE - ce.size(1)
            if pad > 0:
                ce = torch.nn.functional.pad(ce, (0, 0, 0, pad), value=0)
            current_edge = ce

            ep = torch.cat([
                torch.zeros(1, 1, ce.size(1) - pad, dtype=torch.int16, device=device),
                torch.ones (1, 1, pad,              dtype=torch.int16, device=device)
            ], dim=-1)
            edge_padding_mask = ep

        return intent



    # ---------- 观测/决策 ----------
    def pre_process_input(self):
        width_in, height_in, _ = self.predictor.config['image_shape']
        width_map, height_map = self.map_info.map.shape

        pad = width_map < width_in and height_map < height_in
        if pad:
            pad_left = (width_in - width_map) // 2
            pad_top = (height_in - height_map) // 2
            pad_right = width_in - width_map - pad_left
            pad_bottom = height_in - height_map - pad_top
            belief = np.pad(self.map_info.map, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
        else:
            belief = self.map_info.map

        mask = belief.copy()
        mask[mask != UNKNOWN] = 0
        mask[mask == UNKNOWN] = FREE

        x_raw = Image.fromarray(self.map_info.map).convert('L')
        x_belief = Image.fromarray(belief).convert('L')
        mask = Image.fromarray(mask).convert('1')
        if not pad:
            x_belief = transforms.Resize((width_in, height_in))(x_belief)
            mask = transforms.Resize((width_in, height_in))(mask)
        x_belief = transforms.ToTensor()(x_belief).unsqueeze(0).to(self.predictor.device).mul_(2).add_(-1)
        x_raw = transforms.ToTensor()(x_raw).unsqueeze(0).to(self.predictor.device).mul_(2).add_(-1)
        mask = transforms.ToTensor()(mask).unsqueeze(0).to(self.predictor.device)
        return x_belief, mask, x_raw

    def _assemble_observation_from_shared_graph(self, robot_locations=None, global_intents=None):
        """
        特征顺序更新为：
        rel_xy(2) + utility(1) + pred_prob(1) + explored_sign(1) + guidepost2(1,占位0)
        + occupancy(1) + intent_mask(N_AGENTS) + connectivity_mask(N_AGENTS) + rdv_path(1)
        => 共 8 + 2*N_AGENTS 维
        """
        # ---------- 1) 收集所有节点坐标 ----------
        all_node_coords = []
        for n in self.node_manager.nodes_dict.__iter__():
            all_node_coords.append(np.around(n.data.coords, 1))
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)

        n_nodes = all_node_coords.shape[0]
        if n_nodes == 0:
            node_inputs = torch.zeros((1, NODE_PADDING_SIZE, NODE_INPUT_DIM), dtype=torch.float32, device=self.device)
            node_padding_mask = torch.ones((1, 1, NODE_PADDING_SIZE), dtype=torch.int16, device=self.device)
            edge_mask = torch.ones((1, NODE_PADDING_SIZE, NODE_PADDING_SIZE), dtype=torch.float32, device=self.device)
            current_index_t = torch.tensor([0], dtype=torch.long, device=self.device).reshape(1, 1, 1)
            current_edge = torch.zeros((1, K_SIZE, 1), dtype=torch.long, device=self.device)
            edge_padding_mask = torch.ones((1, 1, K_SIZE), dtype=torch.int16, device=self.device)
            pack = [node_inputs, node_padding_mask, edge_mask, current_index_t, current_edge, edge_padding_mask]
            meta = [np.zeros((0,2)), np.zeros((0,)), np.zeros((0,)), np.zeros((0,)), np.ones((0,0), dtype=int), np.zeros((0,), dtype=int)]
            return pack, meta

        # ---------- 2) utility 与邻接矩阵 ----------
        utility = []
        adjacent_matrix = np.ones((n_nodes, n_nodes), dtype=int)
        key_all = all_node_coords[:, 0] + 1j * all_node_coords[:, 1]
        for i, coords in enumerate(all_node_coords):
            nd = self.node_manager.nodes_dict.find(coords.tolist()).data
            utility.append(nd.utility)
            for nb in nd.neighbor_set:
                idx = np.argwhere(key_all == (nb[0] + 1j*nb[1]))
                if idx.size > 0:
                    adjacent_matrix[i, idx[0][0]] = 0
        utility = np.array(utility, dtype=np.float32)

        # ---------- 3) explored_sign ----------
        explored_sign = []
        for coords in all_node_coords:
            cell = get_cell_position_from_coords(coords, self.map_info)
            known = (self.map_info.map[cell[1], cell[0]] != UNKNOWN)
            explored_sign.append(1 if known else 0)
        explored_sign = np.array(explored_sign, dtype=np.float32)

        # ---------- 4) pred_prob ----------
        if self.pred_mean_map_info is None:
            pred_prob = np.zeros((n_nodes,), dtype=np.float32)
        else:
            pred_prob = []
            for coords in all_node_coords:
                cell = get_cell_position_from_coords(coords, self.pred_mean_map_info)
                pred_prob.append(float(self.pred_mean_map_info.map[cell[1], cell[0]]))
            pred_prob = np.array(pred_prob, dtype=np.float32)

        # ---------- 5) current_index 与邻居 ----------
        mykey = self.location[0] + 1j * self.location[1]
        idx_arr = np.argwhere(key_all == mykey)
        if idx_arr.size == 0:
            nn = self.node_manager.nodes_dict.nearest_neighbors(self.location.tolist(), 1)[0].data.coords
            mykey = nn[0] + 1j * nn[1]
            idx_arr = np.argwhere(key_all == mykey)
        current_index = int(idx_arr[0][0])

        curr_node = self.node_manager.nodes_dict.find(all_node_coords[current_index].tolist()).data
        neighbor_indices = []
        for nb in curr_node.neighbor_set:
            idx = np.argwhere(key_all == (nb[0] + 1j*nb[1]))
            if idx.size > 0:
                neighbor_indices.append(int(idx[0][0]))
        neighbor_indices = np.unique(np.array(neighbor_indices, dtype=np.int64))

        # ---------- 6) occupancy ----------
        occupancy = np.zeros((n_nodes,), dtype=np.float32)
        occupancy[current_index] = -1.0
        if robot_locations is not None:
            for pos in robot_locations:
                if np.allclose(pos, self.location):
                    continue
                try:
                    nn = self.node_manager.nodes_dict.nearest_neighbors(pos.tolist(), 1)[0].data.coords
                    jarr = np.argwhere(key_all == (nn[0] + 1j*nn[1]))
                    if jarr.size > 0:
                        j = int(jarr[0][0])
                        if j != current_index:
                            occupancy[j] = 1.0
                except Exception:
                    pass

        # ---------- 7) intent mask ----------
        def to_key_xy(p):
            return (round(float(p[0]), 1), round(float(p[1]), 1))

        if global_intents is None:
            intent_mask = np.zeros((n_nodes, N_AGENTS), dtype=np.float32)
        else:
            path_sets = {}
            for aid in range(N_AGENTS):
                path = global_intents.get(aid, []) if isinstance(global_intents, dict) else []
                path_sets[aid] = {to_key_xy(p) for p in path} if path else set()

            intent_mask_cols = []
            for aid in range(N_AGENTS):
                s = path_sets[aid]
                col = np.zeros((n_nodes, 1), dtype=np.float32)
                if len(s) > 0:
                    for i, c in enumerate(all_node_coords):
                        key = (round(float(c[0]), 1), round(float(c[1]), 1))
                        if key in s:
                            col[i, 0] = 1.0
                intent_mask_cols.append(col)
            intent_mask = np.concatenate(intent_mask_cols, axis=1) if len(intent_mask_cols) > 0 else np.zeros((n_nodes, N_AGENTS), dtype=np.float32)

        # ---------- 8) 新增：connectivity mask（每节点复制同一行） ----------
        # 直连（≤ COMMS_RANGE）可通信为 1；自身也置 1
        connectivity = np.zeros((N_AGENTS,), dtype=np.float32)
        connectivity[self.id] = 1.0
        if robot_locations is not None:
            me = np.array(self.location, dtype=float)
            for aid in range(N_AGENTS):
                pos = np.array(robot_locations[aid], dtype=float)
                if np.linalg.norm(pos - me) <= float(COMMS_RANGE) + 1e-6:
                    connectivity[aid] = 1.0
        connectivity_mask = np.tile(connectivity.reshape(1, -1), (n_nodes, 1))  # (n_nodes, N_AGENTS)

        # ---------- 9) 新增：rdv_path mask（节点在路径上=1） ----------
        rdv_path = np.zeros((n_nodes, 1), dtype=np.float32)
        if isinstance(self.rdv_path_nodes_set, set) and len(self.rdv_path_nodes_set) > 0:
            path_keys = self.rdv_path_nodes_set
            for i, c in enumerate(all_node_coords):
                key = (round(float(c[0]), 1), round(float(c[1]), 1))
                if key in path_keys:
                    rdv_path[i, 0] = 1.0

        # ---------- 10) 组装特征 ----------
        node_coords = all_node_coords
        current_node_coords = node_coords[current_index]
        rel_xy = np.concatenate(
            (node_coords[:, [0]] - current_node_coords[0], node_coords[:, [1]] - current_node_coords[1]),
            axis=-1
        ) / UPDATING_MAP_SIZE / 2.0

        node_utility   = utility.reshape(-1, 1) / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        node_predprob  = pred_prob.reshape(-1, 1) / float(FREE)
        node_guidepost = explored_sign.reshape(-1, 1)
        node_guidepost2 = np.zeros_like(node_guidepost)
        node_occupancy = occupancy.reshape(-1, 1)

        feats = np.concatenate(
            (
                rel_xy, node_utility, node_predprob, node_guidepost,
                node_guidepost2, node_occupancy,
                intent_mask,                 # N_AGENTS
                connectivity_mask,           # N_AGENTS
                rdv_path                     # 1
            ),
            axis=1
        )
        assert feats.shape[1] == NODE_INPUT_DIM, f"NODE_INPUT_DIM({NODE_INPUT_DIM}) != feats({feats.shape[1]})"
        node_inputs = torch.as_tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)

        # ---------- 11) padding 与 mask ----------
        n_node = node_coords.shape[0]
        assert n_node < NODE_PADDING_SIZE, f"{n_node} >= {NODE_PADDING_SIZE}"
        node_inputs = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))(node_inputs)

        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16, device=self.device)
        node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16, device=self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        edge_mask = torch.as_tensor(adjacent_matrix, dtype=torch.float32, device=self.device).unsqueeze(0)
        edge_mask = torch.nn.ConstantPad2d((0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)(edge_mask)

        if current_index not in neighbor_indices.tolist():
            neighbor_indices = np.unique(np.concatenate([neighbor_indices, np.array([current_index], dtype=np.int64)]))

        current_edge = torch.as_tensor(neighbor_indices, dtype=torch.long, device=self.device).unsqueeze(0)
        k_size = current_edge.size(-1)
        try:
            current_in_edge = int(np.argwhere(neighbor_indices == current_index)[0][0])
        except Exception:
            current_in_edge = 0
        current_edge = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)(current_edge).unsqueeze(-1)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16, device=self.device)
        if k_size > 0 and 0 <= current_in_edge < k_size:
            edge_padding_mask[0, 0, current_in_edge] = 1
        edge_padding_mask = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)(edge_padding_mask)

        current_index_t = torch.as_tensor([current_index], dtype=torch.long, device=self.device).reshape(1, 1, 1)

        pack = [node_inputs, node_padding_mask, edge_mask, current_index_t, current_edge, edge_padding_mask]
        meta = [node_coords, utility, node_guidepost.squeeze(1), explored_sign, adjacent_matrix, neighbor_indices]
        return pack, meta

    def get_observation(self, robot_locations=None, global_intents=None):
        pack, meta = self._assemble_observation_from_shared_graph(
            robot_locations=robot_locations,
            global_intents=global_intents
        )
        [self.node_coords, self.utility, self.guidepost, self.explored_sign,
         self.adjacent_matrix, self.neighbor_indices] = meta
        return pack

    def select_next_waypoint(self, observation, greedy=False):
        _, _, _, _, current_edge, _ = observation
        with torch.no_grad():
            logp = self.policy_net(*observation)
        action_index = (torch.argmax(logp, dim=1).long()
                        if greedy else torch.multinomial(logp.exp(), 1).long().squeeze(1))
        next_node_index = int(current_edge[0, action_index.item(), 0].item())
        next_position = self.node_coords[next_node_index]
        return next_position, next_node_index, action_index

    # ---------- 缓冲保存 ----------
    def save_observation(self, observation, critic_observation):
        n, m, e, ci, ce, ep = observation
        self.episode_buffer[0] += n
        self.episode_buffer[1] += m.bool()
        self.episode_buffer[2] += e.bool()
        self.episode_buffer[3] += ci
        self.episode_buffer[4] += ce
        self.episode_buffer[5] += ep.bool()

        cn, cm, cee, cci, cce, cep = critic_observation
        self.episode_buffer[15] += cn
        self.episode_buffer[16] += cm.bool()
        self.episode_buffer[17] += cee.bool()
        self.episode_buffer[18] += cci
        self.episode_buffer[19] += cce
        self.episode_buffer[20] += cep.bool()

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.reshape(1, 1, 1)

    def save_reward(self, reward):
        self.episode_buffer[7] += torch.as_tensor([reward], dtype=torch.float32, device=self.device).reshape(1, 1, 1)

    def save_done(self, done):
        self.episode_buffer[8] += torch.as_tensor([int(done)], dtype=torch.long, device=self.device).reshape(1, 1, 1)

    def save_next_observations(self, observation, critic_observation):
        n, m, e, ci, ce, ep = observation
        self.episode_buffer[9] += n
        self.episode_buffer[10] += m.bool()
        self.episode_buffer[11] += e.bool()
        self.episode_buffer[12] += ci
        self.episode_buffer[13] += ce
        self.episode_buffer[14] += ep.bool()

        cn, cm, cee, cci, cce, cep = critic_observation
        self.episode_buffer[21] += cn
        self.episode_buffer[22] += cm.bool()
        self.episode_buffer[23] += cee.bool()
        self.episode_buffer[24] += cci
        self.episode_buffer[25] += cce
        self.episode_buffer[26] += cep.bool()

    # ---------- 可视化 ----------
    def plot_env(self):
        if not self.plot:
            return
        plt.switch_backend('agg')
        plt.figure(figsize=(19, 5))
        plt.subplot(1, 4, 2)
        plt.axis('off')

        nodes = get_cell_position_from_coords(self.node_coords, self.map_info)
        if len(self.frontier) > 0:
            fr = get_cell_position_from_coords(np.array(list(self.frontier)), self.map_info).reshape(-1, 2)
            plt.scatter(fr[:, 0], fr[:, 1], c='r', s=2, zorder=4)

        robot = get_cell_position_from_coords(self.location, self.map_info)

        # 叠加预测 + belief
        if self.pred_max_map_info is not None:
            plt.imshow(self.pred_max_map_info.map, cmap='gray', vmin=0, vmax=255)
        alpha_mask = (self.map_info.map == FREE) * 0.5
        plt.imshow(self.map_info.map, cmap='Blues', alpha=alpha_mask)

        util_vis = np.where(self.utility > 0, self.utility, 0).astype(np.uint8)
        plt.scatter(nodes[:, 0], nodes[:, 1], c=util_vis, zorder=2)
        for c, u in zip(nodes, util_vis):
            if u > 0:
                plt.text(c[0], c[1], str(u), fontsize=8, zorder=3)
        plt.plot(robot[0], robot[1], 'mo', markersize=16, zorder=5)

        # 画边：使用共享 NodeManager
        for coords in self.node_coords:
            n = self.node_manager.nodes_dict.find(coords.tolist())
            if n is None:
                continue
            node = n.data
            for neigh in node.neighbor_set:
                end = (np.array(neigh) - coords) / 2 + coords
                plt.plot((np.array([coords[0], end[0]]) - self.map_info.map_origin_x) / self.cell_size,
                         (np.array([coords[1], end[1]]) - self.map_info.map_origin_y) / self.cell_size,
                         'tan', linewidth=1, zorder=1)
