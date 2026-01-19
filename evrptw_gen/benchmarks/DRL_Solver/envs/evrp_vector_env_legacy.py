import gym
import numpy as np
from gym import spaces

from evrptw_gen.generator import InstanceGenerator

def assign_env_config(self, kwargs):
    """
    Set self.key = value, for each key in kwargs
    """
    for key, value in kwargs.items():
        setattr(self, key, value)

def gen_dist_matrix(nodes1, nodes2):
    # nodes1: (N1, d)
    # nodes2: (N2, d)
    diff = nodes1[:, None, :] - nodes2[None, :, :]
    return np.linalg.norm(diff, axis=-1)

def fun_dist(loc1, loc2):
    return ((loc1[:, 0] - loc2[:, 0]) ** 2 + (loc1[:, 1] - loc2[:, 1]) ** 2) ** 0.5

class EVRPTWVectorEnv(gym.Env):
    def __init__(self, *args, **kwargs):  

        ##################   Configuration  ##################
        self.config_path = kwargs.get("config_path", None)
        if not self.config_path:
            raise ValueError("config_path to construct env is required!")

        n_traj = kwargs.get("n_traj", None)
        self.n_traj            = n_traj if n_traj else 100
        ######################################################
        
        # load dataset
        save_path = kwargs.get("save_path", None)
        num_instances = kwargs.get("num_instances", 1)
        plot_instances = kwargs.get("plot_instances", False)
        self.dataset = InstanceGenerator(self.config_path, 
                                         save_path=save_path, 
                                         num_instances=num_instances, 
                                         plot_instances=plot_instances,
                                         kwargs=kwargs)

        config_data = self.dataset.config.data

        # 

        self.cus_num         = config_data.get('num_customers', None)
        self.rs_num          = config_data.get('num_charging_stations', None)

        if not self.cus_num or not self.rs_num:
            raise ValueError("customer number of charging station number is not predefined!")

        self.env_mode = "train"

        assign_env_config(self, kwargs)

        obs_dict = {"cus_loc": spaces.Box(low=0, high=1, shape=(self.cus_num, 2))}
        obs_dict["depot_loc"] = spaces.Box(low=0, high=1, shape=(1,2))
        obs_dict["rs_loc"] = spaces.Box(low = 0, high=1, shape=(self.rs_num, 2))
        obs_dict["demand"] = spaces.Box(low=0, high=1, shape=(1 + self.cus_num + self.rs_num,))
        obs_dict["time_window"] = spaces.Box(low = 0, high=1, shape=(1 + self.cus_num + self.rs_num, 2))
        obs_dict["action_mask"] = spaces.MultiBinary(
            [self.n_traj, self.cus_num + self.rs_num + 1]
        )  # 1: OK, 0: cannot go
        obs_dict["last_node_idx"] = spaces.MultiDiscrete([self.cus_num + self.rs_num + 1] * self.n_traj)
        obs_dict["current_load"] = spaces.Box(low=0, high=1, shape=(self.n_traj,))
        obs_dict["current_battery"] = spaces.Box(low=0, high=1, shape=(self.n_traj,))
        obs_dict["current_time"] = spaces.Box(low=0, high=1, shape=(self.n_traj,))
        obs_dict["service_time"] = spaces.Box(low=0, high=1, shape=(1 + self.cus_num + self.rs_num,))
        obs_dict["battery_capacity"] = spaces.Box(low=0, high=np.inf, shape=(1,))
        obs_dict["loading_capacity"] = spaces.Box(low=0, high=np.inf, shape=(1,))

        # obs_dict["instance_mask"] = spaces.Box(low=0, high=1, shape=(self.cus_num + self.rs_num + 1,), dtype=bool)
        # obs_dict["test"] = spaces.Box(low=0, high=1, shape=(self.cus_num + self.rs_num + 1,), dtype=bool)

        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.MultiDiscrete([self.rs_num + self.cus_num + 1] * self.n_traj)
        self.reward_space = None

        self.reset()

    def seed(self, seed):
        np.random.seed(seed)

    def _STEP(self, action):
        self._go_to(action)  # Go to node 'action', modify the reward
        self.num_steps += 1
        self.state = self._update_state()

        # need to revisit the first node after visited all other nodes
        self.done = (action == 0) & self.is_all_visited()
        return self.state, self.reward, self.done, self.info

    def is_all_visited(self):
        # assumes no repetition in the first `cus_num` steps
        return self.visited[:, (1 + self.rs_num):].all(axis=1)

    def _update_state(self, update_mask=True):

        # arrangement: [depot, customers1, customer2, ... , cs1, cs2, ...]
        obs = {}
        obs["cus_loc"] = self.cus_nodes  # n x 2 array
        obs["depot_loc"] = self.depot_nodes
        obs["rs_loc"] = self.rs_nodes
        obs["demand"] = self.demands
        obs["time_window"] = self.time_window

        # Dynamic
        obs["action_mask"] = self._update_mask() if update_mask else self.mask
        obs["last_node_idx"] = self.last
        obs["current_load"] = self.load
        obs["current_battery"] = self.battery
        obs["current_time"] = self.current_time
        obs["service_time"] = self.service_time
        obs["battery_capacity"] = np.array(self.battery_capacity).reshape(-1)
        obs["loading_capacity"] = np.array(self.loading_capacity).reshape(-1)
        # obs["instance_mask"] = None

        return obs

    def _update_mask(self):
        cus_start_idx = 1
        rs_start_idx  = 1 + self.cus_num

        action_mask = ~self.visited

        # (2) load
        load_mask = (self.load[:, None] + self.demands[None, :]) <= self.loading_capacity  # loading_capacity=1.0
        action_mask &= load_mask

        # (3) battery
        battery_need = self.battery[:, None] + self.edge_energy[self.last, :]  # normalized SoC 消耗
        battery_mask = (battery_need <= self.battery_capacity)                 # battery_capacity=1.0
        action_mask &= battery_mask

        # (4) arrival time within TW close
        time_after_arrival = self.current_time[:, None] + self.travel_time[self.last, :]  # normalized time
        time_mask = (time_after_arrival <= self.time_window[:, 1][None, :])
        action_mask &= time_mask

        # (5) FFP
        cus_idx = np.arange(cus_start_idx, rs_start_idx)
        rs_cols = np.r_[0, rs_start_idx:self.travel_time.shape[1]]
        # n_cus = len(cus_idx)
        # n_rs_plus_depot = len(rs_cols)

        time_to_customer = self.travel_time[self.last[:, None], cus_idx]  # (n_traj, n_cus)
        battery_at_customer = self.battery[:, None] + self.edge_energy[self.last[:, None], cus_idx]  # (n_traj, n_cus)

        tw_open = self.time_window[cus_start_idx:rs_start_idx, 0][None, :]
        arrival_time = self.current_time[:, None] + time_to_customer
        start_service_time = np.maximum(arrival_time, tw_open)
        time_after_service = start_service_time + self.service_time[cus_start_idx:rs_start_idx][None, :]

        time_cust_to_rs = self.travel_time[np.ix_(cus_idx, rs_cols)][None, :, :]  # (1, n_cus, n_rs+1)
        battery_cust_to_rs = self.edge_energy[np.ix_(cus_idx, rs_cols)][None, :, :]
        battery_at_customer_3d = battery_at_customer[:, :, None]
        time_after_service_3d  = time_after_service[:, :, None]

        battery_at_rs = battery_at_customer_3d + battery_cust_to_rs  # 如果你要精确按 time 算耗电，可以简化直接用 edge_energy[np.ix_(cus_idx, rs_cols)]
        time_at_rs    = time_after_service_3d + time_cust_to_rs

        # 充到满电所需的 normalized 时间
        time_charge_at_rs = (1.0 - battery_at_rs) * self.charging_beta
        time_charge_at_rs[:, :, 0] = 0.0  # depot 不充电

        RS_time_to_depot_3d = self.RS_time_to_depot[None, None, :]  # normalized

        total_finish_time = time_at_rs + time_charge_at_rs + RS_time_to_depot_3d  # normalized
        time_feasible    = (total_finish_time <= self.instance_max_time)          # <= 1.0
        battery_feasible = (battery_at_rs <= self.battery_capacity)               # <= 1.0

        feasible = time_feasible & battery_feasible
        FFP_cus_mask = feasible.any(axis=2)
        action_mask[:, cus_start_idx:rs_start_idx] &= FFP_cus_mask

        self.mask = action_mask
        return action_mask

    def _RESET(self):
        self.visited = np.zeros((self.n_traj, self.cus_num + self.rs_num + 1), dtype=bool)
        self.visited[:, 0] = True
        self.num_steps = 0
        self.traj = []
        self.restrictions = np.empty((0, 2), dtype=int)
        self.last = np.zeros(self.n_traj, dtype=int)  # idx of the cur elem
        self.load = np.zeros(self.n_traj, dtype=float)  # current load
        self.battery = np.zeros(self.n_traj, dtype=float)  # current battery
        self.current_time = np.zeros(self.n_traj, dtype=float)  # current battery

        if self.env_mode == "eval":
            # evaluation mode
            self._eval_data_generate()
        elif self.env_mode == "train":
            # training mode
            self._train_data_generate()
        else:
            raise ValueError("Unknown mode : {}".format(self.env_mode))

        self.state = self._update_state()
        self.info = {}
        self.done = np.array([False] * self.n_traj)
        
        return self.state

    def _train_data_generate(self):
        context = self.dataset.generate_tensors()
        # RS_time_to_depot = context['env']['cs_time_to_depot']
        # self.RS_time_to_depot = np.concatenate([np.zeros((1,)), RS_time_to_depot])

        self.depot_num = context["depot"].shape[0]
        self.cus_num = context['customers'].shape[0]
        self.rs_num = context['charging_stations'].shape[0]

        # Normalizations
        self._normalizations(context)

    def _normalizations(self, context):
        # ----- 原始节点坐标 -----
        nodes_raw = np.concatenate((context["depot"], context['customers'], context["charging_stations"])).astype(np.float32)
        positions = np.zeros_like(nodes_raw)

        # ----- 需求 & 时间窗 & 服务时间（原始） -----
        demands_abs     = context["demands"].astype(np.float32)            # shape (n_cus,)
        time_window_abs = context["tw"].astype(np.float32)                 # shape (n_cus, 2)
        service_time_abs= context["service_time"].astype(np.float32)       # shape (n_cus,)

        data = context['env']
        consumption_per_km = data["consumption_per_distance"]              # kWh / km
        b_s          = data["battery_capacity"]                            # kWh, E_max
        velocity_abs = data["speed"]                                       # km / hour
        loading_capacity = data['loading_capacity']                        # Q_max
        charging_power_abs = data["charging_speed"] * data['charging_efficiency']  # kW
        instance_max_time_abs = data["instance_endTime"]                   # T_scale
        instance_working_start_time = data['working_startTime']
        instance_working_end_time   = data['working_endTime']
        pos_scale = data["area_size"]

        # --------- 1. 位置归一化（给 obs 用） ---------
        x_scale = pos_scale[0][1] - pos_scale[0][0]
        y_scale = pos_scale[1][1] - pos_scale[1][0]
        positions[:, 0] = (nodes_raw[:, 0] - pos_scale[0][0]) / x_scale
        positions[:, 1] = (nodes_raw[:, 1] - pos_scale[1][0]) / y_scale

        # --------- 2. demand 归一化 ---------
        demands_norm_cus   = demands_abs / loading_capacity     # [0,1]
        demands_norm_depot = np.zeros((1,), dtype=np.float32)
        demands_norm_rs    = np.zeros((self.rs_num,), dtype=np.float32)

        # --------- 3. 时间归一化（以 instance_max_time 为 T_scale） ---------
        T_scale = instance_max_time_abs

        time_window_norm_cus = np.clip(time_window_abs / T_scale, 0.0, 1.0)
        time_window_norm_depot = np.array([[0.0, 1.0]], dtype=np.float32)
        time_window_norm_rs    = np.array([[0.0, 1.0] * self.rs_num], dtype=np.float32).reshape(self.rs_num, 2)

        service_time_norm_cus   = np.clip(service_time_abs / T_scale, 0.0, 1.0)
        service_time_norm_depot = np.zeros((1,), dtype=np.float32)
        service_time_norm_rs    = np.zeros((self.rs_num,), dtype=np.float32)

        # --------- 4. 距离矩阵（用 *原始* 坐标算，给时间 & 电量用） ---------
        dist_matrix_raw = gen_dist_matrix(nodes_raw, nodes_raw).astype(np.float32)  # km

        # travel time in normalized units:
        travel_time_norm = (dist_matrix_raw / velocity_abs) / T_scale  # (i,j) 的 normalized travel time

        # --------- 5. 边上的电量消耗（normalized SoC） ---------
        # 每条边消耗的 SoC 比例： dist * (kWh/km) / (kWh)
        edge_energy_frac = dist_matrix_raw * consumption_per_km / b_s   # ∈ [0,1+]，取决于距离

        # --------- 6. 充电相关（normalized 时间） ---------
        # 充满 100% SoC 需要的 normalized 时间
        beta = b_s / (charging_power_abs * T_scale)  # E_max / (P_chg * T_scale)

        # --------- 7. 更新 env 内部变量 ---------
        self.nodes_raw   = nodes_raw
        self.nodes       = positions
        self.depot_nodes = positions[0:1]
        self.cus_nodes   = positions[1 : 1 + self.cus_num]
        self.rs_nodes    = positions[1 + self.cus_num : 1 + self.cus_num + self.rs_num]

        self.demands = np.concatenate((demands_norm_depot, demands_norm_cus, demands_norm_rs))
        self.time_window = np.concatenate((time_window_norm_depot, time_window_norm_cus, time_window_norm_rs))
        self.service_time = np.concatenate((service_time_norm_depot, service_time_norm_cus, service_time_norm_rs))

        # 全部用 normalized 版本
        self.travel_time = travel_time_norm          # (1+cus+rs, 1+cus+rs)
        self.edge_energy = edge_energy_frac          # normalized SoC edge cost
        self.charging_beta = beta                    # 每 1.0 SoC 需要的 normalized 时间

        # 容量在 normalized 世界里都是 1
        self.loading_capacity = 1.0
        self.battery_capacity = 1.0
        self.instance_max_time = 1.0  # 所有 TW 和 current_time 也都在 [0,1]

        # 用于 FFP 的 RS_time_to_depot 也要 normalized
        RS_time_to_depot_abs = context['env']['cs_time_to_depot']         # shape (n_rs,)
        self.RS_time_to_depot = np.concatenate(
            [np.zeros((1,), dtype=np.float32),
            (RS_time_to_depot_abs / T_scale).astype(np.float32)]
        )

        self.dist_matrix = gen_dist_matrix(self.nodes, self.nodes).astype(np.float32)  # 仅用于 reward（normalized 坐标）

    def _eval_data_generate(self):
        raise NotImplementedError("Not Implement Yet!")

    def _go_to(self, destination):
        dest_node = self.nodes[destination]
        dist_display = fun_dist(dest_node, self.nodes[self.last])  # 仅用于 reward (normalized 坐标)
        cus_start_idx = 1
        rs_start_idx = 1 + self.cus_num

        go_to_depot = (destination == 0)
        go_to_rs    = (destination >= rs_start_idx)
        go_to_cus   = (destination >= cus_start_idx) & (destination < rs_start_idx)
        go_to_rs_or_cus = ~go_to_depot

        # -------- Reward: 用 normalized 坐标距离亦可（你之后乘回真实 scale）
        if self.env_mode =="eval":
            self.reward = -dist_display
        elif self.env_mode == "train":
            # We will update later
            self.reward = -dist_display
        else:
            raise ValueError("Unknown Mode : {}".format(self.env_mode))

        # -------- Load Update (normalized) --------
        self.load[go_to_depot] = 0.0
        self.load[go_to_rs_or_cus] += self.demands[destination[go_to_rs_or_cus]]

        # -------- Time Update (全部 normalized) --------
        # 1) travel
        breakpoint()
        self.current_time[go_to_rs_or_cus] += self.travel_time[self.last, destination][go_to_rs_or_cus]

        # 2) wait for customer TW open
        # current_time 与 time_window[:,0] 都在 [0,1]
        self.current_time[go_to_cus] = np.maximum(
            self.current_time[go_to_cus],
            self.time_window[destination[go_to_cus], 0]
        )

        # 3) service
        self.current_time += self.service_time[destination]

        # 4) charging at RS ：假设充到满电
        # self.battery 在这里是 "已消耗的 SoC 比例"，0 = 刚从 depot/RS 出发，1 = 已烧满
        # 剩余 SoC = 1 - self.battery
        if go_to_rs.any():
            remaining_soc = 1.0 - self.battery[go_to_rs]
            charging_time_norm = remaining_soc * self.charging_beta  # beta 已经是 "1 SoC 所需 normalized 时间"
            self.current_time[go_to_rs] += charging_time_norm
            self.battery[go_to_rs] = 0.0  # 重新视为 "消耗=0"

        # -------- Battery Update (edge_energy normalized) --------
        self.battery[go_to_depot] = 0.0
        # RS 的 battery 已经在上面充满归零了，这里只更新 customer 消耗
        self.battery[go_to_cus] += self.edge_energy[self.last, destination][go_to_cus]

        # -------- Visited & last node --------
        self.visited[np.arange(self.n_traj), destination] = True
        self.last = destination

    def step(self, action):
        # return last state after done,
        # for the sake of PPO's abuse of ff on done observation
        # see https://github.com/opendilab/DI-engine/issues/497
        # Not needed for CleanRL
        # if self.done.all():
        #     return self.state, self.reward, self.done, self.info
        
        return self._STEP(action)

    def reset(self):
        return self._RESET()
    