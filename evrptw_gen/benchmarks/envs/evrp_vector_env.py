import gym
import numpy as np
from gym import spaces
import re
import torch
import os
import json
from pathlib import Path

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

    def _sync_state(self, index):
        self.mask = self.mask[index]
        self.last = self.last[index]
        self.load = self.load[index]
        self.battery = self.battery[index]
        self.current_time = self.current_time[index]
        self.state = self._update_state(update_mask=False)

        self.visited = self.visited[index]
        self.done = self.done[index]

    def _update_mask(self):
        # Need to update

        # cus_start_idx
        cus_start_idx = 1
        rs_start_idx = 1 + self.cus_num

        # 有哪些Mask需要更新？
        # (1) 访问过的customer不能在访问
        # (2) Demand 超过载重不能访问
        # (3) battery 不足不能访问
        # (4) time_window 超过不能访问
        # (5)Future Fesibility Pruning

        # (1) 访问过的customer不能在访问
        action_mask = ~self.visited

        # (2) Demand 超过载重不能访问
        load_mask = (self.load[:, None] + self.demands[None, :]) <= self.loading_capacity
        action_mask = action_mask & load_mask

        # (3) battery 不足不能访问
        battery_need = self.battery[:, None] + self.battery_matrix[self.last, :]  # n_traj x (1 + cus_num + rs_num)
        battery_mask = (battery_need <= self.battery_capacity)
        action_mask = action_mask & battery_mask

        # (4) time_window 超过不能访问
        time_after_arrival = self.current_time[:, None] + (self.dist_matrix[self.last, :] / self.velocity)[:, :]  # n_traj x (1 + cus_num + rs_num)
        time_mask = (time_after_arrival <= self.time_window[:, 1][None, :])
        action_mask = action_mask & time_mask

        # (5)Future Fesibility Pruning
        cus_idx = np.arange(cus_start_idx, rs_start_idx)  # 所有 customer 下标
        rs_cols = np.r_[0, rs_start_idx:self.dist_matrix.shape[1]]  # depot + 所有 RS
        n_cus = len(cus_idx)
        n_rs_plus_depot = len(rs_cols)

        # 1) 当前点 -> customer
        time_to_customer = self.dist_matrix[self.last[:, None], cus_idx] / self.velocity  # (n_traj, n_cus)
        # 或者：time_to_customer = self.dist_matrix[self.last, cus_start_idx:rs_start_idx] / self.velocity

        # 电量到达 customer 时
        battery_at_customer = self.battery[:, None] + time_to_customer * self.energy_consum_rate  # (n_traj, n_cus)

        # 已经在 (4) 里保证到达不超过 TW close，这里简单用“到达 + 服务时间”
        # 但是应该从 max(tw[:, 0], self.current_time[:, None] + time_to_customer) 开始算起才对
        # 所有 customer 的 time window 左端点 [open]
        tw_open = self.time_window[cus_start_idx:rs_start_idx, 0][None, :]   # (1, n_cus)

        # 当前到达 customer 的时间
        arrival_time = self.current_time[:, None] + time_to_customer        # (n_traj, n_cus)

        # 逐元素取 max：到得早就等到 tw_open，晚到就直接开始服务
        start_service_time = np.maximum(arrival_time, tw_open)              # (n_traj, n_cus)

        # 离开 customer 的时间 = 开始服务时间 + 服务时长
        time_after_service = start_service_time + self.service_time[cus_start_idx:rs_start_idx][None, :]  # (n_traj, n_cus)

        # 2) customer -> RS/depot 的最短时间
        # dist_matrix[customer, rs_cols]：shape (n_cus, 1+n_rs)
        time_cust_to_rs = self.dist_matrix[np.ix_(cus_idx, rs_cols)] / self.velocity  # (n_cus, n_rs_plus_depot)

        # 扩成 3D： (n_traj, n_cus, n_rs_plus_depot)
        time_cust_to_rs = time_cust_to_rs[None, :, :]

        # 3) customer 到达时的电量/时间扩成 3D
        battery_at_customer_3d = battery_at_customer[:, :, None]        # (n_traj, n_cus, 1)
        time_after_service_3d = time_after_service[:, :, None]          # (n_traj, n_cus, 1)

        # 4) 到达 RS/depot 时的电量 & 时间
        battery_at_rs = battery_at_customer_3d + time_cust_to_rs * self.energy_consum_rate  # (n_traj, n_cus, n_rs_plus_depot)
        time_at_rs = time_after_service_3d + time_cust_to_rs                               # (n_traj, n_cus, n_rs_plus_depot)

        # 5) 在 RS 充电，然后回 depot
        RS_time_to_depot = np.concatenate([np.zeros((1,)), self.RS_time_to_depot])  # (n_rs_plus_depot,)
        RS_time_to_depot_3d = RS_time_to_depot[None, :, None]                       # (1,1,n_rs_plus_depot)

        # 简单起见：充满电需要的时间（或你可以用别的策略）
        time_charge_at_rs = (battery_at_rs / self.charging_power)
        time_charge_at_rs[:, :, 0] = 0.0  # depot 不充电
        total_finish_time = time_at_rs + time_charge_at_rs + RS_time_to_depot_3d  # (n_traj, n_cus, n_rs_plus_depot)

        # 6) 可行性判断
        time_feasible = (total_finish_time <= self.instance_max_time)
        battery_feasible = (battery_at_rs <= self.battery_capacity)

        feasible = time_feasible & battery_feasible  # (n_traj, n_cus, n_rs_plus_depot)

        # 对每个 customer，看是否存在至少一个 RS/depot 可行
        FFP_cus_mask = feasible.any(axis=2)  # (n_traj, n_cus)

        # 只更新 customer 的 action_mask
        action_mask[:, cus_start_idx:rs_start_idx] &= FFP_cus_mask

        self.mask = action_mask
        return action_mask

    def _RESET(self, env = None):
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
            self._eval_data_generate(env)
        elif self.env_mode == "train":
            # training mode
            self._train_data_generate(env)
        else:
            raise ValueError("Unknown mode : {}".format(self.env_mode))

        self.state = self._update_state()
        self.info = {}
        self.done = np.array([False] * self.n_traj)
        
        return self.state

    def _train_data_generate(self, env = None):
        if env:
            self._update_env(env)
        
        context = self.dataset.generate_tensors()

        RS_time_to_depot = context['env']['cs_time_to_depot']
        self.RS_time_to_depot = np.concatenate([np.zeros((1,)), RS_time_to_depot])

        self.depot_num = context["depot"].shape[0]
        self.cus_num = context['customers'].shape[0]
        self.rs_num = context['charging_stations'].shape[0]

        # Normalizations
        self._normalizations(context)

    
    def _normalizations(self, context):         
        # Node Information
        nodes = np.concatenate((context["depot"], context['customers'], context["charging_stations"])).astype(np.float32)
        positions = np.zeros_like(nodes)

        demands = context["demands"].astype(np.float32)
        time_window = context["tw"].astype(np.float32)
        service_time = context["service_time"].astype(np.float32)

        # Graph Information
        data = context['env']
        energy_consum_rate = data["consumption_per_distance"]
        b_s = data["battery_capacity"]
        velocity = data["speed"]
        loading_capacity = data['loading_capacity']
        charging_power = data["charging_speed"] * data['charging_efficiency']
        instance_max_time = data["instance_endTime"]
        instance_working_start_time = data['working_startTime']
        instance_working_end_time = data['working_endTime']
        pos_scale = data["area_size"]

        # Node Normalization:
        # position rescale:
        x_scale = pos_scale[0][1] - pos_scale[0][0]
        y_scale = pos_scale[1][1] - pos_scale[1][0]
        positions[:, 0] =  (nodes[:, 0] - pos_scale[0][0]) / x_scale
        positions[:, 1] = (nodes[:, 1] - pos_scale[1][0]) / y_scale

        # demand rescale
        demands_norm_cus = demands / loading_capacity
        demands_norm_depot = np.zeros((1,))
        demands_norm_rs = np.zeros((self.rs_num,))

        # time rescale
        working_time_span = instance_working_end_time - instance_working_start_time
        time_window_norm_cus = np.clip((time_window - instance_working_start_time) / working_time_span, 0.0, 1.0)
        time_window_norm_depot = np.array([[0,1]])
        time_window_norm_rs = np.array([[0,1] * self.rs_num]).reshape(self.rs_num, 2)

        service_time_norm_cus = np.clip(service_time / working_time_span, 0.0, 1.0)
        service_time_norm_depot = np.zeros((1,))
        service_time_norm_rs = np.zeros((self.rs_num,))

        # energy
        energy_consum_rate = energy_consum_rate / b_s
        charging_power = charging_power / b_s

        # update
        self.nodes_raw = nodes
        self.nodes = positions
        self.depot_nodes = positions[0:1]
        self.cus_nodes = positions[1 : 1 + self.cus_num]
        self.rs_nodes = positions[1 + self.cus_num : 1 + self.cus_num + self.rs_num]
       
        self.demands = np.concatenate((demands_norm_depot, demands_norm_cus, demands_norm_rs))
        self.time_window = np.concatenate((time_window_norm_depot, time_window_norm_cus, time_window_norm_rs))
        self.service_time = np.concatenate((service_time_norm_depot, service_time_norm_cus, service_time_norm_rs))
        self.energy_consum_rate = energy_consum_rate
        self.charging_power = charging_power
        self.velocity = velocity
        self.instance_max_time = instance_max_time
        self.loading_capacity = loading_capacity
        self.battery_capacity = b_s

        self.dist_matrix = gen_dist_matrix(self.nodes, self.nodes)
        self.battery_matrix = self.dist_matrix  / self.energy_consum_rate

    def _eval_data_generate(self, env = None):
        raise NotImplementedError("Not Implement Yet!")
    
    def _go_to(self, destination):
        dest_node = self.nodes[destination]
        dist = fun_dist(dest_node, self.nodes[self.last]) 
        cus_start_idx = 1
        rs_start_idx = 1 + self.cus_num

        go_to_depot = (destination == 0)
        go_to_rs = (destination >= rs_start_idx)
        go_to_cus = (destination >= cus_start_idx) & (destination < rs_start_idx)
        go_to_rs_or_cus = ~go_to_depot

        ################################ Reward Update ################################
        if self.env_mode == "eval":
            self.reward = -dist
        elif self.env_mode == "train":
            self.reward = -dist
        else:
            raise ValueError("Unknown Mode : {}".format(self.eval_partition))

        ################################ Load Update ################################
        # Go to Depot -> unload all
        self.load[go_to_depot] = 0

        # Go to Customer / RS -> load demand (demands at RS is 0)
        self.load[go_to_rs_or_cus] += self.demands[destination[go_to_rs_or_cus]]

        ################################ Time Update ################################
        self.current_time[go_to_depot] = 0
        self.current_time[go_to_rs_or_cus] += (self.dist_matrix[self.last, destination] / self.velocity)[go_to_rs_or_cus]

        # arrive time >= time_window start time
        self.current_time[go_to_cus] = np.max((self.current_time[go_to_cus], self.time_window[destination[go_to_cus], 0]), axis=0)
        
        # end_service_time
        self.current_time += self.service_time[destination]

        # charging time at RS (Need TO Update)
        self.current_time[go_to_rs] += (self.battery + self.battery_matrix[self.last, destination])[go_to_rs] / self.charging_power

        ################################ Battery Update ################################
        self.battery[go_to_depot] = 0
        self.battery[go_to_rs] = 0
        self.battery[go_to_cus] += (self.dist_matrix[self.last, destination] / self.energy_consum_rate)[go_to_cus]

        ################################ Visit Node Update ############################
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

    def reset(self, env = None):
        return self._RESET()
    