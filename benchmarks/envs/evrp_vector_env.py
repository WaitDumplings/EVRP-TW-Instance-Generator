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
                                         plot_instances=plot_instances)
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
        # self._dist_matrix()
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

        self.prev = self.prev[index]
        self.visited = self.visited[index]
        self.done = self.done[index]

    def _update_mask(self):
        # Need to update
        action_mask = ~self.visited

        self.mask = action_mask
        return action_mask

    def _RESET(self, env = None):
        self.visited = np.zeros((self.n_traj, self.cus_num + self.rs_num + 1), dtype=bool)
        self.visited[:, 0] = True
        self.num_steps = 0
        self.traj = []
        self.restrictions = np.empty((0, 2), dtype=int)
        self.last = np.zeros(self.n_traj, dtype=int)  # idx of the cur elem
        self.prev = np.zeros(self.n_traj, dtype=int)  # idx of the prev elem
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
        self.prev = self.last.copy()
        self.last = destination.copy()
        cus_start_idx = 1
        rs_start_idx = 1 + self.cus_num

        go_to_depot = (destination == 0)
        go_to_rs = (destination >= rs_start_idx)
        go_to_cus = (cus_start_idx <= destination) & (destination < rs_start_idx)

        ################################ Reward Update ################################
        if self.env_mode == "eval":
            self.reward = -dist
        elif self.env_mode == "train":
            self.reward = -dist
        else:
            raise ValueError("Unknown Mode : {}".format(self.eval_partition))

        ################################ Load Update ################################
        self.load[destination == 0] = 0
        self.load[destination > 0] += self.demands[destination[destination > 0]]

        ################################ Time Update ################################
        self.current_time[destination == 0] = 0

        # arrive and serve time
        # Start from node i -> node j (travel: serve for node i, travel from i -> j)
        # arrive time (current_time = service time + travel time)
        self.current_time[destination > 0] += (self.dist_matrix[self.prev, destination] / self.velocity)[destination > 0]

        # arrive time >= time_window start time
        self.current_time[go_to_cus] = np.max((self.current_time[go_to_cus], self.time_window[destination[go_to_cus], 0]), axis=0)
        
        # end_service_time
        self.current_time += self.service_time[destination]

        # charging time at RS (Need TO Update)
        self.current_time[go_to_rs] += (self.battery + self.battery_matrix[self.prev, destination])[go_to_rs] / self.charging_power

        ################################ Demand Update ################################
        # self.demands_with_depot[destination[destination > 0] - 1] = 0
        self.load += self.demands[destination]
        self.visited[np.arange(self.n_traj), destination] = True
        
        ################################ Battery Update ################################
        # less_than = destination < (self.rs_num + 1)
        # self.battery[less_than] -= self.battery_matrix[self.prev, destination][less_than]

        self.battery[go_to_depot] = 0
        self.battery[go_to_rs] = 0
        self.battery[go_to_cus] += (self.dist_matrix[self.prev, destination] / self.energy_consum_rate)[go_to_cus]

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
    