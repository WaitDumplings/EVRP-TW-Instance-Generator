import gym
import numpy as np
from gym import spaces
import re
import torch
import os

from evrptw_gen.generator import InstanceGenerator

def assign_env_config(self, kwargs):
    """
    Set self.key = value, for each key in kwargs
    """
    for key, value in kwargs.items():
        setattr(self, key, value)


def dist(loc1, loc2):
    return ((loc1[:, 0] - loc2[:, 0]) ** 2 + (loc1[:, 1] - loc2[:, 1]) ** 2) ** 0.5

import json
from pathlib import Path
import gym

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
        self.cus_nodes         = config_data.get('num_customers', None)
        self.rs_nodes          = config_data.get('num_charging_stations', None)

        if not self.cus_nodes or not self.rs_nodes:
            raise ValueError("customer number of charging station number is not predefined!")

        # if eval_data==True, load from 'test' set, the '0'th data
        self.eval_data = False
        self.eval_partition = "test"
        self.eval_data_idx = 0

        assign_env_config(self, kwargs)

        obs_dict = {"cus_loc": spaces.Box(low=0, high=1, shape=(self.cus_nodes, 2))}
        obs_dict["depot_loc"] = spaces.Box(low=0, high=1, shape=(2,))
        obs_dict["rs_loc"] = spaces.Box(low = 0, high=1, shape=(self.rs_nodes, 2))
        obs_dict["demand"] = spaces.Box(low=0, high=1, shape=(self.cus_nodes + 1 + self.rs_nodes,))
        obs_dict["time_window"] = spaces.Box(low = 0, high=1, shape=(self.cus_nodes + 1 + self.rs_nodes, 2))
        obs_dict["action_mask"] = spaces.MultiBinary(
            [self.n_traj, self.cus_nodes + self.rs_nodes + 1]
        )  # 1: OK, 0: cannot go
        obs_dict["last_node_idx"] = spaces.MultiDiscrete([self.cus_nodes + 1] * self.n_traj)
        obs_dict["current_load"] = spaces.Box(low=0, high=1, shape=(self.n_traj,))
        obs_dict["current_battery"] = spaces.Box(low=0, high=1, shape=(self.n_traj,))
        obs_dict["current_time"] = spaces.Box(low=0, high=1, shape=(self.n_traj,))
        obs_dict["instance_mask"] = spaces.Box(low=0, high=1, shape=(self.cus_nodes + self.rs_nodes + 1,), dtype=bool)

        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.MultiDiscrete([self.rs_nodes + self.cus_nodes + 1] * self.n_traj)
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
        # assumes no repetition in the first `cus_nodes` steps
        return self.visited[:, (1 + self.rs_nodes):].all(axis=1)

    def _update_state(self, update_mask=True):

        # arrangement: [depot, customers1, customer2, ... , cs1, cs2, ...]
        obs = {"cus_loc": self.nodes[1:1 + self.cus_nodes, :]}  # n x 2 array
        obs["depot_loc"] = self.nodes[0]
        obs["rs_loc"] = self.nodes[1 + self.cus_nodes:, :]
        obs["demand"] = self.demands
        obs["time_window"] = self.time_window
        
        # Dynamic
        obs["action_mask"] = self._update_mask() if update_mask else self.mask
        obs["last_node_idx"] = self.last
        obs["current_load"] = self.load
        obs["current_battery"] = self.battery
        obs["current_time"] = self.current_time
        # obs["instance_mask"] = self.instance_mask

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
        action_mask = ~self.visited

        self.mask = action_mask
        return action_mask

    def _RESET(self, env = None):
        self.visited = np.zeros((self.n_traj, self.cus_nodes + self.rs_nodes + 1), dtype=bool)
        self.visited[:, 0] = True
        self.num_steps = 0
        self.traj = []
        self.restrictions = np.empty((0, 2), dtype=int)
        self.last = np.zeros(self.n_traj, dtype=int)  # idx of the cur elem
        self.prev = np.zeros(self.n_traj, dtype=int)  # idx of the prev elem
        self.load = np.ones(self.n_traj, dtype=float)  # current load
        self.battery = np.ones(self.n_traj, dtype=float)  # current battery
        self.current_time = np.zeros(self.n_traj, dtype=float)  # current battery

        if self.eval_data:
            # evaluation mode
            self._eval_data_generate(env)
        else:
            # training mode
            self._train_data_generate(env)

        self.state = self._update_state()
        self.info = {}
        self.done = np.array([False] * self.n_traj)
        
        return self.state

    def _train_data_generate(self, env = None):
        if env:
            self._update_env(env)
        
        context = self.dataset.generate_tensors()
        self.depot_nodes = context["depot"].shape[0]
        self.cus_nodes = context['customers'].shape[0]
        self.rs_nodes = context['charging_stations'].shape[0]

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
        demands_norm = demands / loading_capacity

        # time rescale
        working_time_span = instance_working_end_time - instance_working_start_time
        time_window_norm = np.clip((time_window - instance_working_start_time) / working_time_span, 0.0, 1.0)
        service_time_norm = np.clip(service_time / working_time_span, 0.0, 1.0)

        # energy
        energy_consum_rate = energy_consum_rate / b_s
        charging_power = charging_power / b_s

        # update
        self.nodes_raw = nodes
        self.nodes = positions
        self.demands = demands_norm
        self.time_window = time_window_norm
        self.service_time = service_time_norm
        self.energy_consum_rate = energy_consum_rate
        self.charging_power = charging_power
        self.velocity = velocity
        self.instance_max_time = instance_max_time
        self.loading_capacity = loading_capacity
        self.battery_capacity = b_s

    def _eval_data_generate(self, env = None):
        raise NotImplementedError("Not Implement Yet!")
    
    def _go_to(self, destination):
        # destination : 0 ~ N (N = 1(depot) + m(rs nodes) + n(customer nodes))
        # self.traj.append(destination)
        dest_node = self.nodes[destination]

        self.serve_number[destination >= self.rs_nodes + 1] += 1
        self.route_number[(self.prev==0) & (destination>0)] += 1

        dist = self.cost(dest_node, self.nodes[self.last]) 
        self.prev = self.last.copy()
        self.last = destination.copy()

        if self.eval_partition == "eval":
            # wait time + service time
            # self.obj_reward[obj_go_to_Customer] -= np.max((self.dist_matrix[self.prev, destination] / self.velocity, self.time_window[destination][:,0]), axis=0)[obj_go_to_Customer] + self.service_time
            # self.obj_reward[obj_go_to_Depot] -= (self.dist_matrix[self.prev, destination] / self.velocity)[obj_go_to_Depot]
            # self.obj_reward[obj_go_to_RS] -= ((self.dist_matrix[self.prev, destination] / self.velocity) + (1 - (self.battery - self.battery_matrix[self.prev, destination]) / self.rs_speed))[obj_go_to_RS]
            # self.obj_reward*= self.dist_fee
            # self.reward = self.obj_reward
            self.reward = dist
        else:
            # === RS travel count update ===
            penalty = np.zeros_like(self.prev, dtype=np.float32)  # shape: (200,)
            if self.condition  == 3:
                mask = (self.prev >= 1) & (self.prev <= self.rs_nodes) & \
                    (self.last >= 1) & (self.last <= self.rs_nodes)
                penalty = np.zeros_like(self.prev, dtype=np.float32)  # shape: (200,)
                penalty[mask] = -100
            elif self.condition == 2:
                valid_idx = (self.prev > 0) & (self.prev <= self.rs_nodes) & (self.last > 0) & (self.last <= self.rs_nodes)
                if (valid_idx.any()):
                    prev_idx = self.prev[valid_idx] - 1
                    last_idx = self.last[valid_idx] - 1
                    self.rs_travel_count[valid_idx, prev_idx, last_idx] += 1
                    mask = self.rs_travel_count[valid_idx,prev_idx, last_idx] > 1
                    penalty[valid_idx][mask] -= 100

            # Update Reward:
            new_route = (self.prev == 0) & (destination > 0)
            go_to_Depot = (self.prev > 0) & (destination == 0)
            go_to_RS = (destination < self.rs_nodes + 1) & (destination > 0)
            go_to_RS_low_SoC = (self.battery < 0.3) & (self.prev >= self.rs_nodes + 1) & (destination > 0)
            go_to_RS_low_SoC_large_capacity = go_to_RS_low_SoC & (self.load > 0.3)
            go_to_Depot_with_non_serve = go_to_Depot & (self.serve_number == 0)
            go_to_Depot_with_one_serve = go_to_Depot & (self.serve_number == 1)
            go_to_Depot_with_serve = go_to_Depot & (self.serve_number > 2)
            go_to_Customer = (self.prev >= self.rs_nodes + 1) & (destination >= self.rs_nodes + 1)

            obj_go_to_Customer = (destination > self.rs_nodes)
            obj_go_to_RS = (destination < self.rs_nodes + 1) & (destination > 0)
            obj_go_to_Depot = (destination == 0)

            # self.obj_reward = dist * 0
            new_route_penalty = np.sum(2 * self.dist_matrix[0, self.rs_nodes+1:])

            # obj_reward
            # self.obj_reward[obj_go_to_Customer] -= np.max((self.dist_matrix[self.prev, destination] / self.velocity, self.time_window[destination][:,0]), axis=0)[obj_go_to_Customer] + self.service_time
            # self.obj_reward[obj_go_to_Depot] -= (self.dist_matrix[self.prev, destination] / self.velocity)[obj_go_to_Depot]
            # self.obj_reward[obj_go_to_RS] -= ((self.dist_matrix[self.prev, destination] / self.velocity) + (1 - (self.battery - self.battery_matrix[self.prev, destination]) / self.rs_speed))[obj_go_to_RS]
            # self.obj_reward *= self.dist_fee
            self.obj_reward = -dist*self.dist_fee

            # # serve reward
            self.serve_reward = np.zeros_like(self.obj_reward)
            # self.serve_reward[go_to_Depot_with_one_serve] -= 0.2
            if self.condition == 3:
                self.serve_reward[go_to_Depot_with_non_serve] -= 1.0
                self.serve_reward[go_to_Depot_with_serve] += self.serve_number[go_to_Depot_with_serve] * 0.1
            elif self.condition == 2:
                self.serve_reward[go_to_Depot_with_non_serve] -= 1.0
                self.serve_reward[go_to_Depot_with_serve] += self.serve_number[go_to_Depot_with_serve] * 0.01
            
            # # rs_cus_reward
            self.rs_reward = np.zeros_like(self.obj_reward)
            if self.condition == 2:
                self.rs_reward[go_to_RS] += 0.01
                self.rs_reward[go_to_RS_low_SoC] += 0.03
                self.rs_reward[go_to_RS_low_SoC_large_capacity] += 0.02

            self.go_to_cus_reward = np.zeros_like(self.obj_reward)
            # self.go_to_cus_reward[go_to_Customer] += 0.01

            self.reward = self.obj_reward + self.serve_reward + self.rs_reward + self.go_to_cus_reward + penalty

            # self.reward[new_route] -= (self.route_number[new_route] - 1) * 0.02

        self.serve_number[destination == 0] = 0

        # load update
        self.load[destination == 0] = 1
        self.load[destination > 0] -= self.demands[destination[destination > 0]]

        self.current_time[destination == 0] = 0
        # arrive and serve time
        # Start from node i -> node j (travel: serve for node i, travel from i -> j)
        # arrive time (current_time = service time + travel time)
        self.current_time[destination > 0] += (self.dist_matrix[self.prev, destination] / self.velocity)[destination > 0]

        # arrive time >= time_window start time
        self.current_time[destination >= self.rs_nodes + 1] = np.max((self.current_time[destination >= self.rs_nodes + 1], self.time_window[destination[destination >= self.rs_nodes + 1], 0]), axis=0)
        
        # end_service_time
        self.current_time[destination >= self.rs_nodes + 1] += self.service_time

        # charging time at RS
        go_to_rs = (destination < self.rs_nodes + 1) & (destination > 0)
        self.current_time[go_to_rs] += (1 - (self.battery[go_to_rs] - self.battery_matrix[self.prev, destination][go_to_rs])) / self.rs_speed

        # self.demands_with_depot[destination[destination > 0] - 1] = 0
        self.visited[np.arange(self.n_traj), destination] = True
        
        # self.battery update
        less_than = destination < (self.rs_nodes + 1)
        # self.battery[less_than] -= self.battery_matrix[self.prev, destination][less_than]

        self.battery[destination < (self.rs_nodes + 1)] = 1
        self.battery[destination >= (self.rs_nodes + 1)] -= self.battery_matrix[self.prev, destination][destination >= (self.rs_nodes + 1)]

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
    