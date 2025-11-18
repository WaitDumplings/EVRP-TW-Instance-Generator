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
        self.dataset = InstanceGenerator(self.config_path)

        # 
        self.cus_nodes         = None
        self.rs_nodes          = None

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
        self._dist_matrix()
        obs = {"cus_loc": self.nodes[self.rs_nodes + 1:]}  # n x 2 array
        obs["depot_loc"] = self.nodes[0]
        obs["rs_loc"] = self.nodes[1:self.rs_nodes + 1]
        obs["demand"] = self.demands
        obs["time_window"] = self.time_window
        
        # Dynamic
        obs["action_mask"] = self._update_mask() if update_mask else self.mask
        obs["last_node_idx"] = self.last
        obs["current_load"] = self.load
        obs["current_battery"] = self.battery
        obs["current_time"] = self.current_time
        obs["instance_mask"] = self.instance_mask
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
        # visited_mask (Done) | one_step_ahead_mask | RS_mask (Done) | energy_mask (Done) | time_window_mask |
        # Only allow to visit unvisited nodes
        action_mask = ~self.visited
        # Depot & RS can always be visited but not visit itself
        action_mask[:, :(self.rs_nodes + 1)] |= True
        action_mask[range(self.n_traj), self.last] = False
        action_mask[(self.prev == 0) & (self.last > 0) & (self.last < self.rs_nodes + 1), 1:self.rs_nodes + 1] = False

        # 1️⃣ 获取所有 `self.last == 0` 的行索引 (valid_row0s)
        valid_rows = np.where(self.last == 0)[0]  # 形状 (N,)

        # 2️⃣ 获取 `self.restrictions` 中匹配 `valid_rows` 的行
        mask_rows = np.isin(self.restrictions[:, 0], valid_rows)  # 形状 (M,), 选出符合的索引

        # 3️⃣ 直接批量更新 action_mask
        action_mask[self.restrictions[mask_rows, 0], self.restrictions[mask_rows, 1]] = False

        # not allow visit nodes with higher demand than capacity
        action_mask &= self.demands[None, :] <= (self.load[:, None] + 1e-10)  # to handle the floating point subtraction precision
        
        # not allow visit nodes with higher battery than capacity
        action_mask &= self.battery_matrix[self.last,:] <= (self.battery.reshape(-1, 1) + 1e-10)

        # time_window mask
        action_mask[:, 1+self.rs_nodes:] &= ((self.current_time.reshape(-1, 1) + (self.dist_matrix[self.last, :] / self.velocity)) < self.time_window[:, 1].reshape(1, -1))[:, 1+self.rs_nodes:]

        # one step ahead
        action_mask &= (self.battery_matrix[self.last,:] + self.min_battery_value.reshape(1, -1)) <= (self.battery.reshape(-1, 1) + 1e-10)

        # return depot when cannot serve any customers (prerequesiti: enough battery to the depot)
        back_to_depot_when_compelte = ((self.battery_matrix[self.last, 0] < self.battery) & ((np.sum(self.visited[:, self.rs_nodes + 1:]==True,axis=-1) == self.cus_nodes)))    
        action_mask[back_to_depot_when_compelte, 0]  = True
        action_mask[back_to_depot_when_compelte, 1:] = False

        # if visit a RS, cannot goto other RSs / Depot (iself.demands[None, :] <= (self.load[:, None] + 1e-10)f we still have chance to visit other customers)
        if self.condition > 1:
            action_mask[((~self.is_all_visited()) & (self.last < self.rs_nodes + 1) & (action_mask[:,0])), 1:self.rs_nodes + 1] = False
            action_mask[((self.current_time > 1.0) & (self.last < self.rs_nodes + 1) & (action_mask[:,0])), 1:self.rs_nodes + 1] = False

        # 计算 restriction mask
        restric_mask = (self.prev == 0) & (self.last > 0) & (self.last < self.rs_nodes + 1) & (action_mask[:, 0]) & (np.sum(action_mask[:, 1:], axis=-1) == 0)

        # 获取符合条件的索引 (行索引)
        index = np.where(restric_mask)[0]  # 取出行索引

        # 生成 [[row_index, self.last[row_index]]] 形式的 NumPy 数组
        restricted_values = np.column_stack((index, self.last[index]))  # Shape: (N, 2)

        # 只有在 restricted_values 不为空时才更新 self.restrictions
        if restricted_values.size > 0:
            self.restrictions = np.vstack((self.restrictions, restricted_values))

        # if all customers have been visited, only stay at the depot 
        action_mask[(self.last == 0) & (self.is_all_visited()), 1:] = False
        action_mask[(self.last == 0) & (self.is_all_visited()), 0] = True
        
        rows_all_false = np.all(action_mask == False, axis=1).any()

        if rows_all_false:
            print("Abnormal: ", np.where(np.all(action_mask == False, axis=1)))
            action_mask[np.where(np.all(action_mask == False, axis=1))[0], 0] = True

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
        self.serve_number = np.zeros(self.n_traj, dtype=int)
        self.route_number = np.zeros(self.n_traj, dtype=int)
        self.rs_travel_count = np.zeros((self.n_traj, self.rs_nodes, self.rs_nodes))

        if self.eval_data:
            self._load_orders()         # eval dataset
        else:
            self._generate_solomon_data() # train dataset

        self.state = self._update_state()
        self.info = {}
        self.done = np.array([False] * self.n_traj)

        if env:
            self._reset_env(env)

        return self.state

    def _reset_env(self, env):
        raise NotImplementedError

    def _generate_solomon_data(self):
        data, data_env = self.dataset._generate_instances()
        self.nodes = np.concatenate((data["depot_loc"], data['rs_loc'], data["cus_loc"]))
        self.demands = data["demand"]
        self.time_window = data["time_window"]
        self.energy_consum_rate = data["energy_consumption"]
        self.b_s = data["battery_capacity"]
        self.parameter_setting(velocity_base = data["velocity_base"],
                                rscc_base = data["energy_consumption"],
                                rs_speed_base = data["charging_rate"],
                                time_window_limit = data["max_time"],
                                pos_scale = self.pos_scale,
                                battery_scale = data["battery_capacity"],
                                service_time_base = data["service_time"])
        self.cus_nodes = data["cus_loc"].shape[0]
        self.rs_nodes = data['rs_loc'].shape[0]
        self.type = data['types']
        self.instance_mask = data['instance_mask']

    def _load_orders(self):
        data = EVRPTWDataset[self.eval_partition, self.cus_nodes, self.eval_data_idx]
        self.nodes = np.concatenate((data["depot_loc"], data['rs_loc'], data["cus_loc"]))
        self.demands = data["demand"]
        self.time_window = data["time_window"]
        self.energy_consum_rate = data["energy_consumption"]
        self.b_s = data["battery_capacity"]
        self.cus_nodes = data["cus_loc"].shape[0]
        self.rs_nodes = data['rs_loc'].shape[0]
        self.type = data['types']
        self.instance_mask = data['instance_mask']
    
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
    