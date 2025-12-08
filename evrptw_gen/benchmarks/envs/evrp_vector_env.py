import gym
import numpy as np
from gym import spaces

from evrptw_gen.generator import InstanceGenerator


def assign_env_config(self, kwargs):
    """
    Set self.key = value for each key in kwargs.
    """
    for key, value in kwargs.items():
        setattr(self, key, value)


def gen_dist_matrix(nodes1, nodes2):
    """
    Compute pairwise Euclidean distance matrix between two sets of nodes.

    Args:
        nodes1: np.ndarray of shape (N1, d)
        nodes2: np.ndarray of shape (N2, d)

    Returns:
        np.ndarray of shape (N1, N2)
    """
    diff = nodes1[:, None, :] - nodes2[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def fun_dist(loc1, loc2):
    """
    Elementwise Euclidean distance between two arrays of coordinates.

    Args:
        loc1: np.ndarray of shape (B, 2)
        loc2: np.ndarray of shape (B, 2)

    Returns:
        np.ndarray of shape (B,)
    """
    return np.sqrt(
        (loc1[:, 0] - loc2[:, 0]) ** 2 +
        (loc1[:, 1] - loc2[:, 1]) ** 2
    )


class EVRPTWVectorEnv(gym.Env):
    """
    Vectorized EVRPTW environment with fully normalized internal dynamics.

    Internal conventions:
        - Time is normalized by T_scale = instance_max_time_abs  => [0, 1].
        - Load is normalized by loading_capacity => in [0, 1].
        - Battery is represented as "consumed SoC fraction" in [0, 1]:
              0.0 => just fully charged (no consumption yet)
              1.0 => completely exhausted
        - edge_energy[i, j] = SoC fraction consumed when traveling i -> j.
        - charging_beta = normalized time needed to charge 1.0 SoC.
    """

    metadata = {"render.modes": []}

    def __init__(self, *args, **kwargs):

        # ====== Configuration ======
        self.terminate = False
        self.config_path = kwargs.get("config_path", None)
        if not self.config_path:
            raise ValueError("config_path to construct env is required!")

        self.n_traj = kwargs.get("n_traj", 100)

        # ====== Load dataset / config ======
        save_path = kwargs.get("save_path", None)
        num_instances = kwargs.get("num_instances", 1)
        plot_instances = kwargs.get("plot_instances", False)

        self.dataset = InstanceGenerator(
            self.config_path,
            save_path=save_path,
            num_instances=num_instances,
            plot_instances=plot_instances,
            kwargs=kwargs,
        )

        config_data = self.dataset.config.data
        self.cus_num = config_data.get("num_customers", None)
        self.rs_num = config_data.get("num_charging_stations", None)

        if not self.cus_num or not self.rs_num:
            raise ValueError(
                "Customer number or charging station number is not predefined!"
            )
        self.env_mode = kwargs.get("env_mode", "train")  # "train" or "eval"
        assign_env_config(self, kwargs)
        # self.snap_shot = {}

        # ====== Observation / Action spaces ======
        obs_dict = {
            "cus_loc": spaces.Box(low=0, high=1, shape=(self.cus_num, 2)),
            "depot_loc": spaces.Box(low=0, high=1, shape=(1, 2)),
            "rs_loc": spaces.Box(low=0, high=1, shape=(self.rs_num, 2)),
            "demand": spaces.Box(
                low=0, high=1, shape=(1 + self.cus_num + self.rs_num,)
            ),
            "time_window": spaces.Box(
                low=0, high=1, shape=(1 + self.cus_num + self.rs_num, 2)
            ),
            "action_mask": spaces.MultiBinary(
                [self.n_traj, self.cus_num + self.rs_num + 1]
            ),  # 1: OK, 0: cannot go
            "last_node_idx": spaces.MultiDiscrete(
                [self.cus_num + self.rs_num + 1] * self.n_traj
            ),
            "current_load": spaces.Box(low=0, high=1, shape=(self.n_traj,)),
            "current_battery": spaces.Box(low=0, high=1, shape=(self.n_traj,)),
            "current_time": spaces.Box(low=0, high=1, shape=(self.n_traj,)),
            "service_time": spaces.Box(
                low=0, high=1, shape=(1 + self.cus_num + self.rs_num,)
            ),
            # capacity scalars are kept for info / logging (can be 1.0 in normalized world)
            "battery_capacity": spaces.Box(low=0, high=np.inf, shape=(1,)),
            "loading_capacity": spaces.Box(low=0, high=np.inf, shape=(1,)),
            "visited_customers_raio": spaces.Box(low=0, high=1, shape=(self.n_traj, 1)),
            "remain_feasible_customers_raio": spaces.Box(low=0, high=1, shape=(self.n_traj, 1)),
        }

        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.MultiDiscrete(
            [self.rs_num + self.cus_num + 1] * self.n_traj
        )
        self.reward_space = None

        self.reset()

    # ======================================================================
    #  Gym API
    # ======================================================================

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        """
        Reset environment and generate a new instance (train/eval).
        All internal dynamic states are initialized in normalized space.
        """
        self.num_steps = 0
        self.traj = []
        self.restrictions = np.empty((0, 2), dtype=int)
        self.info = {}

        # 1) Static data (instance) generation + normalization
        if self.env_mode == "eval":
            self._eval_data_generate(mode = self.eval_mode)
        elif self.env_mode == "train":
            self._train_data_generate()
        else:
            raise ValueError(f"Unknown mode: {self.env_mode}")

        # 2) Dynamic state initialization (normalized)
        #    arrangement: [depot, customers..., RS...]
        self.visited = np.zeros(
            (self.n_traj, self.cus_num + self.rs_num + 1), dtype=bool
        )
        self.visited[:, 0] = True  # depot initially visited (start here)

        self.last = np.zeros(self.n_traj, dtype=int)    # current node index
        self.load = np.zeros(self.n_traj, dtype=float)  # current load (0..1)
        # battery = consumed SoC fraction in [0,1], start with 0 consumption (full battery)
        self.battery = np.zeros(self.n_traj, dtype=float)
        # current_time in [0,1], start at 0
        self.current_time = np.zeros(self.n_traj, dtype=float)

        self.done = np.zeros(self.n_traj, dtype=bool)
        self.finish = np.zeros(self.n_traj, dtype=bool)

        self.state = self._update_state()
        return self.state

    def step(self, action):
        """
        One vectorized step.

        Args:
            action: np.ndarray of shape (n_traj,) with node indices.

        Returns:
            obs, reward, done, info
        """
        return self._STEP(action)

    # ======================================================================
    #  Internal step / reset helpers
    # ======================================================================

    def _STEP(self, action):
        self._go_to(action)  # Go to node 'action', update reward & dynamic states
        self.num_steps += 1
        self.state = self._update_state()

        # terminate trajectory when returning to depot after all customers visited
        if self.terminate:
            self.done = np.ones_like(self.done, dtype=bool)
        else:
            self.done = (action == 0) & self.is_all_visited()

        if not self.terminate:
            bonus = 10.0
            self.reward[self.done & ~self.finish] += bonus  # bonus for finishing
        self.finish = self.finish | self.done

        if self.terminate:
            self.terminate = False
        # self.snap_shot['last'] = self.last.copy()
        # self.snap_shot['battery'] = self.battery.copy()
        # self.snap_shot['current_time'] = self.current_time.copy()
        # self.snap_shot['load'] = self.load.copy()
        # self.snap_shot['visited'] = self.visited.copy()

        return self.state, self.reward, self.done, self.info

    def is_all_visited(self):
        """
        Check if all customers (not depot, not RSs) have been visited.
        Node order: [0: depot, 1..cus_num: customers, cus_num+1..: RSs]
        """
        cus_start = 1
        rs_start = 1 + self.cus_num
        return self.visited[:, cus_start:rs_start].all(axis=1)

    def _update_state(self, update_mask=True):
        """
        Build current observation dict from internal normalized state.
        """
        obs = {
            "cus_loc": self.cus_nodes,      # (n_cus, 2)
            "depot_loc": self.depot_nodes,  # (1, 2)
            "rs_loc": self.rs_nodes,        # (n_rs, 2)
            "demand": self.demands,         # (1+n_cus+n_rs,)
            "time_window": self.time_window,
            "action_mask": self._update_mask() if update_mask else self.mask,
            "last_node_idx": self.last,
            "current_load": self.load,
            "current_battery": self.battery,
            "current_time": self.current_time,
            "service_time": self.service_time,
            "battery_capacity": np.array(self.battery_capacity).reshape(-1),
            "loading_capacity": np.array(self.loading_capacity).reshape(-1),
            "visited_customers_raio": (self.visited[:, 1:1+self.cus_num]).sum(axis=1, keepdims=True)/self.cus_num,
            "remain_feasible_customers_raio": ((~self.visited[:, 1:1+self.cus_num]) & (self.mask[:, 1:1+self.cus_num])).sum(axis=1, keepdims=True)/self.cus_num,
        }
        return obs

    # ======================================================================
    #  Mask (feasibility + FFP)
    # ======================================================================

    def _update_mask(self):
        """
        Compute action mask based on:
            (1) visited customers
            (2) load capacity
            (3) battery capacity
            (4) time window feasibility
            (5) Future Feasibility Pruning (FFP)
        All checks are in normalized units.
        """
        cus_start_idx = 1
        rs_start_idx = 1 + self.cus_num

        # False: cannot go, True: can go
        # base: cannot revisit nodes marked visited
        action_mask = ~self.visited

        # Depot & RSs can always be visited
        action_mask[:, 0] = True
        action_mask[:, rs_start_idx:] = True

        # cannot visit itself
        action_mask[np.arange(self.n_traj), self.last] = False

        # (2) load feasibility: load + demand <= capacity (1.0)
        load_mask = (self.load[:, None] + self.demands[None, :]) <= self.loading_capacity
        action_mask &= load_mask

        # (3) battery feasibility: current consumption + edge consumption <= 1.0
        battery_need = self.battery[:, None] + self.edge_energy[self.last, :]
        battery_mask = battery_need <= self.battery_capacity
        action_mask &= battery_mask

        # (4) time window (arrival <= close)
        time_after_arrival = self.current_time[:, None] + self.travel_time[self.last, :]
        time_mask = time_after_arrival <= self.time_window[:, 1][None, :]
        action_mask &= time_mask

        # (5) Future Feasibility Pruning (FFP): from current -> candidate customer -> some RS/depot -> depot
        cus_idx = np.arange(cus_start_idx, rs_start_idx)   # all customers
        rs_cols = np.r_[0, rs_start_idx:self.travel_time.shape[1]]  # depot + all RS

        # 1) current -> customer
        time_to_customer = self.travel_time[self.last[:, None], cus_idx]         # (n_traj, n_cus)
        energy_to_customer = self.edge_energy[self.last[:, None], cus_idx]      # (n_traj, n_cus)
        battery_at_customer = self.battery[:, None] + energy_to_customer        # (n_traj, n_cus)

        # start service time: max(arrival, TW open)
        tw_open = self.time_window[cus_start_idx:rs_start_idx, 0][None, :]      # (1, n_cus)
        arrival_time = self.current_time[:, None] + time_to_customer           # (n_traj, n_cus)
        start_service_time = np.maximum(arrival_time, tw_open)                 # (n_traj, n_cus)
        time_after_service = start_service_time + self.service_time[cus_start_idx:rs_start_idx][None, :]

        # 2) customer -> RS/depot
        time_cust_to_rs = self.travel_time[np.ix_(cus_idx, rs_cols)][None, :, :]   # (1, n_cus, n_rs+1)
        energy_cust_to_rs = self.edge_energy[np.ix_(cus_idx, rs_cols)][None, :, :] # (1, n_cus, n_rs+1)

        battery_at_customer_3d = battery_at_customer[:, :, None]   # (n_traj, n_cus, 1)
        time_after_service_3d = time_after_service[:, :, None]     # (n_traj, n_cus, 1)

        battery_at_rs = battery_at_customer_3d + energy_cust_to_rs  # (n_traj, n_cus, n_rs+1), consumed SoC
        time_at_rs = time_after_service_3d + time_cust_to_rs        # (n_traj, n_cus, n_rs+1)

        # 3) charging to full then RS/depot -> depot
        # remaining SoC at RS = battery_at_rs
        time_charge_at_rs = battery_at_rs * self.charging_beta
        time_charge_at_rs[:, :, 0] = 0.0  # depot 不充电

        RS_time_to_depot_3d = self.RS_time_to_depot[None, None, :]  # (1, 1, n_rs+1), normalized

        total_finish_time = time_at_rs + time_charge_at_rs + RS_time_to_depot_3d  # normalized
        time_feasible = total_finish_time <= self.instance_max_time              # <= 1.0
        battery_feasible = battery_at_rs <= self.battery_capacity                # consumed SoC <= 1.0

        feasible = time_feasible & battery_feasible
        FFP_cus_mask = feasible.any(axis=2)  # (n_traj, n_cus)

        # only apply FFP on customers
        action_mask[:, cus_start_idx:rs_start_idx] &= FFP_cus_mask

        # (6) FFP on Charging Stations
        # if time Cur Node -> RS -> depot is infeasible, mask RS
        # (self.last, self.battery, self.current_time)
        time_to_rs = self.travel_time[self.last[:, None], rs_cols]         # (n_traj, n_rs+1)
        energy_to_rs = self.edge_energy[self.last[:, None], rs_cols]      # (n_traj, n_rs+1)
        battery_at_rs = self.battery[:, None] + energy_to_rs        # (n_traj, n_rs+1)
        time_at_rs = self.current_time[:, None] + time_to_rs        # (n_traj, n_rs+1)

        # charging to full
        time_charge_at_rs = battery_at_rs * self.charging_beta
        time_charge_at_rs[:, 0] = 0.0  # depot 不充电
        RS_time_to_depot_2d = self.RS_time_to_depot[None, :]  # (1, n_rs+1), normalized
        total_finish_time_rs = time_at_rs + time_charge_at_rs + RS_time_to_depot_2d  # normalized
        rs_time_feasible = total_finish_time_rs <= self.instance_max_time              # <= 1.0
        rs_battery_feasible = battery_at_rs <= self.battery_capacity
        rs_feasible = rs_time_feasible & rs_battery_feasible
        action_mask[:, rs_cols] &= rs_feasible

        # customer visited mission complete
        customer_has_been_visited = self.is_all_visited()
        action_mask[customer_has_been_visited, 0] = True

        # mission complete: cannot go to any other nodes except stay at depot
        customer_has_been_visited_and_at_depot = customer_has_been_visited & (self.last == 0)
        action_mask[customer_has_been_visited_and_at_depot, 1:] = False

        # if action_mask.sum(axis=1).min() == 0:
        #     print("===== DEBUG INFO =====")
        #     print("Warning: some trajectory has no feasible actions!")
        #     idx = np.where(action_mask.sum(axis=1) == 0)
        #     print(idx, "Current location:", self.last[idx])
            
        #     # idx: 29 -> Location: ? -> 113
        #     breakpoint()
        #     # load snap_shot
        #     self.last = self.snap_shot['last'].copy()
        #     self.battery = self.snap_shot['battery'].copy()
        #     self.current_time = self.snap_shot['current_time'].copy()
        #     self.load = self.snap_shot['load'].copy()
        #     self.visited = self.snap_shot['visited'].copy()

            # self.snap_shot
        self.mask = action_mask
        return action_mask

    def _print_matrix(self, array, idx):
        for i in range(len(array)):
            print(array[i][idx], end="\t")
            
    # ======================================================================
    #  Data generation / normalization
    # ======================================================================

    def _train_data_generate(self):
        """
        Generate one training instance and normalize all static data
        into the internal normalized representation.
        """
        context = self.dataset.generate_tensors()
        self.context = context
        self.depot_num = context["depot"].shape[0]
        self.cus_num = context["customers"].shape[0]
        self.rs_num = context["charging_stations"].shape[0]

        self._normalizations(context)

    def _eval_data_generate(self, mode):
        """
        TODO: implement evaluation data generation if needed.
        """
        if mode == "fixed":
            context = self.dataset.generate_tensors()
            self.context = context
            self.depot_num = context["depot"].shape[0]
            self.cus_num = context["customers"].shape[0]
            self.rs_num = context["charging_stations"].shape[0]

            self._normalizations(context)
        elif mode == "solomon_txt":
            raise NotImplementedError("Not Implemented Yet!")
        else:
            raise ValueError(f"Unknown eval mode: {mode}")

    def _normalizations(self, context):
        """
        Normalize all static quantities:
            - node positions -> [0,1]^2
            - demand -> fraction of loading_capacity
            - time window / service time -> fraction of instance_max_time
            - travel_time -> fraction of instance_max_time
            - edge_energy -> fraction of battery capacity (SoC drop)
            - RS_time_to_depot -> fraction of instance_max_time
        """
        # ----- Raw node coordinates -----
        nodes_raw = np.concatenate(
            (context["depot"], context["customers"], context["charging_stations"])
        ).astype(np.float32)
        positions = np.zeros_like(nodes_raw)

        # ----- Demand & time-related raw data -----
        demands_abs = context["demands"].astype(np.float32)       # (n_cus,)
        time_window_abs = context["tw"].astype(np.float32)        # (n_cus, 2)
        service_time_abs = context["service_time"].astype(np.float32)  # (n_cus,)

        data = context["env"]
        consumption_per_km = data["consumption_per_distance"]     # kWh / km
        b_s = data["battery_capacity"]                            # kWh, E_max
        velocity_abs = data["speed"]                              # km / hour
        loading_capacity = data["loading_capacity"]               # Q_max
        charging_power_abs = (
            data["charging_speed"] * data["charging_efficiency"]
        )  # kW
        instance_max_time_abs = data["instance_endTime"]          # T_scale
        pos_scale = data["area_size"]

        # --------- 1. Position normalization (for obs only) ---------
        x_scale = pos_scale[0][1] - pos_scale[0][0]
        y_scale = pos_scale[1][1] - pos_scale[1][0]
        positions[:, 0] = (nodes_raw[:, 0] - pos_scale[0][0]) / x_scale
        positions[:, 1] = (nodes_raw[:, 1] - pos_scale[1][0]) / y_scale

        # --------- 2. Demand normalization ---------
        demands_norm_cus = demands_abs / loading_capacity
        demands_norm_depot = np.zeros((1,), dtype=np.float32)
        demands_norm_rs = np.zeros((self.rs_num,), dtype=np.float32)

        # --------- 3. Time normalization (T_scale = instance_max_time_abs) ---------
        T_scale = instance_max_time_abs

        time_window_norm_cus = np.clip(time_window_abs / T_scale, 0.0, 1.0)
        time_window_norm_depot = np.array([[0.0, 1.0]], dtype=np.float32)
        time_window_norm_rs = np.array(
            [[0.0, 1.0] * self.rs_num], dtype=np.float32
        ).reshape(self.rs_num, 2)

        service_time_norm_cus = np.clip(service_time_abs / T_scale, 0.0, 1.0)
        service_time_norm_depot = np.zeros((1,), dtype=np.float32)
        service_time_norm_rs = np.zeros((self.rs_num,), dtype=np.float32)

        # --------- 4. Raw distance matrix (for time & energy) ---------
        dist_matrix_raw = gen_dist_matrix(nodes_raw, nodes_raw).astype(np.float32)  # km

        # travel time in normalized units
        travel_time_norm = (dist_matrix_raw / velocity_abs) / T_scale

        # --------- 5. Edge energy consumption (normalized SoC) ---------
        # each edge consumes: dist * (kWh/km) / (kWh)
        edge_energy_frac = dist_matrix_raw * consumption_per_km / b_s

        # --------- 6. Charging: time to charge 1.0 SoC (normalized) ---------
        charging_beta = b_s / (charging_power_abs * T_scale)  # E_max / (P_chg * T_scale)

        # --------- 7. Update env internal static state ---------
        self.nodes_raw = nodes_raw
        self.nodes = positions
        self.depot_nodes = positions[0:1]
        self.cus_nodes = positions[1 : 1 + self.cus_num]
        self.rs_nodes = positions[1 + self.cus_num : 1 + self.cus_num + self.rs_num]

        self.demands = np.concatenate(
            (demands_norm_depot, demands_norm_cus, demands_norm_rs)
        )
        self.time_window = np.concatenate(
            (time_window_norm_depot, time_window_norm_cus, time_window_norm_rs)
        )
        self.service_time = np.concatenate(
            (service_time_norm_depot, service_time_norm_cus, service_time_norm_rs)
        )

        # normalized travel time & energy
        self.travel_time = travel_time_norm
        self.edge_energy = edge_energy_frac
        self.charging_beta = charging_beta

        # in normalized world, capacities are 1.0
        self.loading_capacity = 1.0
        self.battery_capacity = 1.0
        self.instance_max_time = 1.0  # all TW & current_time in [0,1]

        # RS -> depot time in normalized units (prepend depot itself with 0)
        RS_time_to_depot_abs = context["env"]["cs_time_to_depot"]  # (n_rs,)
        self.RS_time_to_depot = np.concatenate(
            [
                np.zeros((1,), dtype=np.float32),
                (RS_time_to_depot_abs / T_scale).astype(np.float32),
            ]
        )

        # distance in normalized coordinate space (only for reward)
        self.dist_matrix = gen_dist_matrix(self.nodes, self.nodes).astype(np.float32)

    # ======================================================================
    #  Transition dynamics (in normalized space)
    # ======================================================================

    def _go_to(self, destination):
        """
        Transition for n_traj parallel vehicles going to 'destination'.

        Args:
            destination: np.ndarray of shape (n_traj,), each entry in [0, n_nodes-1]
        """
        dest_node = self.nodes[destination]
        last_node = self.nodes[self.last]

        # reward uses normalized Euclidean distance in [0, ~1.4]
        dist_display = fun_dist(dest_node, last_node)

        cus_start_idx = 1
        rs_start_idx = 1 + self.cus_num

        go_to_depot = destination == 0
        go_to_rs = destination >= rs_start_idx
        go_to_cus = (destination >= cus_start_idx) & (destination < rs_start_idx)
        go_to_rs_or_cus = ~go_to_depot

        # -------- Reward update --------
        # You can rescale this later (e.g., multiply by pos_scale)
        if self.env_mode == "eval":
            self.reward = -dist_display
        elif self.env_mode == "train":
            # TODO: replace with your RL reward shaping
            self.reward = -dist_display
            self.reward[go_to_cus] += 1.0  # reward for serving customer

            if self.terminate:
                self.reward -= 10.0
                # served_cus  = self.visited[:, cus_start_idx:rs_start_idx].sum(axis=1)
                # total_cus   = rs_start_idx - cus_start_idx
                # unserved_cus = total_cus - served_cus

                # C = self.cus_num  # 惩罚系数，你可以之后调
                # terminal_penalty = -C * unserved_cus.astype(np.float32)
                # self.reward = self.reward + terminal_penalty

        else:
            raise ValueError(f"Unknown Mode: {self.env_mode}")

        # -------- Load update (normalized) --------
        # going to depot: unload all
        self.load[go_to_depot] = 0.0
        # going to customers/RS: add demand (RS demand is 0)
        self.load[go_to_rs_or_cus] += self.demands[destination[go_to_rs_or_cus]]

        # -------- Time update (normalized) --------
        self.current_time[go_to_depot] = 0.0  # at depot, time reset to 0
        # 1) travel time
        self.current_time[go_to_rs_or_cus] += self.travel_time[
            self.last, destination
        ][go_to_rs_or_cus]

        # 2) waiting for customer time window open
        self.current_time[go_to_cus] = np.maximum(
            self.current_time[go_to_cus],
            self.time_window[destination[go_to_cus], 0],
        )

        # 3) service time
        self.current_time += self.service_time[destination]

        # 4) charging at RS (charge to full)
        if go_to_rs.any():
            # battery = consumed SoC after arriving at RS
            self.battery[go_to_rs] += self.edge_energy[self.last, destination][go_to_rs]
            charging_time_norm = self.battery[go_to_rs] * self.charging_beta
            self.current_time[go_to_rs] += charging_time_norm
            # after charging, consider "consumed SoC" reset to 0
            self.battery[go_to_rs] = 0.0

        # -------- Battery update (normalized consumed SoC) --------
        # at depot we assume fully charged, so consumed = 0
        self.battery[go_to_depot] = 0.0

        # already handled RS above (set to 0)
        # here we only add travel consumption when going to customers
        self.battery[go_to_cus] += self.edge_energy[self.last, destination][go_to_cus]

        # -------- Visited & last node update --------
        self.visited[np.arange(self.n_traj), destination] = True
        self.last = destination

