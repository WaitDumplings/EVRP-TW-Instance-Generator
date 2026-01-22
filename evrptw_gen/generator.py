# evrptw_gen/generator.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import math
import os
import copy
from tqdm import tqdm

from .configs.load_config import Config
from .policies.position import CustomerPositionPolicies
from .policies.time_window import TimeWindowPolicies
from .policies.service_time import ServiceTimePolicies
from .policies.demand import DemandPolicies
from .policies.perturb import Perturbation

from .io.saving import save_instance_npz, save_instances_npz
from .utils.geometry import clamp
from .utils.feasibility import cs_min_time_to_depot, effective_charging_power_kw
from .utils.visualization import plot_instance, save_instances
from .utils.energy_consumption_model import consumption_model


class InstanceGenerator:
    def __init__(self, config_path: str, **kwargs):
        self.config= kwargs.get("config", None)
        if self.config is None:
            self.config = Config(config_path)

        self.save_path: Optional[str] = kwargs.get("save_path")
        self.num_instances: int = int(kwargs.get("num_instances", 100))
        raw_env: Dict = self.config.setup_env_parameters()

        # Persist RNG (seeded if provided)
        seed = raw_env.get("rng_seed", None)
        self.rng = np.random.default_rng(seed)
        self.env = self._prepare_env(raw_env)

    def _add_perturb_env(self, env: Dict, perturb_dict: Dict, perturber) -> Dict:
        # Add perturbation logic here
        update_value = perturber.perturb(env, perturb_keys=perturb_dict)

        for key, value in update_value.items():
            # print("Before Perturb:", key, env[key], "After Perturb:", value)
            env[key] = value
        return env

    def _update_seeds(self, seed):
        self.rng = np.random.default_rng(seed)

    def _prepare_env(self, raw_env: Dict) -> Tuple[Dict, Dict]:
        """
        Parse raw config into a flat env dict and extract a perturbation spec.

        Returns
        -------
        env : Dict
        perturb_dict : Dict[str, Tuple[float, float]]
            e.g., {"num_customers_p": (-0.1, +0.2), ...}
        """
        not_copy_key = [
            "vehicles_profiles",
            "charging_profiles",
            "instance_time_range",
            "working_schedule_profiles",
        ]

        def time_to_minutes(hhmm: str) -> int:
            h, m = hhmm.split(":")
            return 60 * int(h) + int(m)

        env: Dict = {}
        for k, v in raw_env.items():
            if k not in not_copy_key:
                env[k] = v

        # Vehicles (assume homogeneous: index 0)
        vprof = raw_env["vehicles_profiles"][0]
        # Keep canonical names consistent downstream
        env["speed"] = float(vprof["speed"])  # km/h
        env["battery_capacity"] = float(vprof["battery_capacity"])  # kWh
        env["consumption_per_distance"] = float(vprof["consumption_per_distance"])  # kWh/km
        env["loading_capacity"] = float(vprof["loading_capacity"])

        # Charging (kW) and efficiency (default charging stations: DC_Fast_150kW)
        cprof = raw_env["charging_profiles"][1]
        env["charging_speed"] = float(cprof["power_kw"])  # kW
        env["charging_efficiency"] = float(cprof.get("efficiency", 1.0))
        # Optional vehicle AC limit (kW). If absent, assume no additional limit.
        if "vehicle_ac_limit_kw" in raw_env:
            env["vehicle_ac_limit_kw"] = float(raw_env["vehicle_ac_limit_kw"])

        # Time horizon and working window
        inst_start, inst_end = raw_env["instance_time_range"]

        ws_idx = 0
        work_start = raw_env["working_schedule_profiles"][ws_idx]["start"]
        work_end = raw_env["working_schedule_profiles"][ws_idx]["end"]
        env["instance_startTime"] = time_to_minutes(inst_start)
        env["instance_endTime"] = time_to_minutes(inst_end)
        env["working_startTime"] = time_to_minutes(work_start)
        env["working_endTime"] = time_to_minutes(work_end)

        # Persist RNG metadata (optional)
        if "rng_seed" in raw_env:
            env["rng_seed"] = raw_env["rng_seed"]

        return env

    def generate(self, perturb_dict=None, node_generater_scheduler=None, **kwargs) -> List[Dict]:
        if perturb_dict is None:
            perturb_dict = {}

        instances = []
        node_generate_policy = kwargs.get("node_generate_policy", "fixed")
        save_solomon = kwargs.get("save_template_solomon", False)
        save_pickle = kwargs.get("save_template_pickle", False)
        plot_instances = kwargs.get("plot_instances", False)

        if save_pickle:
            node_generate_policy = "fixed"

        perturber = Perturbation()

        # 绑定到局部变量，减少循环内查找开销
        base_env = self.env
        add_perturb = self._add_perturb_env
        gen_one = self._generate_one_instance
        scheduler = node_generater_scheduler

        # scheduler 为空时给个更清晰的报错（比 NoneType is not callable 更友好）
        if scheduler is None:
            raise ValueError("node_generater_scheduler is None, but generate() expects a callable scheduler.")

        num_cus, num_cs = scheduler(policy_name=node_generate_policy)
        base_env['num_customers'] = num_cus
        base_env['num_charging_stations'] = num_cs
            
        for id in tqdm(range(self.num_instances)):
            # 关键：浅拷贝即可（前提：后续不要原地改 base_env 的嵌套对象）
            env = base_env.copy()

            # add perturbations
            env = add_perturb(env, perturb_dict, perturber)
            inst = gen_one(env)
            inst['id'] = id
            instances.append(inst)

        # 保存阶段不在循环里
        if self.save_path:
            if save_solomon:
                save_instances(instances, self.save_path, template="solomon")
            if save_pickle:
                save_instances(instances, self.save_path, template="pickle")
            if plot_instances:
                instance_save_path = os.path.join(self.save_path, "plot_instances")
                plot_instance(instances, instance_save_path)
        return instances

    def generate_tensors(self, env = None, **kwargs):
        if env == None:
            env = self.env

        perturber = Perturbation()
        perturb_dict = kwargs.get("perturb_dict", {})
        env["num_customers"] = kwargs.get("num_customers", env['num_customers'])
        env["num_charging_stations"] = kwargs.get("num_charging_stations", env['num_charging_stations'])
        perturb_env = self._add_perturb_env(env, perturb_dict, perturber)
        perturb_env = copy.deepcopy(perturb_env)
        context = self._generate_one_instance(perturb_env)
        return context

    def _generate_one_instance(self, env: Dict) -> Dict:
        # Select policies from env
        pos_policy = CustomerPositionPolicies.from_env(env)
        tw_policy  = TimeWindowPolicies.from_env(env)

        # time unit: hours
        service_time_policy = ServiceTimePolicies.from_env(env)
        demand_policy = DemandPolicies.from_env(env)

        depot_pos = self._get_depot_position(env)
        cs_pos, cs_time_to_depot, depot_time_to_cs = self._get_CSs_positions(env, depot_pos)
        # In case we may need for different policies.
        env['instance_type'] = pos_policy.NAME
        env['cs_time_to_depot'] = cs_time_to_depot
        env['time_window_type'] = tw_policy.NAME
        env['service_time_type'] = service_time_policy.NAME
        env['demand_type'] = demand_policy.NAME

        cus_pos, service_time, t_earliest, t_latest, demand = pos_policy.sample(
            env, depot_pos, cs_pos, cs_time_to_depot, depot_time_to_cs, service_time_policy, rng=self.rng, demand_policy=demand_policy
        )
        if env['num_customers'] != cus_pos.shape[0]:
            raise ValueError("Inconsistent shapes in customer position generation.")

        # tensor use minutes as format
        tw = tw_policy.build(
        env, t_earliest, t_latest, service_time, rng=self.rng, tw_format = "minutes"
        )
        customers_pos = cus_pos
        demand = demand
        service_time = service_time * 60  # convert to minutes

        return {"env": env, 
                "depot": depot_pos, 
                "customers": customers_pos, 
                "charging_stations": cs_pos, 
                "demands":demand, 
                "tw":tw,
                "service_time":service_time}

    def _get_depot_position(self, env: Dict) -> np.ndarray:
        """
        Uniformly sample a depot position within the valid instance area.

        env['area_size'] must be ((x_min, x_max), (y_min, y_max))
        Returns a (1, 2) array rounded to 2 decimals.
        """
        (xmin, xmax), (ymin, ymax) = env["area_size"]
        x = self.rng.uniform(xmin, xmax)
        y = self.rng.uniform(ymin, ymax)
        return np.round(np.array([[x, y]], dtype=float), 3)

    def _get_CSs_positions(self, env: Dict, depot_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample charging station (CS) positions within the full-charge reach region,
        and compute each CS's minimal travel time to the depot under the
        full-charging policy.

        Returns
        -------
        cs_positions : (N, 2) float array
        cs_time_to_depot : (N,) float array of hours
        """
        (xmin, xmax), (ymin, ymax) = env["area_size"]

        # Max distance on a full charge (km): battery(kWh) / consumption(kWh/km)
        consumption_per_distance = consumption_model(env, model_type = None)
        radius_cs = float(env['battery_capacity']) / consumption_per_distance

        speed = float(env['speed'])  # km/h
        p_eff = effective_charging_power_kw(env)  # kW (kWh/h)

        # Initial sampling box centered at the depot and clamped to area bounds
        cx, cy = float(depot_pos[0, 0]), float(depot_pos[0, 1])
        sxmin, sxmax = clamp(cx - radius_cs, xmin, xmax), clamp(cx + radius_cs, xmin, xmax)
        symin, symax = clamp(cy - radius_cs, ymin, ymax), clamp(cy + radius_cs, ymin, ymax)

        num_cs = int(env['num_charging_stations'])
        cs_positions: List[np.ndarray] = []
        cs_time_to_depot: List[float] = []
        depot_time_to_cs: List[float] = []

        max_trials = env.get('max_trials_per_cs', 5000)  # hard cap to avoid infinite loops
        
        trials = 0
        while len(cs_positions) < num_cs and trials < max_trials:
            trials += 1
            x = self.rng.uniform(sxmin, sxmax)
            y = self.rng.uniform(symin, symax)
            cand = np.array([round(x, 2), round(y, 2)], dtype=float)

            output = cs_min_time_to_depot(
                env=env,
                depot_pos=depot_pos[0],          # (2,)
                candidate_cs_pos=cand,           # (2,)
                cs_positions=cs_positions,       # list of (2,)
                cs_time_to_depot=cs_time_to_depot,
                depot_time_to_cs=depot_time_to_cs,
                radius=radius_cs,
                speed=speed,
                p_eff=p_eff,
                use_cs_range=False
            )

            feasible, time_candidate_depot, time_depot_candidate = output

            if feasible:
                cs_positions.append(cand)
                cs_time_to_depot.append(time_candidate_depot)
                depot_time_to_cs.append(time_depot_candidate)

                # Optionally re-center local sampling window around the last accepted CS
                x0, y0 = cand
                sxmin = clamp(x0 - radius_cs, xmin, xmax)
                sxmax = clamp(x0 + radius_cs, xmin, xmax)
                symin = clamp(y0 - radius_cs, ymin, ymax)
                symax = clamp(y0 + radius_cs, ymin, ymax)

        return (
                np.asarray(cs_positions, dtype=float),
                np.asarray(cs_time_to_depot, dtype=float),
                np.asarray(depot_time_to_cs, dtype=float),
                )