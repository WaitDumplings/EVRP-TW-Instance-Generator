# evrptw_gen/generator.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import math
import os

from .configs.load_config import Config
from .policies.positions import CustomerPositionPolicies
from .policies.timewindows import TimeWindowPolicies
from .policies.servicetimes import ServiceTimePolicies
from .policies.demands import DemandPolicies
from .policies.cluster_number import ClusterNumberPolicies
from .policies.cluster_assignment import ClusterAssignmentPolicies

from .io.saving import save_instance_npz, save_instances_npz
from .utils.geometry import clamp
from .utils.feasibility import cs_min_time_to_depot, effective_charging_power_kw
from .utils.visualization import plot_instance, save_instances
from .utils.energy_consumption_model import consumption_model

# from configs.load_config import Config
# from policies.positions import CustomerPositionPolicies
# from policies.timewindows import TimeWindowPolicies
# from policies.servicetimes import ServiceTimePolicies
# from policies.demands import DemandPolicies
# from policies.cluster_number import ClusterNumberPolicies
# from policies.cluster_assignment import ClusterAssignmentPolicies

# from io.saving import save_instance_npz, save_instances_npz
# from utils.geometry import clamp
# from utils.feasibility import cs_min_time_to_depot, effective_charging_power_kw
# from utils.visualization import plot_instance, save_instances
# from utils.energy_consumption_model import consumption_model


class InstanceGenerator:
    def __init__(self, config_path: str, **kwargs):
        self.config = Config(config_path)
        self.save_path: Optional[str] = kwargs.get("save_path")
        self.num_instances: int = int(kwargs.get("num_instances", 100))
        self.plot_instance: bool = bool(kwargs.get("plot_instances", True))
        raw_env: Dict = self.config.setup_env_parameters()

        # Persist RNG (seeded if provided)
        seed = raw_env.get("rng_seed", None)
        self.rng = np.random.default_rng(seed)

        self.add_perturb: bool = bool(raw_env.get("add_perturb", False))
        self.env, self.perturb_dict = self._prepare_env(raw_env)

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
            "perturb",
        ]

        def time_to_minutes(hhmm: str) -> int:
            h, m = hhmm.split(":")
            return 60 * int(h) + int(m)

        env: Dict = {}
        for k, v in raw_env.items():
            if k not in not_copy_key:
                env[k] = v

        # Vehicles (assume homogeneous: index 0)
        vprof = raw_env["vehicles_profiles"][-1]
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

        # Perturb spec
        perturb_dict: Dict = {}
        for p in raw_env.get("perturb", []):
            perturb_dict.update(p)

        return env, perturb_dict

    def _add_perturb(self, raw_env: dict, perturb_dict: dict) -> dict:
        copy_env = raw_env.copy()
        for key, value in perturb_dict.items():
            raw_env_feature = key[:-2]  # strip "_p"
            sample = float(self.rng.uniform(value[0], value[1]))
            if raw_env_feature.startswith("num"):
                copy_env[raw_env_feature] += int(sample)
                copy_env[raw_env_feature] = max(1, copy_env[raw_env_feature])
            else:
                copy_env[raw_env_feature] *= (1 + sample)
                # keep reasonable precision for reals
                if isinstance(copy_env[raw_env_feature], float):
                    copy_env[raw_env_feature] = round(copy_env[raw_env_feature], 4)
        return copy_env

    def generate(self) -> List[Dict]:
        instances = []
        for _ in range(self.num_instances):
            env = self._add_perturb(self.env, self.perturb_dict) if self.add_perturb else dict(self.env)
            inst = self._generate_one_instance(env)
            instances.append(inst)

        if self.save_path:
            # save_instance_npz(inst, self.save_path)  # IO in io/saving.py
            save_instances(instances, self.save_path, template='solomon')
            if self.plot_instance:
                Instance_save_path = os.path.join(self.save_path, 'plot_instances')
                plot_instance(instances, Instance_save_path)
        return instances

    def _generate_one_instance(self, env: Dict) -> Dict:
        # Select policies from env
        pos_policy = CustomerPositionPolicies.from_env(env)
        tw_policy  = TimeWindowPolicies.from_env(env)

        # time unit: hours
        service_time_policy = ServiceTimePolicies.from_env(env)
        demand_policy = DemandPolicies.from_env(env)

        depot_pos = self._get_depot_position(env)
        cs_pos, cs_time_to_depot, depot_time_to_cs = self._get_CSs_positions(env, depot_pos)

        demand = demand_policy.build(env, num_customers=env['num_customers'], rng=self.rng)

        # In case we may need for different policies.
        env['demand'] = demand 

        cus_pos, service_time, t_earliest, t_latest = pos_policy.sample(
            env, depot_pos, cs_pos, cs_time_to_depot, depot_time_to_cs, service_time_policy, rng=self.rng
        )

        tw = tw_policy.build(
            env, t_earliest, t_latest, service_time, rng=self.rng
        )
        tw /= 60
        
        customers = np.hstack([cus_pos, demand.reshape(-1, 1), tw, service_time.reshape(-1, 1)])
        return {"env": env, "depot": depot_pos, "customers": customers, "charging_stations": cs_pos}

    def _get_depot_position(self, env: Dict) -> np.ndarray:
        """
        Uniformly sample a depot position within the valid instance area.

        env['area_size'] must be ((x_min, x_max), (y_min, y_max))
        Returns a (1, 2) array rounded to 2 decimals.
        """
        (xmin, xmax), (ymin, ymax) = env["area_size"]
        x = self.rng.uniform(xmin, xmax)
        y = self.rng.uniform(ymin, ymax)
        return np.round(np.array([[x, y]], dtype=float), 2)

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