# evrptw_gen/policies/servicetime.py
from __future__ import annotations
from typing import Dict, Protocol
import numpy as np


class ServiceTimePolicies(Protocol):
    def build(
        self,
        env: Dict,
        num_customers: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (num_customers,), dtype=float or int (minutes).
        """
        ...


class RandomServiceTimePolicy:
    """
    Uniform random service time in [min, max].
    Config layout in env:
        env['service_time_random'] = {
            'min': <float|int>,      # inclusive lower bound (minutes)
            'max': <float|int>,      # inclusive upper bound (minutes)
            'integer': True|False,   # if True -> integers; else floats
            # optional:
            # 'round_ndigits': 2      # when integer=False, round to ndigits
        }
    """
    NAME = "Random"

    def build(self, env: Dict, num_customers: int, rng: np.random.Generator, unit = "hour") -> np.ndarray:
        cfgs = env.get("servicetime_type_config", {})
        Find = False
        if unit == "hour":
            time_multiplier = 60.0  # convert hours to minutes
        else:
            time_multiplier = 1.0   # already in minutes

        for cfg in cfgs:
            if cfg['policy'] == self.NAME:
                Find = True
                service_time_range = cfg.get('st_range')
                lo, hi = service_time_range
                integer = bool(cfg.get("integer", True))
                break

        if not Find:
            raise ValueError("Not Policy Found in your config files.")

        if integer:
            # inclusive upper bound for integers
            lo_i = int(np.floor(lo))
            hi_i = int(np.ceil(hi))
            if hi_i < lo_i:
                hi_i = lo_i
            st = rng.integers(lo_i, hi_i + 1, size=num_customers, dtype=np.int32)
            return st.astype(np.int32) / time_multiplier

        # float minutes
        st = rng.uniform(lo, hi, size=num_customers).astype(np.float32)
        nd = int(cfg.get("round_ndigits", 2))
        return np.round(st, nd).astype(np.float32) / time_multiplier


class ServiceTimePolicies:
    """
    Factory for service time policy selection.

    Selection priority:
        1) env['test_servicetime_type'] (deterministic testing)
        2) sample from env['servicetime_type_distribution'] (e.g., {'Random': 1.0})
        3) raise if neither is provided/valid
    """
    REGISTRY = {
        "Random": RandomServiceTimePolicy,
    }

    @classmethod
    def _sample_choice(cls, env: Dict) -> str:
        if "test_servicetime_type" in env:
            return env["test_servicetime_type"]

        dist = env.get("servicetime_type_distribution", None)
        if isinstance(dist, dict) and len(dist) > 0:
            keys = list(dist.keys())
            probs = np.array(list(dist.values()), dtype=float)
            probs /= probs.sum()
            return np.random.choice(keys, p=probs)
        elif isinstance(dist, list) and len(dist) > 0:
            keys = dist
            return np.random.choice(keys)

        raise ValueError(
            "Neither 'servicetime_type_distribution' nor 'test_servicetime_type' provided or valid in env."
        )

    @classmethod
    def from_env(cls, env: Dict) -> ServiceTimePolicy:
        choice = cls._sample_choice(env)
        if choice not in cls.REGISTRY:
            raise ValueError(f"Unknown servicetime type: {choice} (expected one of {list(cls.REGISTRY)})")
        env["servicetime_type"] = choice
        return cls.REGISTRY[choice]()
