# evrptw_gen/policies/demand.py
from __future__ import annotations
from typing import Dict, Protocol
import numpy as np


class DemandPolicy(Protocol):
    def build(self, env: Dict, num_customers: int, rng: np.random.Generator) -> np.ndarray:
        """Return demand array of shape (num_customers,)."""
        ...


class RandomDemandPolicy:
    """
    Uniformly sample customer demands in [min, max].
    If unspecified, defaults to [0.1, env['loading_capacity']].
    """
    NAME = "Random"

    def build(self, env: Dict, num_customers: int, rng: np.random.Generator) -> np.ndarray:
        cap = float(env.get("loading_capacity", 1.0))
        cfg = env.get("demand_random_config", {})
        dmin = float(cfg.get("min", 0.1))
        dmax = float(cfg.get("max", cap))

        dmax = 0.15

        if dmax <= dmin:
            dmax = dmin + 0.01
        demands = rng.uniform(dmin, dmax, size=num_customers)
        return np.round(demands, 2).astype(np.float32)


class DemandPolicies:
    REGISTRY = {"Random": RandomDemandPolicy}

    @classmethod
    def from_env(cls, env: Dict) -> DemandPolicy:
        """
        Currently only supports 'Random'.
        Future: can extend with 20_80 or clustered demand distributions.
        """
        choice = env.get("demand_policy", "Random")
        if choice not in cls.REGISTRY:
            raise ValueError(f"Unknown demand policy: {choice}")
        env["demand_policy"] = choice
        return cls.REGISTRY[choice]()
