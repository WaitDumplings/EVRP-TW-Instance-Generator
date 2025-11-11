# evrptw_gen/policies/demand.py
from __future__ import annotations
from typing import Dict, Protocol
import numpy as np


class ClusterNumberPolicy(Protocol):

    def build(self, env: Dict, rng: np.random.Generator) -> np.ndarray:
        """Return demand array of shape (num_customers,)."""
        ...


class FixedClusterNumberPolicy:
    """
 
    """
    NAME = "Fixed"

    def build(self, env: Dict, rng: np.random.Generator) -> np.ndarray:
        cluster_number = int(env.get("cluster_number", 3))
        return cluster_number

class ClusterNumberPolicies:
    REGISTRY = {"Fixed": FixedClusterNumberPolicy}

    @classmethod
    def from_env(cls, env: Dict) -> ClusterNumberPolicy:
        """
        Currently only supports 'Random'.
        Future: can extend with 20_80 or clustered demand distributions.
        """
        choice = env.get("cluster_number_policy", "Fixed")
        if choice not in cls.REGISTRY:
            raise ValueError(f"Unknown cluster number policy: {choice}")
        env["cluster_number_policy"] = choice
        return cls.REGISTRY[choice]()
