# evrptw_gen/policies/demand.py
from __future__ import annotations
from typing import Dict, Protocol
import numpy as np


class ClusterAssignmentPolicy(Protocol):

    def build(self, env: Dict, num_cluster: int, rng: np.random.Generator, num_customers = None) -> np.ndarray:
        """Return demand array of shape (num_customers,)."""
        ...


class BalancedClusterNumberPolicy:
    """
 
    """
    NAME = "Balanced"

    def build(self, env: Dict, num_cluster: int, rng: np.random.Generator, num_customers = None) -> np.ndarray:
        customer_num = int(env.get("num_customers", 10)) if num_customers is None else num_customers
        base = customer_num // num_cluster
        remainder = customer_num % num_cluster
        assignemts = [base] * num_cluster
        for i in range(remainder):
            idx = np.random.randint(0, num_cluster - 1)
            assignemts[idx] += 1
        return np.array(assignemts)
        
class DirichletClusterAssignmentPolicy:
    """
 
    """
    NAME = "Dirichlet"

    def build(self, env: Dict, num_cluster: int, rng: np.random.Generator) -> np.ndarray:
        customer_num = int(env.get("num_customers", 10))
        alpha = env.get("dirichlet_alpha", 1.0)
        proportions = rng.dirichlet([alpha] * num_cluster)
        assignments = np.round(proportions * customer_num).astype(int)

        # Adjust to ensure the sum matches customer_num
        diff = customer_num - np.sum(assignments)
        for _ in range(abs(diff)):
            idx = rng.integers(0, num_cluster)
            if diff > 0:
                assignments[idx] += 1
            elif assignments[idx] > 0:
                assignments[idx] -= 1

        return assignments

class ClusterAssignmentPolicies:
    REGISTRY = {"Balanced": BalancedClusterNumberPolicy,
                "Dirichlet": DirichletClusterAssignmentPolicy}

    @classmethod
    def from_env(cls, env: Dict) -> ClusterAssignmentPolicy:
        """

        """
        choice = env.get("cluster_assignment_policy", "Balanced")
        if choice not in cls.REGISTRY:
            raise ValueError(f"Unknown cluster assignment policy: {choice}")
        env["cluster_assignment_policy"] = choice
        return cls.REGISTRY[choice]()
