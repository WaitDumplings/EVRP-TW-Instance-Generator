# evrptw_gen/policies/demand.py
from __future__ import annotations
from typing import Dict, Protocol, Tuple
import numpy as np


class RCSplitPolicy(Protocol):

    def build(self, env: Dict, rng: np.random.Generator) -> Tuple[int, int]:
        """Return demand array of shape (num_customers,)."""
        ...


class FixedRCSplit:
    """
 
    """
    NAME = "Fixed"

    def build(self, env: Dict, rng: np.random.Generator) -> Tuple[int, int]:
        random_ratio = float(env.get('random_customer_ratio'))
        cluster_ratio = float(env.get('cluster_customer_ratio'))

        if random_ratio + cluster_ratio != 1:
            raise ValueError(f"random type ratio plus cluster type ratio should be 1, but we get {random_ratio + cluster_ratio} instead.")
        customer_num = env['num_customers']
        random_customer_number = (customer_num * random_ratio)
        cluster_customer_number = customer_num - random_customer_number
        return int(random_customer_number), int(cluster_customer_number)

class RCPolicies:
    REGISTRY = {"Fixed": FixedRCSplit}

    @classmethod
    def from_env(cls, env: Dict) -> RCSplitPolicy:
        """
        Currently only supports 'Random'.
        Future: can extend with 20_80 or clustered demand distributions.
        """
        choice = env.get("rc_split_policy", "Fixed")
        if choice not in cls.REGISTRY:
            raise ValueError(f"Unknown rc splict policy: {choice}")
        env["rc_split_policy"] = choice
        return cls.REGISTRY[choice]()
