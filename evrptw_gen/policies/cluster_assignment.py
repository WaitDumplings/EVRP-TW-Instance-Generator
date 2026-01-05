# evrptw_gen/policies/demand.py
from __future__ import annotations
from typing import Dict, Protocol
from collections import Counter
import numpy as np


testset_name = "cluster_assignment_policy"
config_list_name = "cluster_assignment_policy_list"
config_type_name = "cluster_assignment_policy_config"

class ClusterAssignmentPolicy(Protocol):

    def build(self, env: Dict, num_cluster: int, rng: np.random.Generator, num_customers = None) -> np.ndarray:
        """Return demand array of shape (num_customers,)."""
        ...


class BalancedClusterNumberPolicy:
    """
 
    """
    NAME = "balanced"

    def build(self, env: Dict, num_cluster: int, rng: np.random.Generator, num_customers = None) -> np.ndarray:
        customer_num = env.get("num_customers", None) if not num_customers else num_customers
        if not customer_num:
            raise ValueError("customer_num must be specified in env for BalancedClusterNumberPolicy.")
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
    NAME = "dirichlet"

    def build(self, env: Dict, num_cluster: int, rng: np.random.Generator, num_customers=None) -> np.ndarray:
        customer_num = num_customers
        if customer_num is None:
            raise ValueError("customer_num must be specified in env.")
        if num_cluster <= 0:
            raise ValueError("num_cluster must be positive.")
        if customer_num < 0:
            raise ValueError("customer_num must be non-negative.")

        # 不可行：无法做到每个 cluster >= 1 且总和=customer_num
        if customer_num < num_cluster:
            breakpoint()
            return []
            raise ValueError("customer_num must be >= num_cluster for DirichletClusterAssignmentPolicy.")

        # 每个 cluster 先给 1，确保不空
        pre_assignments = np.ones(num_cluster, dtype=int)
        remaining = customer_num - num_cluster  # 需要再分配的客户数

        # remaining==0 直接返回（例如 customer_num == num_cluster）
        if remaining == 0:
            return pre_assignments

        alpha = float(env.get("dirichlet_alpha", 1.0))
        proportions = rng.dirichlet([alpha] * num_cluster)

        # 用 floor + 小数部分补齐，保证整数化后总和严格等于 remaining
        raw = proportions * remaining                    # float
        extra = np.floor(raw).astype(int)                # 不会超 remaining
        short = remaining - int(extra.sum())             # short >= 0 必然成立

        if short > 0:
            frac = raw - extra                           # 小数部分
            # 选小数部分最大的 short 个 cluster 各 +1
            # 若你希望随机打破相同小数的平局，可以加一个很小的随机扰动
            idxs = np.argsort(frac)[-short:]
            extra[idxs] += 1

        assignments = pre_assignments + extra

        # 强校验（建议保留，定位任何未来改动引入的问题）
        if assignments.sum() != customer_num:
            raise ValueError(
                f"Inconsistent customer number after Dirichlet cluster assignment: "
                f"sum={assignments.sum()} expected={customer_num}"
            )
        if assignments.min() < 1:
            raise ValueError(
                f"Min cluster size violated: min={assignments.min()} (expected >= 1)"
            )

        return assignments


class ClusterAssignmentPolicies:
    REGISTRY = {"balanced": BalancedClusterNumberPolicy,
                "dirichlet": DirichletClusterAssignmentPolicy}

    @classmethod
    def _sample_choice(cls, env: Dict) -> str:
        """
        Select the cluster assignment type with the following priority:
        1) If 'test_cluster_assignment_type' is present in env, use it (for deterministic testing).
        2) Otherwise, sample from 'cluster_assignment_type_distribution' according to probabilities.
        3) If both missing or invalid, raise ValueError.
        """
        if testset_name in env and env[testset_name] is not None:
            return env[testset_name]

        type_list = env.get(config_list_name, None)
        if Counter(type_list) != Counter(cls.REGISTRY.keys()):
            raise ValueError(f"Unknown policy: {type_list}")
        score_list = []
        for key in type_list:
            cfg = env[config_type_name][key]
            score = cfg.get("score", 0)
            score_list.append(score)

        choice = np.random.choice(
            type_list,
            p=np.array(score_list) / np.sum(score_list)
        )
        choice = 'dirichlet'
        return choice

    @classmethod
    def from_env(cls, env: Dict) -> ClusterAssignmentPolicy:
        choice = cls._sample_choice(env)
        return cls.REGISTRY[choice]()
