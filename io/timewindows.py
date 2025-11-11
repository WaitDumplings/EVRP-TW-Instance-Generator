# evrptw_gen/policies/timewindows.py
from __future__ import annotations
from typing import Dict, Protocol
import numpy as np

class TimeWindowPolicy(Protocol):
    def build(
        self,
        env: Dict,
        depot_pos: np.ndarray,
        cs_pos: np.ndarray,
        cus_pos: np.ndarray,
        time_depot_to_css: np.ndarray,
        time_depot_to_cuss: np.ndarray,
        time_cus_to_depot: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        ...

class RandomTWPolicy:
    NAME = "Random"
    def build(self, env, depot_pos, cs_pos, cus_pos, time_depot_to_css, time_depot_to_cuss, time_cus_to_depot, rng):
        # TODO: 按 alpha/beta、工作时间、最小可达时间生成 (N,2)
        raise NotImplementedError

class ClusterAwareTWPolicy:
    NAME = "ClusterAware"
    def build(self, env, depot_pos, cs_pos, cus_pos, time_depot_to_css, time_depot_to_cuss, time_cus_to_depot, rng):
        # TODO: 面向簇的中心/宽度策略
        raise NotImplementedError

class MixedRCTWPolicy:
    NAME = "RC"
    def __init__(self, random_policy: TimeWindowPolicy = None, cluster_policy: TimeWindowPolicy = None):
        self.random_policy = random_policy or RandomTWPolicy()
        self.cluster_policy = cluster_policy or ClusterAwareTWPolicy()

    def build(self, env, depot_pos, cs_pos, cus_pos, time_depot_to_css, time_depot_to_cuss, time_cus_to_depot, rng):
        N = cus_pos.shape[0]
        mask = env.get('_rc_mask', None)
        if mask is None or len(mask) != N:
            ratio = float(env.get('rc_random_ratio', 0.5))
            n_r = int(round(N * ratio)); n_c = N - n_r
            labels = (np.concatenate([np.ones(n_r, dtype=bool), np.zeros(n_c, dtype=bool)])
                      if (n_r and n_c) else (np.ones(n_r, dtype=bool) if n_r else np.zeros(n_c, dtype=bool)))
            perm = rng.permutation(N) if N > 0 else np.arange(0)
            mask = labels[perm]

        r_idx = mask
        c_idx = ~mask
        tw = np.zeros((N, 2), dtype=float)
        if r_idx.any():
            tw_r = self.random_policy.build(env, depot_pos, cs_pos, cus_pos[r_idx],
                                            time_depot_to_css, time_depot_to_cuss[r_idx], time_cus_to_depot[r_idx], rng)
            tw[r_idx] = tw_r
        if c_idx.any():
            tw_c = self.cluster_policy.build(env, depot_pos, cs_pos, cus_pos[c_idx],
                                             time_depot_to_css, time_depot_to_cuss[c_idx], time_cus_to_depot[c_idx], rng)
            tw[c_idx] = tw_c
        return tw

class TimeWindowPolicies:
    REGISTRY = {"Random": RandomTWPolicy, "ClusterAware": ClusterAwareTWPolicy, "RC": MixedRCTWPolicy,
                "Narrow": RandomTWPolicy, "Wide": RandomTWPolicy}  # 可按需映射
    @classmethod
    def from_env(cls, env: Dict) -> TimeWindowPolicy:
        key = env.get("time_window_type", "Random")
        if key not in cls.REGISTRY:
            raise ValueError(f"Unknown time_window_type: {key}")
        return cls.REGISTRY[key]()

