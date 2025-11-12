# evrptw_gen/policies/timewindows.py
from __future__ import annotations
from typing import Dict, Protocol
import numpy as np
from scipy.stats import truncnorm

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


# ---- Two concrete strategies: Narrow / Wide ----
class NarrowTWPolicy:
    NAME = "Narrow"

    def build(
        self,
        env,
        t_earliest,          # (N,)
        t_latest,            # (N,)
        service_time,         # (N,) or scalar
        rng,
    ):
        """
        Generate (N, 2) time windows under 'Narrow' policy.

        Assumptions:
        - All time quantities are in the same unit as working/instance times in env (typically minutes).
        - time_depot_to_cuss[i] is the earliest-arrival travel time from depot to customer i.
        - service_time can be scalar or length-N array (minutes).
        - time_route_instance optionally provides a per-customer margin to finish the route before instance_endTime.
        """
        cfg = env.get("time_window_narrow_config", {})
        alpha = float(cfg.get("alpha", 0.3))  # mean width factor of feasible span
        beta  = float(cfg.get("beta",  0.05)) # std factor of feasible span
        tw_min_width = float(cfg.get("min_width", 1.0))  # absolute minimum width (minutes)
        round_ndigits = int(cfg.get("round_ndigits", 2))

        N = int(env.get("num_customers", service_time.shape[0]))
        lb = float(env["working_startTime"])
        ub = float(env["working_endTime"])
 
        if np.any(t_earliest > t_latest + 1e-6):
            raise ValueError("t_earliest cannot be greater than t_latest for any customer.")

        centers = t_earliest + (t_latest - t_earliest) * np.random.rand(len(t_earliest))
        span = t_latest - t_earliest

        # target width ~ Normal(alpha*span, beta*span)
        mean_w = alpha * span
        std_w  = beta  * span

        a, b = (0 - mean_w) / std_w, np.inf
        widths = truncnorm.rvs(a, b, loc=mean_w, scale=std_w, size=N, random_state=rng)
        # widths = rng.normal(loc=mean_w, scale=std_w)
        # enforce width >= tw_min_width but also not exceed span (otherwise clamp to span)
        # widths = np.clip(widths, a_min=tw_min_width, a_max=np.maximum(span, tw_min_width))

        # compose windows, then intersect with [lb, ub]
        starts = centers - 0.5 * widths
        ends   = centers + 0.5 * widths

        # intersect with feasible [lb, ub]
        starts = np.maximum(starts, lb)
        ends   = np.minimum(ends, ub)

        for i in range(N):
            if ends[i] <= starts[i]:
                breakpoint()
        # breakpoint()
        # print(ends - starts)
        # print()
        tw = np.stack([np.round(starts, round_ndigits), np.round(ends, round_ndigits)], axis=1)
        return tw



class WideTWPolicy:
    NAME = "Wide"

    def build(
        self,
        env,
        t_earliest,          # (N,)
        t_latest,            # (N,)
        service_time,         # (N,) or scalar
        rng,
    ):
        """
        Generate (N, 2) time windows under 'Narrow' policy.

        Assumptions:
        - All time quantities are in the same unit as working/instance times in env (typically minutes).
        - time_depot_to_cuss[i] is the earliest-arrival travel time from depot to customer i.
        - service_time can be scalar or length-N array (minutes).
        - time_route_instance optionally provides a per-customer margin to finish the route before instance_endTime.
        """
        cfg = env.get("time_window_wide_config", {})
        alpha = float(cfg.get("alpha", 0.3))  # mean width factor of feasible span
        beta  = float(cfg.get("beta",  0.05)) # std factor of feasible span
        tw_min_width = float(cfg.get("min_width", 1.0))  # absolute minimum width (minutes)
        round_ndigits = int(cfg.get("round_ndigits", 2))

        N = int(env.get("num_customers", service_time.shape[0]))
        lb = float(env["working_startTime"])
        ub = float(env["working_endTime"])
 
        if np.any(t_earliest > t_latest + 1e-6):
            breakpoint()
            raise ValueError("t_earliest cannot be greater than t_latest for any customer.")

        centers = t_earliest + (t_latest - t_earliest) * np.random.rand(len(t_earliest))
        span = t_latest - t_earliest

        # target width ~ Normal(alpha*span, beta*span)
        mean_w = alpha * span
        std_w  = beta  * span
        a, b = (0 - mean_w) / std_w, np.inf
        widths = truncnorm.rvs(a, b, loc=mean_w, scale=std_w, size=N, random_state=rng)

        # widths = rng.normal(loc=mean_w, scale=std_w)
        # enforce width >= tw_min_width but also not exceed span (otherwise clamp to span)
        # widths = np.clip(widths, a_min=tw_min_width, a_max=np.maximum(span, tw_min_width))

        # compose windows, then intersect with [lb, ub]
        starts = centers - 0.5 * widths
        ends   = centers + 0.5 * widths

        # intersect with feasible [lb, ub]
        starts = np.maximum(starts, lb)
        ends   = np.minimum(ends, ub)
        # breakpoint()
        # print(ends - starts)
        # print()

        for i in range(N):
            if ends[i] <= starts[i]:
                breakpoint()
        tw = np.stack([np.round(starts, round_ndigits), np.round(ends, round_ndigits)], axis=1)
        return tw

# ---- Factory: only Narrow / Wide are currently supported ----
class TimeWindowPolicies:
    REGISTRY = {
        "Narrow": NarrowTWPolicy,
        "Wide": WideTWPolicy,
    }

    @classmethod
    def _sample_choice(cls, env: Dict) -> str:
        """
        Select the time window type with the following priority:
        1) If 'test_timewindow_type' is present in env, use it (for deterministic testing).
        2) Otherwise, sample from 'time_window_type_distribution' according to probabilities.
        3) If both missing or invalid, raise ValueError.
        """
        if "test_timewindow_type" in env:
            return env["test_timewindow_type"]

        dist = env.get("time_window_type_distribution", None)
        if dist is not None and isinstance(dist, dict) and len(dist) > 0:
            keys = list(dist.keys())
            probs = np.array(list(dist.values()), dtype=float)
            probs /= probs.sum()
            return np.random.choice(keys, p=probs)

        raise ValueError(
            "Neither 'time_window_type_distribution' nor 'test_timewindow_type' provided or valid in env."
        )

    @classmethod
    def from_env(cls, env: Dict) -> TimeWindowPolicy:
        choice = cls._sample_choice(env)
        if choice not in cls.REGISTRY:
            raise ValueError(f"Unknown time_window_type: {choice} (expected one of {list(cls.REGISTRY)})")
        env["time_window_type"] = choice  # record the sampled type for reproducibility
        return cls.REGISTRY[choice]()
