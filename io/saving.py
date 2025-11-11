# evrptw_gen/io/saving.py
from __future__ import annotations
from typing import Dict, List, Optional
import os
import numpy as np

__all__ = [
    "save_instance_npz",
    "save_instances_npz",
    "load_instance_npz",
]

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_instance_npz(instance: Dict, save_dir: str, name: Optional[str] = None) -> str:
    """
    Save a single instance to a compressed NPZ file.
    Arrays saved:
      - depot             (1,2)
      - charging_stations (S,2)
      - customers         (N,6)
      - env               (dict)  -- stored as object, load with allow_pickle=True
    Returns the file path.
    """
    _ensure_dir(save_dir)
    base = name or f"instance_{np.random.randint(1e12):012d}"
    path = os.path.join(save_dir, f"{base}.npz")

    # NOTE: allow_pickle=True 是 np.load 的参数，不是 np.savez_compressed 的参数
    np.savez_compressed(
        path,
        depot=instance["depot"],
        charging_stations=instance["charging_stations"],
        customers=instance["customers"],
        env=instance.get("env", {}),
    )
    return path

def save_instances_npz(instances: List[Dict], save_dir: str, prefix: str = "instance") -> List[str]:
    paths = []
    for i, inst in enumerate(instances):
        name = f"{prefix}_{i:05d}"
        p = save_instance_npz(inst, save_dir, name=name)
        paths.append(p)
    return paths

def load_instance_npz(path: str) -> Dict:
    """
    Load one instance from NPZ.
    Important: need allow_pickle=True to load 'env' dict.
    """
    data = np.load(path, allow_pickle=True)
    depot = data["depot"]
    charging_stations = data["charging_stations"]
    customers = data["customers"]
    env = data["env"].item() if "env" in data else {}
    return {
        "env": env,
        "depot": depot,
        "customers": customers,
        "charging_stations": charging_stations,
    }
