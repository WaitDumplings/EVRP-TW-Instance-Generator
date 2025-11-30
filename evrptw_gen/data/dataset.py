# evrptw_gen/data/dataset.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Callable
import os
import numpy as np
import torch
from torch.utils.data import Dataset


def default_instance_to_tensors(instance: Dict, device: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """
    Convert a single EVRPTW instance dict into tensors.

    Expected `instance` keys:
      - depot: (1, 2)
      - charging_stations: (S, 2)
      - customers: (N, 6) -> [x, y, demand, tw_start, tw_end, service_time]
      - env: dict (kept as-is in Python, not tensor)

    Returns a dict with:
      depot: (1,2), charging_stations: (S,2),
      customers: (N,2), demand: (N,), tw: (N,2), service_time: (N,)
    """
    dep = torch.as_tensor(instance["depot"], dtype=torch.float32)
    cs = torch.as_tensor(instance["charging_stations"], dtype=torch.float32)
    cus_raw = torch.as_tensor(instance["customers"], dtype=torch.float32)

    sample = {
        "depot": dep,                              # (1, 2)
        "charging_stations": cs,                   # (S, 2)
        "customers": cus_raw[:, :2],               # (N, 2)
        "demand": cus_raw[:, 2],                   # (N,)
        "tw": cus_raw[:, 3:5],                     # (N, 2)
        "service_time": cus_raw[:, 5],             # (N,)
        "env": instance.get("env", {}),
    }
    if device is not None:
        for k in ["depot", "charging_stations", "customers", "demand", "tw", "service_time"]:
            sample[k] = sample[k].to(device)
    return sample


class EVRPTWDataset(Dataset):
    """
    EVRP-TW Dataset.

    Two modes:
      (A) In-memory: pass `instances` (list of dicts).
      (B) On-disk: pass `root_dir` containing .npz files (one instance per file).

    Parameters
    ----------
    instances : Optional[List[dict]]
        In-memory instances (as returned by InstanceGenerator.generate()).
    root_dir : Optional[str]
        Directory with saved `.npz` files.
    to_tensor : Optional[Callable[[dict], dict]]
        Converter that maps an instance dict -> dict of tensors.
    device : Optional[str]
        'cpu' or 'cuda'. If provided, tensors are moved accordingly.
    """

    def __init__(
        self,
        instances: Optional[List[Dict]] = None,
        root_dir: Optional[str] = None,
        to_tensor: Optional[Callable[[Dict], Dict]] = None,
        device: Optional[str] = None,
    ):
        if (instances is None) and (root_dir is None):
            raise ValueError("Provide either `instances` or `root_dir`.")
        self.instances = instances
        self.root_dir = root_dir
        self.device = device
        self.to_tensor = to_tensor or (lambda inst: default_instance_to_tensors(inst, device=device))

        if self.instances is None:
            self.files = sorted([f for f in os.listdir(root_dir) if f.endswith(".npz")])
            if len(self.files) == 0:
                raise FileNotFoundError(f"No .npz instances found in {root_dir}")
        else:
            self.files = None

    def __len__(self) -> int:
        return len(self.instances) if self.instances is not None else len(self.files)

    def _load_npz(self, path: str) -> Dict:
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

    def __getitem__(self, idx: int) -> Dict:
        if self.instances is not None:
            instance = self.instances[idx]
        else:
            path = os.path.join(self.root_dir, self.files[idx])
            instance = self._load_npz(path)
        return self.to_tensor(instance)

    # -------- collate helpers -------- #

    @staticmethod
    def collate_list(batch: List[Dict]) -> List[Dict]:
        """Return a list of samples (no padding)."""
        return batch

    @staticmethod
    def collate_pad(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Pad customers and charging stations to batch max sizes; return masks.
        Returned keys:
          depot: (B,2)
          charging_stations: (B,S_max,2), charging_mask: (B,S_max)
          customers: (B,N_max,2), customer_mask: (B,N_max)
          demand: (B,N_max), tw: (B,N_max,2), service_time: (B,N_max)
          env: list[dict]
        """
        def pad_tensor_list(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            sizes = [t.shape[0] for t in tensors]
            max_n = max(sizes) if sizes else 0
            if max_n == 0:
                # Handle degenerate batches
                feat_dim = tensors[0].shape[1] if (tensors and tensors[0].ndim > 1) else 1
                return (torch.zeros((len(tensors), 0, feat_dim), dtype=torch.float32),
                        torch.zeros((len(tensors), 0), dtype=torch.bool))
            feat_dim = tensors[0].shape[1] if tensors[0].ndim > 1 else 1
            out = torch.zeros((len(tensors), max_n, feat_dim), dtype=tensors[0].dtype, device=tensors[0].device)
            mask = torch.zeros((len(tensors), max_n), dtype=torch.bool, device=tensors[0].device)
            for i, t in enumerate(tensors):
                n = t.shape[0]
                if t.ndim == 1:
                    out[i, :n, 0] = t
                else:
                    out[i, :n, :t.shape[1]] = t
                mask[i, :n] = True
            return out, mask

        depots = torch.stack([b["depot"].squeeze(0) for b in batch], dim=0)  # (B,2)

        cs_list = [b["charging_stations"] for b in batch]
        cus_list = [b["customers"] for b in batch]
        dem_list = [b["demand"] for b in batch]
        tw_list  = [b["tw"] for b in batch]
        st_list  = [b["service_time"] for b in batch]

        cs_pad, cs_mask = pad_tensor_list(cs_list)
        cus_pad, cus_mask = pad_tensor_list(cus_list)

        dem_pad, _ = pad_tensor_list([d.view(-1, 1) for d in dem_list]); dem_pad = dem_pad.squeeze(-1)
        tw_pad, _  = pad_tensor_list(tw_list)
        st_pad, _  = pad_tensor_list([s.view(-1, 1) for s in st_list]); st_pad = st_pad.squeeze(-1)

        return {
            "depot": depots,                    # (B,2)
            "charging_stations": cs_pad,       # (B,S_max,2)
            "charging_mask": cs_mask,          # (B,S_max)
            "customers": cus_pad,              # (B,N_max,2)
            "customer_mask": cus_mask,         # (B,N_max)
            "demand": dem_pad,                 # (B,N_max)
            "tw": tw_pad,                      # (B,N_max,2)
            "service_time": st_pad,            # (B,N_max)
            "env": [b.get("env", {}) for b in batch],
        }
