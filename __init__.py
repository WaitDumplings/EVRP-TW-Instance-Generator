# evrptw_gen/__init__.py
"""
EVRPTW Instance Generator Package
---------------------------------
This package provides:
- InstanceGenerator: main class for generating EVRPTW benchmark data
- EVRPTWDataset: torch-compatible dataset wrapper
- Submodules: policies, utils, io, configs

Example
-------
>>> from evrptw_gen import InstanceGenerator, EVRPTWDataset
>>> gen = InstanceGenerator("./configs/config.yaml", save_path="./Instances", num_instances=10)
>>> instances = gen.generate()
"""

from .generator import InstanceGenerator
from .data.dataset import EVRPTWDataset

# Optionally expose useful submodules
from . import policies, utils, io, configs

__all__ = [
    "InstanceGenerator",
    "EVRPTWDataset",
    "policies",
    "utils",
    "io",
    "configs",
]
