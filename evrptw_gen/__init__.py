# evrptw_gen/__init__.py
"""
EVRPTW Instance Generator Package
---------------------------------
This package provides:
- InstanceGenerator: main class for generating EVRPTW benchmark data
- EVRPTWDataset: torch-compatible dataset wrapper
- Config: configuration loader
- Submodules: policies, utils, io, configs

Example
-------
>>> from evrptw_gen import InstanceGenerator, EVRPTWDataset, Config
>>> cfg = Config("configs/config.yaml")
>>> gen = InstanceGenerator(cfg, save_path="./Instances", num_instances=10)
>>> instances = gen.generate()
"""

# 核心类
from .generator import InstanceGenerator
from .data.dataset import EVRPTWDataset
from .configs.load_config import Config

# 可选：暴露子模块（方便 from evrptw_gen import policies 使用）
from . import policies, utils, io, configs

__all__ = [
    "InstanceGenerator",
    "EVRPTWDataset",
    "Config",
    "policies",
    "utils",
    "io",
    "configs",
]
