"""
Unified entry point for EVRPTW benchmark training.
Can be executed as:
    python run_benchmark.py
No need for -m or changing working directory.
"""

import os
import sys

# -------------------------------------------------------------------
# 1) 确保 evrptw_gen 项目根目录被加入 sys.path
# -------------------------------------------------------------------
# run_benchmark.py 所在目录是 evrptw_gen/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR   # 当前文件所在的 evrptw_gen 即为包根目录

# 确保父目录（Git_Pro）是 sys.path 的第一位
PARENT_DIR = os.path.dirname(PROJECT_ROOT)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# -------------------------------------------------------------------
# 2) 现在可以安全导入 benchmarks/train.py 
# -------------------------------------------------------------------
from evrptw_gen.benchmarks.DRL_Solver.train import train
from evrptw_gen.benchmarks.DRL_Solver.DRL_train import parse_args  # 可复用你已有的 argparse 配置


# -------------------------------------------------------------------
# 3) 主入口 
# -------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    train(args)
