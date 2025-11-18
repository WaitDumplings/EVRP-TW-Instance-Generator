import argparse
import os
import random
import shutil
import time
from distutils.util import strtobool

# import gym
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

from evrptw_gen.benchmarks.wrappers.recordWrapper import RecordEpisodeStatistics
from evrptw_gen.benchmarks.wrappers.syncVectorEnvPomo import SyncVectorEnv
from evrptw_gen.configs.load_config import Config

# from wrappers.recordWrapper import RecordEpisodeStatistics
# # from models.attention_model_wrapper import Agent
# from wrappers.syncVectorEnvPomo import SyncVectorEnv
# from configs.load_config import Config

def make_env(env_id, seed, cfg={}):
    def thunk():
        env = gym.make(env_id, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def train(args):
    #######################
    #### Env definition ####
    #######################
    # 注册环境（注意 entry_point 要写完整路径，下面第 2 点再说）
    gym.envs.register(
        id=args.env_id,
        entry_point=args.env_entry_point,
    )

    # 读取配置
    

    # 如果要调试可以暂时开：
    # breakpoint()

    # 向量化训练环境
    breakpoint()
    envs = SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + i,
                cfg={"config_path": args.config_path, "n_traj": args.n_traj},
            )
            for i in range(args.num_envs)
        ]
    )

    # （这里原本应该有你的 PPO 训练循环、log、保存 ckpt 等）

    # 目前先在函数结尾关掉 envs
    envs.close()

    # test_envs 暂时根本没定义，先删掉
    # test_envs.close()



