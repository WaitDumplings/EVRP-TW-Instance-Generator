# evrptw_gen/benchmark_train.py

import os
import argparse
from distutils.util import strtobool

import os
import time
import numpy as np
from tqdm import tqdm
from distutils.util import strtobool
import pickle
from evrptw_gen.utils.nodes_generatro_scheduler import NodesGeneratorScheduler
from evrptw_gen.configs.load_config import Config
import torch.nn.functional as F

import warnings

warnings.filterwarnings(
    "ignore",
    message="WARN: A Box observation space has an unconventional shape*",
    category=UserWarning,
)

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

from evrptw_gen.benchmarks.models.attention_model_wrapper import Agent

from evrptw_gen.benchmarks.wrappers.recordWrapper import RecordEpisodeStatistics
from evrptw_gen.benchmarks.wrappers.syncVectorEnvPomo import SyncVectorEnv
from evrptw_gen.benchmarks.utils.utils import update_lambda_fail


# 1) 设定 GPU 可见性（可根据需要调整/删除）
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 2) 从 benchmarks/train.py 导入 train 函数
# 方法一：包内绝对导入（推荐，清晰）
from evrptw_gen.benchmarks.train import train

# 如果你以后想把包名改掉，可以用相对导入（但运行时一定要用 -m）：
# from .benchmarks.train import train


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument(
        "--seed", type=int, default=1234,
        help="seed of the experiment",
    )
    parser.add_argument(
        "--cuda-id", type=int, default=0,
        help="cuda device id",
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default",
    )
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="cleanRL",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="the entity (team) of wandb's project",
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)",
    )
    # Algorithm specific arguments
    parser.add_argument(
        "--problem",
        type=str,
        default="evrptw",
        help="the OR problem we are trying to solve, it will be passed to the agent",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="evrptw-v0",
        help="the id of the environment",
    )
    parser.add_argument(
        "--env-entry-point",
        type=str,
        default="evrptw_gen.benchmarks.envs.evrp_vector_env:EVRPTWVectorEnv",
        help="the path to the definition of the environment",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=200,
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--tanh_clipping",
        type=float,
        default=10.0,
        help="tanh clipping in the agent",
    )
    parser.add_argument(
        "--n_encode_layers",
        type=int,
        default=3,
        help="number of encoder layers",
    )
    parser.add_argument(
        "--n-traj",
        type=int,
        default=64,
        help="number of trajectories(players) in a vectorized sub-environment",
    )
    parser.add_argument(
        "--test_agent",
        type=int,
        default=1,
        help="test agent",
    )
    parser.add_argument("--use_graph_token", type=bool, default=True, help="whether to use graph token")
    parser.add_argument("--env_mode", type=str, default="eval", help="env mode: train / eval")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="the batch size for evaluation")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/run_20260114-144503/best_model.pth", help="path to load model checkpoint")
    parser.add_argument("--eval_data_path", type=str, default="./eval_data_1000/pickle/evrptw_100C_20R.pkl", help="path to evaluation data when eval_env_mode is solomon_txt")
    parser.add_argument("--save_log_dir", type=str, default="checkpoints", help="directory to save models and logs")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./evrptw_gen/configs/config.yaml",
        help="path to evrptw_gen config file",
    )

    args = parser.parse_args()
    # fmt: on
    return args

def make_env(env_id, seed, cfg=None):
    if cfg is None:
        cfg = {}

    def thunk():
        env = gym.make(env_id, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(int(seed))
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def main(args):
    #########################
    #### Env Definition #####
    #########################
    # Register the environment.
    # Note: entry_point must be a fully-qualified import path 
    # (details explained in the discussion above).
    gym.envs.register(
        id=args.env_id,
        entry_point=args.env_entry_point,
    )

    #########################
    ### Model Definition ####
    #########################
    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"

    agent = Agent(device=device, 
                  name=args.problem, 
                  tanh_clipping = args.tanh_clipping, 
                  n_encode_layers = args.n_encode_layers,
                  use_graph_token = args.use_graph_token).to(device)

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    agent.load_state_dict(ckpt)
    agent.to(device)
    print("Loaded model from {}".format(args.checkpoint_path))
    agent.eval()

    save_log_dir = args.save_log_dir
    os.makedirs(save_log_dir, exist_ok=True)

    # num_updates = args.total_timesteps // args.batch_size
    config = Config(args.config_path)
    eval_data = pickle.load(open(args.eval_data_path, "rb"))
    num_test_envs = len(eval_data)
    batch_size = args.eval_batch_size
    record_info = []

    for batch in range(0, num_test_envs, batch_size):
        #########################
        #### Env Creation #######
        #########################
        batch_test_env_id = list(
            range(batch, min(batch + batch_size, num_test_envs))
        )
        test_envs = SyncVectorEnv(
            [
                make_env(
                    args.env_id,
                    int(args.seed + i),
                    cfg={"env_mode": "eval", 
                        "config": config, 
                        "n_traj": args.test_agent,
                        "eval_data": eval_data[i]},   # New Arg
                )
                for i in batch_test_env_id
            ]
        )

        # Update Next Environment ##
        # A policy to update the customer_numbers and charging_stations_numbers and other env parameters (Curriculum Learning)
        t_eval_start = time.time()
        # Evaluation Process
        # TRY NOT TO MODIFY: start the game

        # del obs, actions, logprobs, rewards, dones, values, advantages, returns  # 举例
        # torch.cuda.empty_cache()
        test_obs = test_envs.reset()
        for _ in range(0, args.eval_steps):
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logits = agent(test_obs)
            action = action.to("cpu").numpy()
            # TRY NOT TO MODIFY: execute the game and log data.
            test_obs, _, test_done, test_info = test_envs.step(action)

            for item in test_info:
                if "episode" in item.keys():
                    record_info.append(item)

            if test_done.all():
                break
            test_envs.close()
    avg_reward = np.mean([item["episode"]["r"] for item in record_info])
    print("Eval Time: {:.4f}s".format(time.time() - t_eval_start))
    print("Average Reward over {} test episodes: {:.3f}".format(len(record_info), avg_reward))

if __name__ == "__main__":
    args = parse_args()
    main(args)