# evrptw_gen/benchmark_train.py

import os
import argparse
from distutils.util import strtobool

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
        "--total-timesteps",
        type=int,
        default=500_000_000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=4e-5,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="the weight decay of the optimizer",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=96,
        help="the number of parallel game environments",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=160,
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="the discount factor gamma",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="the lambda for the general advantage estimation",
    )
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=8,
        help="the number of mini-batches",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=3,
        help="the K epochs to update the policy",
    )
    parser.add_argument(
        "--tanh_clipping",
        type=float,
        default=15.0,
        help="tanh clipping in the agent",
    )
    parser.add_argument(
        "--n_encode_layers",
        type=int,
        default=3,
        help="number of encoder layers",
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="coefficient of the entropy",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="coefficient of the value function",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="the target KL divergence threshold",
    )
    parser.add_argument(
        "--n-traj",
        type=int,
        default=64,
        help="number of trajectories(players) in a vectorized sub-environment",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=12,
        help="how many test instance",
    )
    parser.add_argument(
        "--multi-greedy-inference",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="whether to use multiple trajectory greedy inference",
    )
    parser.add_argument(
        "--test_agent",
        type=int,
        default=32,
        help="test agent",
    )
    parser.add_argument("--lambda-fail-init", type=float, default=5.0)
    parser.add_argument("--target-success", type=float, default=0.80)
    parser.add_argument("--lambda-lr", type=float, default=1.0)
    parser.add_argument("--lambda-max", type=float, default=20.0)
    parser.add_argument("--lambda_lr_up", type=float, default=0.5,help="dual ascent step size when fail_rate > target_fail (constraint violated)")
    parser.add_argument("--lambda_lr_down", type=float, default=2.0, help="dual descent step size when fail_rate < target_fail (constraint over-satisfied)")
    parser.add_argument("--lambda_tolerance", type=float, default=0.05, help="tolerance band around target_fail where lambda is not updated")
    parser.add_argument("--eval_method", type=str, default="solomon", help="evaluation method: greedy or sampling [fixed / solomon]")
    parser.add_argument("--eval_data_path", type=str, default="./eval_data/evrptw_100C_20R.pkl", help="path to evaluation data when eval_env_mode is solomon_txt")
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "config.yaml"),
        help="path to evrptw_gen config file",
    )

    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()
    # 这里要调用 train，而不是 main
    train(args)
