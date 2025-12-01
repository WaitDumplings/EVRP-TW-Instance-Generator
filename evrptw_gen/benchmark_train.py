# evrptw_gen/benchmark_train.py

import os
import argparse
from distutils.util import strtobool

# 1) 设定 GPU 可见性（可根据需要调整/删除）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
        default=1e-4,
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
        default=2,
        help="the number of parallel game environments",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=40,
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
        default=0.98,
        help="the lambda for the general advantage estimation",
    )
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=32,
        help="the number of mini-batches",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=4,
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
        default=4,
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
        default=0.6,
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
        default=10,
        help="number of trajectories in a vectorized sub-environment",
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
        default=100,
        help="test agent",
    )
    parser.add_argument(
        "--beam_search",
        type=bool,
        default=False,
        help="beam_search",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=20,
        help="beam_search size",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.1,
        help="temperature when sampling",
    )
    # config_path 建议用相对当前文件的绝对路径，避免 cwd 不一致的问题
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "config.yaml"),
        help="path to evrptw_gen config file",
    )

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()
    # 这里要调用 train，而不是 main
    train(args)
