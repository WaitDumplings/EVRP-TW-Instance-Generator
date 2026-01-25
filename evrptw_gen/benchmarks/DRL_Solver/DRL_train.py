# evrptw_gen/benchmark_train.py

import os
import argparse
from distutils.util import strtobool
from evrptw_gen.benchmarks.DRL_Solver.train import train

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
        default="evrptw_gen.benchmarks.DRL_Solver.envs.evrp_vector_env:EVRPTWVectorEnv",
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
        default=3e-5,
        help="the learning rate of the optimizer",
    )
    parser.add_argument("--critic-lr", type=float, default=3e-5)

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="the weight decay of the optimizer",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=256,
        help="the number of parallel game environments",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=150,
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
        default=16,
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
        default=0.001,
        help="coefficient of the entropy",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="coefficient of the value function",
    )
    parser.add_argument("--max-grad-norm-backbone", type=float, default=2.0)
    parser.add_argument("--max-grad-norm-critic", type=float, default=10.0)
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.01,
        help="the target KL divergence threshold",
    )
    parser.add_argument(
        "--n-traj",
        type=int,
        default=20,
        help="number of trajectories(players) in a vectorized sub-environment",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=None,
        help="how many test instance",
    )
    parser.add_argument(
        "--test-sample-mode",
        type=str,
        default="greedy",
        help="sampling methods in testing",
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
        default=1,
        help="test agent",
    )
    parser.add_argument("--use-graph-token", type=bool, default=False)
    parser.add_argument("--lambda-fail-init", type=float, default=10.0)
    parser.add_argument("--target-success", type=float, default=0.95)
    parser.add_argument("--lambda-lr", type=float, default=1.0)
    parser.add_argument("--lambda-max", type=float, default=50.0)
    parser.add_argument("--lambda_lr_up", type=float, default=1.0,help="dual ascent step size when fail_rate > target_fail (constraint violated)")
    parser.add_argument("--lambda_lr_down", type=float, default=2.0, help="dual descent step size when fail_rate < target_fail (constraint over-satisfied)")
    parser.add_argument("--lambda_tolerance", type=float, default=0.05, help="tolerance band around target_fail where lambda is not updated")
    parser.add_argument("--env_mode", type=str, default="train", help="env mode: train / eval")
    parser.add_argument("--accum_steps", type=int, default=4, help="accum grad steps for updating parametes")
    parser.add_argument("--save-dir", type=str, default="./checkpoint/TW_Regime/Loose", help="save path for checkpoints")
    parser.add_argument("--eval_batch_size", type=int, default=100, help="the batch size for evaluation")
    parser.add_argument("--eval_data_path", type=str, default="./eval/Cus_100_sample/pickle/evrptw_100C_20R.pkl", help="path to evaluation data when eval_env_mode is solomon_txt")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./evrptw_gen/configs/config.yaml",
        help="path to evrptw_gen config file",
    )

    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()
    # 这里要调用 train，而不是 main
    train(args)
