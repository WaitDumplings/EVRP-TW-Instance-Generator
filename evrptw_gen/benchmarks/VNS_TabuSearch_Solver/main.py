import time
import argparse
from solver import VNSTSolver
from utils.helpers import plot_solution, set_random_seed
from utils.load_instances import load_instance


def main():
    # Argument parser for command-line inputs
    parser = argparse.ArgumentParser(description="Run VNSTSolver on a given VRPTW instance.")
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to the instance file (e.g., ../../data/solomon_datasets/.../c101C5.txt)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility (default: 1234)"
    )
    parser.add_argument(
        "--predefine_route_number",
        type=int,
        default=2,
        help="Predefined number of routes for VNSTSolver (default: 2)"
    )
    args = parser.parse_args()

    # Load the instance
    instance = load_instance(args.instance_path)

    # Set random seed
    set_random_seed(args.seed)

    # Initialize solver
    solver = VNSTSolver(instance, predefine_route_number=args.predefine_route_number)

    # Print instance info
    print(f"Instance file: {args.instance_path.split('/')[-1]}")

    # Solve and measure runtime
    start_time = time.time()
    solution = solver.solve()
    end_time = time.time()

    # Print solution details
    print(f"Optimal solution uses {len(solution)} vehicles")
    print(f"Total distance: {solver.global_value:.2f}")
    print(f"Elapsed time: {(end_time - start_time):.3f} s")

    # Plot solution
    plot_solution(instance, solution)


if __name__ == "__main__":
    # example.
    # python main.py --instance_path ../../data/solomon_datasets/small_instances/Cplex5er/c101C5.txt --seed 1234 --predefine_route_number 3
    main()
