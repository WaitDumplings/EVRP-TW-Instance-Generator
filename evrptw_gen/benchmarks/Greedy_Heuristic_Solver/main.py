import time
import argparse
import os
from solver import GreedySolver
from utils.helpers import plot_solution, set_random_seed
from utils.load_instances import load_instance


def main():
    # Argument parser for command-line inputs
    parser = argparse.ArgumentParser(description="Run VNSTSolver on a given VRPTW instance.")
    parser.add_argument(
        "--instance_path",
        type=str,
        required=False,
        # default='../../data/solomon_datasets/small_instances/Cplex5er/c101C5.txt',
        default="../../../EVRP-TW-Instance-Generator/eval_data_1000/solomon",
        help="Path to the instance dir"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility (default: 1234)"
    )

    args = parser.parse_args()

    # Set random seed
    set_random_seed(args.seed)

    cnt = []
    # Load the instance
    for file in os.listdir(args.instance_path):
        if not file.endswith('.txt'):
            continue
        instance_path = os.path.join(args.instance_path, file)
        instance = load_instance(instance_path)

        # Initialize solver
        solver = GreedySolver(instance)
        # Print instance info
        print(f"Instance file: {args.instance_path.split('/')[-1]}")

        # Solve and measure runtime
        start_time = time.time()
        solution = solver.solve()
        end_time = time.time()

        if not all(solver.visited):
            cnt.append(file)
        # Print solution details
        print("\nRoutes:")
        for route in solution:
            print(route)
        print(f"\nOptimal solution uses {len(solution)} vehicles")
        print(f"Total distance: {solver.global_value:.2f}")
        print(f"Elapsed time: {(end_time - start_time)} s")

        # Plot solution
        # plot_solution(instance, solution)
    
    print(f"Unsolved instances: {cnt}")
    print(f"Total unsolved count: {len(cnt)}")


if __name__ == "__main__":
    # example.
    # python main.py --instance_path ../../data/solomon_datasets/small_instances/Cplex5er/c101C5.txt --seed 1234 --predefine_route_number 3
    main()
