import time
import argparse
import os
from solver import GreedySolver
from utils.helpers import plot_solution, set_random_seed
from utils.load_instances import load_instance

import json
import pandas as pd

def main():
    # Argument parser for command-line inputs
    parser = argparse.ArgumentParser(description="Run VNSTSolver on a given VRPTW instance.")
    parser.add_argument(
        "--instance_path",
        type=str,
        required=False,
        # default='../../data/solomon_datasets/small_instances/Cplex5er/c101C5.txt',
        default="../../../eval/Cus_15/solomon",
        help="Path to the instance dir"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility (default: 1234)"
    )
    parser.add_argument(
        "--save_log_path",
        type=str,
        default="./logs",
        help="Path to save the log file"
    )

    args = parser.parse_args()

    # Set random seed
    set_random_seed(args.seed)

    summary_records = []
    route_records = []
    failed_files = []

    for file in os.listdir(args.instance_path):
        if not file.endswith(".txt"):
            continue

        instance_path = os.path.join(args.instance_path, file)
        instance = load_instance(instance_path)

        solver = GreedySolver(instance)

        # 这里建议打印文件名，不是文件夹名
        print(f"Instance file: {file}")

        start_time = time.time()
        solution = solver.solve()   # solution: list of routes
        end_time = time.time()

        visited_all = bool(all(solver.visited))
        if not visited_all:
            failed_files.append(file)

        instance_id = instance.get("instance_id", os.path.splitext(file)[0])
        fleet_size = len(solution)
        objective_value = float(getattr(solver, "global_value", float("nan")))
        elapsed_time = float(end_time - start_time)

        # 1) Summary 一行
        summary_records.append({
            "instance_id": instance_id,
            "file": file,
            "fleet_size": fleet_size,
            "objective_value": objective_value,
            "elapsed_time_s": elapsed_time,
            "visited_all": visited_all,
            "unvisited_count": int(len(solver.visited) - sum(bool(x) for x in solver.visited)),
            # 如果你希望快速回看所有 route，也可以直接塞一列（建议 JSON 化）
            "routes_json": json.dumps(solution, ensure_ascii=False),
        })

        # 2) Routes 长表：每条 route 一行
        for r_idx, route in enumerate(solution):
            route_records.append({
                "instance_id": instance_id,
                "file": file,
                "route_idx": r_idx,
                # route 可能是 list[int]/list[str]/自定义对象；统一转成 JSON 字符串更稳
                "route_json": json.dumps(route, ensure_ascii=False, default=str),
                "route_str": str(route),  # 人类可读备用
            })

    # 输出目录
    os.makedirs(args.save_log_path, exist_ok=True)

    df_summary = pd.DataFrame(summary_records)
    df_routes = pd.DataFrame(route_records)

    # 存 CSV（通用）
    summary_csv = os.path.join(args.save_log_path, "greedy_summary.csv")
    routes_csv  = os.path.join(args.save_log_path, "greedy_routes.csv")
    df_summary.to_csv(summary_csv, index=False)
    df_routes.to_csv(routes_csv, index=False)

    # 可选：存 Parquet（更适合大规模与保类型；需要 pyarrow 或 fastparquet）
    try:
        summary_parquet = os.path.join(args.save_log_path, "greedy_summary.parquet")
        routes_parquet  = os.path.join(args.save_log_path, "greedy_routes.parquet")
        df_summary.to_parquet(summary_parquet, index=False)
        df_routes.to_parquet(routes_parquet, index=False)
    except Exception as e:
        print(f"[WARN] Parquet save skipped: {e}")


    failed_path = os.path.join(args.save_log_path, "unvisited_instances.txt")
    with open(failed_path, "w") as f:
        for x in failed_files:
            f.write(x + "\n")

    print(f"Saved summary -> {summary_csv}")
    print(f"Saved routes  -> {routes_csv}")
    print(f"Saved failed list -> {failed_path}")



if __name__ == "__main__":
    # example.
    # python main.py --instance_path ../../data/solomon_datasets/small_instances/Cplex5er/c101C5.txt --seed 1234 --predefine_route_number 3
    main()
