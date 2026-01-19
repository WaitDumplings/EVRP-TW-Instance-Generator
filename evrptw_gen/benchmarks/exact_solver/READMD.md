# EVRP-TW Solver

A Python implementation for solving the **Electric Vehicle Routing Problem with Time Windows (EVRP-TW)** using MILP solvers.  
Currently, the solver supports **CPLEX**.

---

## 🚀 Quick Start

Run the solver with a given instance file:

```bash
python main.py \
  --file_path "../../data/solomon_datasets/small_instances/Cplex5er/c101C5.txt" \
  --dummy 3 \
  --solver cplex
```

> Loads the instance file c101C5.txt
> Creates 3 dummy nodes for each recharging station
> Solves the EVRP-TW using CPLEX
> Prints the optimal value and routes

---

## Arguments
| Argument      | Type  | Default      | Description                                                                                                                                                                                     |
| ------------- | ----- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--file_path` | `str` | **required** | Path to the EVRP-TW instance file (text format). The file should contain depot, customer, and recharging station data with problem parameters.                                                  |
| `--dummy`     | `int` | `3`          | Number of **dummy nodes** per recharging station. Dummy nodes are used to model multiple visits or partial charging at the same station by creating virtual copies of the station in the graph. |
| `--solver`    | `str` | `"cplex"`    | Solver type (currently only `cplex` is supported).                                                                                                                                              |

---

## License
This project is licensed under the Apache-2.0 License.