# VNS-TS for EVRPTW (Electric Vehicle Routing Problem with Time Windows)

A clean and reproducible Python implementation of **Variable Neighborhood Search (VNS)** with **Tabu Search (TS)** refinement for solving the **Electric Vehicle Routing Problem with Time Windows (EVRPTW)**.  
Key features include:
- Adaptive penalty weights for handling constraints
- Simulated annealing–style acceptance criteria
- Optional solution plotting for visualization

---

## Project Structure
```
.
├── main.py
├── solver.py
└── utils
    ├── helpers.py
    ├── __init__.py
    └── load_instances.py
```

---

## 🚀 Quick Start
Run the solver with a sample instance:
```bash
python main.py \
  --instance_path "../../data/solomon_datasets/small_instances/Cplex5er/c101C5.txt" \
  --seed 1234 \
  --predefine_route_number 2
```

---

## Arguments
| Argument                   | Type  | Default      | Description                                       |
| -------------------------- | ----- | ------------ | ------------------------------------------------- |
| `--instance_path`          | `str` | **required** | Path to the EVRPTW instance file                  |
| `--seed`                   | `int` | `1234`       | Random seed for reproducibility                   |
| `--predefine_route_number` | `int` | `2`          | Initial number of routes given to the constructor |


##  Reference
This implementation is based on the following paper:

> Schneider, Michael, Andreas Stenger, and Dominik Goeke.  
> *The electric vehicle-routing problem with time windows and recharging stations.*  
> **Transportation Science**, 48(4), 500–520, 2014.  

Notes:
- Some parameters are not explicitly described in the paper.  
- The original implementation was in **Java**; this repository provides a **Python** implementation with necessary adaptations.


## License
This project is licensed under the Apache-2.0 License.
