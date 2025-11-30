import os
import numpy as np
import time
import matplotlib.pyplot as plt

def _to_2d_points(arr):
    """Coerce array-like into (N,2) float ndarray."""
    a = np.asarray(arr, dtype=float)
    a = np.atleast_2d(a)
    if a.shape[1] != 2:
        if a.ndim == 2 and a.shape[0] == 2:  # e.g., [[x,...],[y,...]] -> transpose
            a = a.T
        else:
            a = a[:, :2]
    return a

def _as_float(x):
    """Robustly convert x (scalar/list/ndarray) to a Python float (take the first element if needed)."""
    arr = np.asarray(x).reshape(-1)
    return float(arr[0])

def _fmt2(x, width):
    """Format a number to a fixed-width string with 2 decimal places, right-aligned."""
    return f"{format(_as_float(x), '.2f'):>{width}}"

def _xy(pos):
    """Return (x, y) as floats from any pos shape: (2,), (1,2), [[x,y]], etc."""
    arr = np.asarray(pos, dtype=float).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Position must have at least 2 numbers, got: {pos}")
    return float(arr[0]), float(arr[1])

def save_instances(instances, save_path, template='solomon'):
    """Save EVRP instances in the Solomon dataset format."""
    os.makedirs(save_path, exist_ok=True)

    if template == 'solomon':
        timestamp = int(time.time())
        
        for i in range(len(instances)):
            time_window_type = instances[i]['env']['time_window_type']
            instance_type = instances[i]['env']['instance_type']
            save_file_name = f'solomon_dataset_{i}_{instance_type}_{time_window_type}_{timestamp}.txt'

            inst = instances[i]
            instance_end_time = inst['env'].get('instance_endTime', 1440.0) / 60

            depot_pos = inst['depot']                      # e.g., np.array([13.55, 45.28]) or [[13.55, 45.28]]
            charging_station_pos = np.concatenate((depot_pos, inst['charging_stations']), axis = 0)  # list of positions
            customers = inst['customers']                  # (x, y, demand, readytime, latesttime, servicetime)
            # breakpoint()
            Q = inst['env']['battery_capacity']
            C = inst['env']['loading_capacity']
            r = inst['env']['consumption_per_distance']   # consumption rate (kWh/km)
            g = 1 / inst['env']['charging_speed']        # inverse refueling rate
            v = inst['env']['speed']

            save_name = os.path.join(save_path, save_file_name)
            with open(save_name, "w") as f:
                # Header
                f.write("StringID   Type       x          y          demand     ReadyTime  DueDate    ServiceTime\n")

                # Depot
                dx, dy = _xy(depot_pos)
                line = (
                    "D0         d          "
                    f"{_fmt2(dx, 10)}{_fmt2(dy, 10)}"
                    f"{_fmt2(0.0, 11)}{_fmt2(00.0, 11)}{_fmt2(instance_end_time, 11)}{_fmt2(0.0, 11)}\n"
                )
                f.write(line)

                # Charging stations
                for j, pos in enumerate(charging_station_pos):
                    sx, sy = _xy(pos)
                    line = (
                        f"S{j}         f          "
                        f"{_fmt2(sx, 10)}{_fmt2(sy, 10)}"
                        f"{_fmt2(0.0, 11)}{_fmt2(0.0, 11)}{_fmt2(instance_end_time, 11)}{_fmt2(0.0, 11)}\n"
                    )
                    f.write(line)

                # Customers
                for k, c in enumerate(customers):
                    # c may be list/tuple/ndarray, possibly nested; robustly extract six fields
                    arr = np.asarray(c, dtype=float).reshape(-1)

                    if arr.size < 6:
                        raise ValueError(f"Customer row must have 6 numbers (x,y,demand,ready,due,service), got: {c}")
                    x, y, demand, ready, due, service = arr[:6]
                    ready  # convert minutes to hours
                    due

                    line = (
                        f"C{k}         c          "
                        f"{_fmt2(x, 10)}{_fmt2(y, 10)}"
                        f"{_fmt2(demand, 11)}{_fmt2(ready, 11)}{_fmt2(due, 11)}{_fmt2(service, 11)}\n"
                    )
                    f.write(line)

                # Environment parameters (two decimals where sensible)
                f.write("\n")
                f.write(f"Q Vehicle fuel tank capacity /{format(_as_float(Q), '.2f')}/\n")
                f.write(f"C Vehicle load capacity /{format(_as_float(C), '.2f')}/\n")
                f.write(f"r fuel consumption rate /{format(_as_float(r), '.2f')}/\n")
                f.write(f"g inverse refueling rate /{format(_as_float(g), '.4f')}/\n")
                f.write(f"v average Velocity /{format(_as_float(v), '.2f')}/\n")

            print(f"✅ Saved: {save_name}")


def plot_instance(instances, save_path):
    os.makedirs(save_path, exist_ok=True)
    timestamp = int(time.time())

    for i in range(len(instances)):
        inst = instances[i]
        time_window_type = inst['env']['time_window_type']
        instance_type = inst['env']['instance_type']
        save_file_name = f'instance_{i}_{instance_type}_{time_window_type}_{timestamp}.png'

        # --- extract & sanitize points ---
        depot_pos = _to_2d_points(inst['depot'])
        cs_pos = _to_2d_points(inst.get('charging_stations', np.empty((0, 2))))
        cust_raw = np.asarray(inst.get('customers', np.empty((0, 2))), dtype=float)
        customers_pos = _to_2d_points(cust_raw)[:, :2] if cust_raw.size else np.empty((0, 2))

        # --- axis ranges ---
        x_range = inst['env']['area_size'][0]
        y_range = inst['env']['area_size'][1]

        # --- plot ---
        fig, ax = plt.subplots(figsize=(6, 6))

        if customers_pos.size:
            ax.scatter(customers_pos[:, 0], customers_pos[:, 1],
                       s=15, marker='o', label='Customers')
        if cs_pos.size:
            ax.scatter(cs_pos[:, 0], cs_pos[:, 1],
                       s=60, marker='^', label='Charging Stations')
        if depot_pos.size:
            ax.scatter(depot_pos[:, 0], depot_pos[:, 1],
                       s=120, marker='*', label='Depot')

        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'EVRP Instance {i}')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linewidth=0.5, alpha=0.4)
        ax.legend(loc='best')

        # --- save ---
        out_path = os.path.join(save_path, save_file_name)
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)