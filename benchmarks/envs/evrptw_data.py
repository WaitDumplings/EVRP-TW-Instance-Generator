import logging
import pickle
from collections import OrderedDict
import random

root_dir = "./data/solomon_evrptw_input/"


def load(filename, root_dir=root_dir):
    return pickle.load(open(root_dir + filename, "rb"))

file_catalog = {
    "test": {
        5: "evrptw5_test_seed1234.pkl",
        10: "evrptw10_test_seed1234.pkl",
        15: "evrptw15_test_seed1234.pkl",
        20: "evrptw20_test_seed1234.pkl",
        50: "evrptw50_test_seed4321.pkl",
        100: "evrptw100_test_seed1234.pkl",
    },
    "eval": {
        # 5: "evrptw5_validation_seed0.pkl",
        5: "evrptw5_validation_seed0.pkl",
        10: "evrptw10_validation_seed4321.pkl",
        15: "evrptw15_validation_seed0.pkl",
        20: "evrptw20_validation_seed4321.pkl",
        50: "evrptw50_validation_seed4321.pkl",
        100: "evrptw100_validation_seed0.pkl",
    },
}

def make_solomon_instance(args):
    return {
            "depot_loc": args["depot_loc"],  # Shape: (instances_number, 1, 2)
            "cus_loc": args["cus_loc"],  # Shape: (instances_number, customer_size, 2)
            "rs_loc": args["rs_loc"],  # Shape: (instances_number, rs_size, 2)
            "time_window": args["time_window"],  # Shape: (instances_number, 1+rs_size+customer_size, 2)
            "demand": args["demand"],  # Shape: (instances_number, 1+rs_size+customer_size)
            "max_time": args["max_time"],  # Shape: (instances_number,)
            "demand_capacity": args["demand_capacity"],  # Shape: (instances_number,)
            "battery_capacity": args["battery_capacity"],  # Shape: (instances_number,)
            "types": args["types"],  # Shape: (instances_number,)
            "velocity_base": args["velocity_base"],  # Scalar
            "energy_consumption": args["energy_consumption"],  # Scalar
            "service_time": args["service_time"],
            "charging_rate": args["charging_rate"],
            "instance_mask": args["instance_mask"]
        }


class lazyClass:
    data = OrderedDict([
        ("test", OrderedDict()),
        ("eval", OrderedDict()),
    ])

    def __getitem__(self, index):
        partition, nodes, idx = index
        if not (partition in self.data) or not (nodes in self.data[partition]):
            logging.warning(
                f"Data sepecified by ({partition}, {nodes}) was not initialized. Attepmting to load it for the first time."
            )
            data = load(file_catalog[partition][nodes])
            breakpoint()
            self.data[partition][nodes] = [make_solomon_instance(instance) for instance in data]
        if idx >= len(self.data[partition][nodes]):
            idx = random.randint(0, len(self.data[partition][nodes]))
        return self.data[partition][nodes][idx]

EVRPTWDataset = lazyClass()
EVRPTWSolomonDataset = lazyClass()
