import logging
from collections import OrderedDict
import random
import pickle
import os

def make_solomon_instance(args):
    return {
            "env": args['env'],
            "depot": args['depot'],
            "customers": args['customers'],
            "charging_stations": args['charging_stations'],
            "demands": args['demands'],
            "tw": args['tw'],
            "service_time": args['service_time'],
        }

def load(file_catalog):
    return pickle.load(open(file_catalog, "rb"))


class lazyClass:
    data = OrderedDict([
        ("train", OrderedDict()),
        ("eval", OrderedDict()),
    ])

    def __init__(self, file_catalog):
        self.file_catalog = file_catalog

    def __getitem__(self, index):
        partition, nodes, idx = index
        if not (partition in self.data) or not (nodes in self.data[partition]):
            logging.warning(
                f"Data sepecified by ({partition}, {nodes}) was not initialized. Attepmting to load it for the first time."
            )
            data = load(self.file_catalog[partition][nodes])
            self.data[partition][nodes] = [make_solomon_instance(instance) for instance in data]
        if idx >= len(self.data[partition][nodes]):
            idx = random.randint(0, len(self.data[partition][nodes]))
        return self.data[partition][nodes][idx]

class EVRPTWDataset:
    def __init__(self, file_catalog):
        self.instance = load(file_catalog)

    def __getitem__(self, index):
        dataset = make_solomon_instance(self.instance[index])
        return dataset

# return 一个 dict_keys(['env', 'depot', 'customers', 'charging_stations', 'demands', 'tw', 'service_time']) context
# dict_keys(['area_size', 'num_customers', 'num_charging_stations', 'num_cluster', 'speed', 'battery_capacity', 'consumption_per_distance', 'loading_capacity', 'charging_speed', 'charging_efficiency', 'instance_startTime', 'instance_endTime', 'working_startTime', 'working_endTime', 'demand', 'cs_time_to_depot'])

# EVRPTWDataset = lazyClass()
# EVRPTWSolomonDataset = lazyClass()
