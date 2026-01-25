import random

import random

class NodesGeneratorScheduler:
    def __init__(self, min_customer_num, max_customer_num, cus_per_cs, seed=None):
        self.min_customer_num = min_customer_num
        self.max_customer_num = max_customer_num
        self.cus_per_cs = cus_per_cs

        self.rng = random.Random(seed)

        self.registry = {
            "linear": self.linear_scheduler,
            "fixed": self.fixed_scheduler,
        }

    def linear_scheduler(self):
        random_cus = self.rng.randint(0, self.max_customer_num - self.min_customer_num)
        num_cus = random_cus + self.min_customer_num
        num_cs_ub = max(1, num_cus // self.cus_per_cs)
        num_cs = max(1, self.rng.randint(num_cs_ub//2, num_cs_ub))
        return num_cus, num_cs

    
    def fixed_scheduler(self):
        num_cus = self.max_customer_num
        num_cs = max(1, num_cus // self.cus_per_cs)
        return num_cus, num_cs

    def __call__(self, policy_name="linear"):
        try:
            fn = self.registry[policy_name]
        except KeyError:
            raise ValueError(f"Unknown policy name: {policy_name}")
        return fn()
