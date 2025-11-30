from evrptw_gen import InstanceGenerator, EVRPTWDataset

gen = InstanceGenerator("./configs/config.yaml", save_path="./Instances/npz", num_instances=10)
instances = gen.generate()  # 自动保存到 npz

# 内存模式
ds_mem = EVRPTWDataset(instances=instances)
sample = ds_mem[0]

# 磁盘模式
ds_disk = EVRPTWDataset(root_dir="./Instances/npz")

