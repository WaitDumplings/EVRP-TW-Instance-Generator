from evrptw_gen import InstanceGenerator, EVRPTWDataset
gen = InstanceGenerator("./evrptw_gen/configs/config.yaml", save_path="./Instances_5R", num_instances=10, plot_instances=True)
instances = gen.generate(save_template = "pickle")  # 自动保存到 npz

# # 内存模式
# ds_mem = EVRPTWDataset(instances=instances)
# sample = ds_mem[0]

# # 磁盘模式
# ds_disk = EVRPTWDataset(root_dir="./Instances/")
