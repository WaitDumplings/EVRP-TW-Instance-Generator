from evrptw_gen import InstanceGenerator, EVRPTWDataset
from evrptw_gen.configs.load_config import Config
from evrptw_gen.utils.nodes_generatro_scheduler import NodesGeneratorScheduler
gen = InstanceGenerator("./evrptw_gen/configs/config.yaml", save_path="./eval_data_1000", num_instances=64, plot_instances=False)

# Generate 可以设置一个Num_Cus, Num_CS 的scheduler
nodes_generatro_scheduler = NodesGeneratorScheduler(min_customer_num=1000, max_customer_num=1000, cus_per_cs=25)
perturb_dict = Config("./evrptw_gen/configs/perturb_config.yaml").setup_env_parameters()

instances = gen.generate(perturb_dict=perturb_dict['perturb'],
                         save_template_solomon=True,
                         save_template_pickle=True,
                         node_generater_scheduler=nodes_generatro_scheduler,
                         node_generate_policy="linear")  # 自动保存到 npz

# # 内存模式
# ds_mem = EVRPTWDataset(instances=instances)
# sample = ds_mem[0]

# # 磁盘模式
# ds_disk = EVRPTWDataset(root_dir="./Instances/")
