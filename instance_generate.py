from evrptw_gen import InstanceGenerator, EVRPTWDataset
from evrptw_gen.configs.load_config import Config
from evrptw_gen.utils.nodes_generator_scheduler import NodesGeneratorScheduler

def main(args):
    config_path = args.config_path
    save_path = args.save_path
    num_instances = args.num_instances
    plot_instances = args.plot_instances
    customer_range = args.customer_range
    node_generate_policy = args.node_generate_policy
    cus_per_cs = args.cus_per_cs
    perturb_config_path = args.perturb_config_path
    save_format = args.save_format

    min_customer_num, max_customer_num = customer_range

    if save_format == "all":
        save_template_solomon = True
        save_template_pickle = True
    elif save_format == "solomon":
        save_template_solomon = True
        save_template_pickle = False
    elif save_format == "pickle":
        save_template_solomon = False
        save_template_pickle = True
    else:
        raise ValueError(f"Unknown save_format: {save_format}")

    gen = InstanceGenerator(config_path, save_path=save_path, num_instances=num_instances, plot_instances=plot_instances)
    nodes_generatro_scheduler = NodesGeneratorScheduler(min_customer_num=min_customer_num, max_customer_num=max_customer_num, cus_per_cs=cus_per_cs)
    perturb_dict = Config(perturb_config_path).setup_env_parameters()

    gen.generate(perturb_dict=perturb_dict['perturb'],
                save_template_solomon=save_template_solomon,
                save_template_pickle=save_template_pickle,
                node_generater_scheduler=nodes_generatro_scheduler,
                node_generate_policy=node_generate_policy) 


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate EVRPTW instances based on a configuration file.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./evrptw_gen/configs/config.yaml",
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./eval_data_1000",
        help="Directory to save generated instances."
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=64,
        help="Number of instances to generate."
    )
    parser.add_argument(
        "--plot_instances",
        action='store_true',
        help="Whether to plot the generated instances."
    )
    parser.add_argument(
        "--customer_range",
        type=int,
        nargs=2,
        default=[5, 5],
        help="Range of number of customers (min max)."
    )
    parser.add_argument(
        "--node_generate_policy",
        type=str,
        default="fixed",
        help="Node generation policy: 'fixed' or 'linear'."
    )
    parser.add_argument(
        "--cus_per_cs",
        type=int,
        default=2,
        help="Number of customers per charging station (used if node_generate_policy is 'fixed_cus_per_cs')."
    )
    parser.add_argument(
        "--add_perturb",
        action='store_true',
        help="Whether to add perturbations to the instances."
    )
    parser.add_argument(
        "--perturb_config_path",
        type=str,
        default="./evrptw_gen/configs/perturb_config.yaml",
        help="Path to the perturbation configuration file."
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="all",
        help="Format to save instances: 'all', 'solomon', 'pickle'."
    )


    args = parser.parse_args()
    main(args)
