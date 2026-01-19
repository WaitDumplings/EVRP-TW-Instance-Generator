from data_parser import parse_file
from EVRP_Graph import Graph_EVRP_TW
from EVRP_Solver import EVPR_TW_Cplex_Solver, EVRP_TW_Gurobi_Solver
import argparse

def main(file_path, dummy=1, mode = "cplex"):
    """
    Main function to run the EVRP-TW solver.
    
    :param file_path: Path to the input file containing the node data.
    """
    # Parse the input file to get depot, customer, RS nodes, and parameters
    Depot_nodes, Customer_nodes, RS_nodes, parameters = parse_file(file_path)
    
    # Create a graph instance for the EVRP-TW
    Graph = Graph_EVRP_TW(Depot_nodes, Customer_nodes, RS_nodes, parameters, RS_dummy_count=dummy)
    
    # Create a solver instance and solve the problem
    if mode == "cplex":
        Solver = EVPR_TW_Cplex_Solver(Graph)
    else:
        print(f"No such a solver named {mode} was find in the candidate! Please try cplex instead")
        return

    # Solve
    Solver.solver()
    # Print the results of the solver
    Solver.print_results(Optimal_Value=True, DV_Info=False, Routes=True)

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run EVRP-TW solver with the provided input file.")
    
    # Define the file path argument
    parser.add_argument('--file_path',required=True, type=str, help="Path to the input text file.")

    # Define the number of dummy nodes
    parser.add_argument('--dummy', type=int, default=1 ,help="Path to the input text file.")

    # Define Solver
    parser.add_argument('--solver', type=str, default="cplex", help="Solver type cplex")

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function with the provided file path
    main(args.file_path, args.dummy, args.solver)


    
