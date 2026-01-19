from utility import find_and_print_routes
import cplex
from cplex.exceptions import CplexError

class EVPR_TW_Cplex_Solver:
    """
    This class defines a solver embedded in CPLEX for solving the Electric Vehicle Routing Problem with Time Windows (EVRP-TW).

    Prerequisites for using this class:
    1. A Graph object should be provided containing:
        - Customer_Idx: List of customer node indices.
        - RS_Idx: List of recharging station node indices.
        - Depot_Idx: List containing start and end depot node indices.
        - Vehicle load capacity (C), battery capacity (Q), battery consumption rate (h), charging speed (g).
        - Distance matrix (distance_matrix) and travel time matrix (Travel_Time).
        - Customer demands, service times, and time windows.
    2. Only one depot is supported for the EVRP-TW.
    3. Constraints are based on "The Electric Vehicle-Routing Problem with Time Windows and Recharging Stations" by Michael Schneider.
       These can be modified to explore other EVRP-TW variants.
    """

    def __init__(self, Graph, l_0=1000):
        """Initialize the EVPR_TW_Solver with a given graph and penalty constant l_0."""
        self.model = cplex.Cplex()
        self.path_link_variable = []  # List to store decision variables for path links
        self.objective_function_coef = []  # List to store the coefficients for the objective function

        # Node index information
        self.V_sequence = Graph.Customer_Idx  # Customer node indices
        self.F_Prime_sequence = Graph.RS_Idx  # Recharging station node indices
        self.D = Graph.Depot_Idx  # Depot node indices (start and end)

        # Depot information
        self.Depot_start = [0]  # Start depot index
        self.Depot_end = [len(Graph.Nodes) - 1]  # End depot index

        # Sequences for node sets
        self.V_Prime_N_plus_1_sequence = self.V_sequence + self.F_Prime_sequence + [self.D[-1]]  # All nodes + end depot
        self.V_Prime_0_sequence = [self.D[0]] + self.V_sequence + self.F_Prime_sequence  # Start depot + all nodes
        self.V_Prime_sequence = self.V_sequence + self.F_Prime_sequence  # All customer and RS nodes

        # Graph-related attributes (parameters)
        self.C = Graph.C  # Vehicle load capacity
        self.Q = Graph.Q  # Vehicle battery capacity
        self.h = Graph.r  # Fuel consumption rate (per distance)
        self.g = Graph.g  # Charging speed
        self.l_0 = l_0  # Large constant for subtour elimination

        # Distance and time data
        self.distance_matrix = Graph.distance_matrix  # Distance matrix between nodes
        self.model_size = len(Graph.Nodes)  # Total number of nodes
        self.CustomerTimeWindow = Graph.CustomerTW  # Time windows for customer nodes
        self.CustomerService = Graph.CustomerServiceTime  # Service times for customers
        self.CustomerDemand = Graph.CustomerDemand  # Customer demands
        self.Travel_Time = Graph.travel_time_matrix  # Travel time between nodes

        # Initialize the optimization model
        self._initialize_model()

    def _initialize_model(self):
        """Initializes the CPLEX model by setting decision variables, objective function, and constraints."""
        self.model.parameters.timelimit.set(7200)  # Maximal computing time = 7200s
        self._set_decision_variable()  # Define decision variables
        self._set_objective_function()  # Define the objective function
        self._set_constraints()  # Define constraints

    def _set_decision_variable(self):
        """Define decision variables for path links, time, load, and battery levels."""
        # Path link variables (binary)
        self.path_link_variable = [f"x_{i}_{j}" for i in range(self.model_size) for j in range(self.model_size) if i != j]
        self.objective_function_coef = [self.distance_matrix[i][j] for i in range(self.model_size) for j in range(self.model_size) if i != j]

        # Continuous variables for time (tao), load (u), and battery (y)
        self.tao_names = [f"tao_{i}" for i in range(self.model_size)]
        self.u_names = [f"u_{i}" for i in range(self.model_size)]
        self.y_names = [f"y_{i}" for i in range(self.model_size)]

    def _set_objective_function(self):
        """Define the objective function (minimizing the total travel distance)."""
        self.model.objective.set_sense(self.model.objective.sense.minimize)

        # Add path link variables to the model with objective function coefficients
        self.model.variables.add(names=self.path_link_variable, obj=self.objective_function_coef, types=["B"] * len(self.path_link_variable))
        
        # Add continuous variables for time, load, and battery with appropriate bounds
        for i in range(self.model_size):
            self.model.variables.add(names=[self.tao_names[i]], obj=[0], types=["C"], lb=[self.CustomerTimeWindow[i][0]], ub=[self.CustomerTimeWindow[i][1]])
            self.model.variables.add(names=[self.u_names[i]], obj=[0], types=["C"], lb=[0], ub=[self.C])
            Initial_Battery = self.Q if i == 0 else 0  # Initial battery for the depot is full
            self.model.variables.add(names=[self.y_names[i]], obj=[0], types=["C"], lb=[Initial_Battery], ub=[self.Q])

    def _set_constraints(self):
        """Define the constraints for the EVRP-TW problem."""
        # Constraints for customers (each customer must be visited exactly once)
        for i in self.V_sequence:
            self.model.linear_constraints.add(
                lin_expr=[[ [f"x_{i}_{j}" for j in self.V_Prime_N_plus_1_sequence if i != j], [1] * (len(self.V_Prime_N_plus_1_sequence) - 1) ]],
                senses=["E"],
                rhs=[1]  # Exactly one outgoing edge from customer i
            )

        # Constraints for recharging stations (at most one visit per RS in each route)
        for i in self.F_Prime_sequence:
            self.model.linear_constraints.add(
                lin_expr=[[ [f"x_{i}_{j}" for j in self.V_Prime_N_plus_1_sequence if i != j], [1] * (len(self.V_Prime_N_plus_1_sequence) - 1) ]],
                senses=["L"],
                rhs=[1]  # At most one outgoing edge from RS i in this route
            )

        # Route consistency constraints
        for j in self.V_Prime_sequence:
            in_vars = [f"x_{j}_{i_start}" for i_start in self.V_Prime_N_plus_1_sequence if i_start != j]
            out_vars = [f"x_{i_end}_{j}" for i_end in self.V_Prime_0_sequence if i_end != j]
            self.model.linear_constraints.add(
                lin_expr=[[in_vars + out_vars, [1] * len(in_vars) + [-1] * len(out_vars)]],
                senses=["E"],
                rhs=[0]  # Flow balance
            )

        # Subtour elimination: travel time and battery constraints
        # Travel Time Constraint(Customer)
        # tao_i - tao_j + (t_ij + s_i + l_0)x_ij <= l_0
        for i in self.Depot_start + self.V_sequence:
            for j in self.V_Prime_N_plus_1_sequence:
                if i != j:
                    self.model.linear_constraints.add(
                        lin_expr=[[ [f"tao_{i}", f"tao_{j}", f"x_{i}_{j}"], [1, -1, self.Travel_Time[i][j] + self.CustomerService[i] + self.l_0] ]],
                        senses=["L"],
                        rhs=[self.l_0]
                    )
        # Travel Time Constraint(RS)
        # tao_i - tao_j + (l_0 + g*Q)x_ij -g*y_i <= l_0
        for i in self.F_Prime_sequence:
            for j in self.V_Prime_N_plus_1_sequence:
                if i != j:
                    self.model.linear_constraints.add(
                        lin_expr=[[ [f"tao_{i}", f"tao_{j}", f"x_{i}_{j}", f"y_{i}"], [1, -1, self.l_0 + self.g * self.Q, -self.g] ]],
                        senses=["L"],
                        rhs=[self.l_0]
                    )

        # Load Capacity Constraint
        # u_j - u_i +(C + q_i)x_ij <= C
        for i in self.V_Prime_0_sequence:
            for j in self.V_Prime_N_plus_1_sequence:
                if i != j:
                    self.model.linear_constraints.add(
                        lin_expr=[[ [f"u_{j}", f"u_{i}", f"x_{i}_{j}"], [1, -1, self.C + self.CustomerDemand[i]] ]],
                        senses=["L"],
                        rhs=[self.C]
                    )

        # Battery Constraints(Customers)
        # y_j - y_i + (h*d_ij +Q)x_ij <= Q
        for i in self.V_sequence:
            for j in self.V_Prime_N_plus_1_sequence:
                if i != j:
                    self.model.linear_constraints.add(
                        lin_expr=[[ [f"y_{j}", f"y_{i}", f"x_{i}_{j}"], [1, -1, self.h * self.distance_matrix[i][j] + self.Q] ]],
                        senses=["L"],
                        rhs=[self.Q]
                    )

        # Battery Constraints(RS)
        # y_j + h*d_ij*x_ij <= Q
        for i in self.Depot_start + self.F_Prime_sequence:
            for j in self.V_Prime_N_plus_1_sequence:
                if i != j:
                    self.model.linear_constraints.add(
                        lin_expr=[[ [f"y_{j}", f"x_{i}_{j}"], [1, self.h * self.distance_matrix[i][j]] ]],
                        senses=["L"],
                        rhs=[self.Q]
                    )

    def solver(self):
        """Solve the EVRP-TW problem using CPLEX."""
        self.model.solve()

    def print_results(self, Optimal_Value=True, DV_Info=True, Routes=True):
        """Print the results including optimal objective value and decision variables."""
        if Optimal_Value:
            print("Optimal Value: ", self.model.solution.get_objective_value())

        if DV_Info or Routes:
            Routes = {}
            for i in range(self.model_size):
                for j in range(self.model_size):
                    if i != j and self.model.solution.get_values(f"x_{i}_{j}") > 0.5:
                        if DV_Info:
                            print(f"x_{i}_{j}:", self.model.solution.get_values(f"x_{i}_{j}"))
                        Routes[f"x_{i}_{j}"] = 1

            if DV_Info:
                print("Travel Time: ", {f"tao_{i}": self.model.solution.get_values(f"tao_{i}") for i in range(self.model_size)})
                print("Load Capacity: ", {f"u_{i}": self.model.solution.get_values(f"u_{i}") for i in range(self.model_size)})
                print("Battery Level: ", {f"y_{i}": self.model.solution.get_values(f"y_{i}") for i in range(self.model_size)})

            if Routes:
                find_and_print_routes(Routes, end_node_idx=self.D[-1])  # Use utility function to print routes