import random
import math
import copy
from collections import deque, defaultdict
import numpy as np
from tqdm import tqdm
import time
from utils.load_instances import load_instance, Route

class VNSTSolver:
    def __init__(self, instance, predefine_route_number=3):
        self.instance = instance
        self.tabu_list = deque(maxlen=30)
        self.recharging_stations = np.array([[instance.stations[i].x, instance.stations[i].y] for i in range(len(instance.stations))])

        # Simulated annealing parameters
        self.temp = -1  # initial temperature

        # Tabu search parameters
        self.tabu_tenure = 30  # max tabu list length
        self.tabu_iter = 100    # number of tabu search iterations

        # Penalty parameters (as in the paper)
        self.alpha, self.beta, self.gamma = 10.0, 10.0, 10.0
        self.alpha_min, self.beta_min, self.gamma_min = 0.5, 0.75, 1.0
        self.alpha_max, self.beta_max, self.gamma_max = 5000, 5000, 5000

        # VNS parameters
        self.k_max = 15      # max number of neighborhoods
        self.η_feas = 700    # max attempts in infeasible phase
        self.η_dist = 100    # max attempts in feasible phase
        self.nearest_station = self.battery_to_nearest_rs(instance.depot)
        self.instance_dist_matrix_calculatrion()
        self.attribute_frequency = defaultdict(int)
        self.attribute_total = 0
        self.lambda_div = 1.0
        self.delta_sa = 0.08
        self.predefine_route_number = predefine_route_number
        self.global_value = 1e10
        self.time_matrix = self.dist_matrix / instance.vehicle_params['velocity']

    def time_cost(self, node1, node2):
        i = self.node_id[node1.id]
        j = self.node_id[node2.id]
        return self.time_matrix[i][j]

    def instance_dist_matrix_calculatrion(self):
        self.node_id = {self.instance.depot.id: 0}
        for i in range(len(self.instance.stations)):
            self.node_id[self.instance.stations[i].id] = i + 1
        offset = i + 1
        for i in range(len(self.instance.customers)):
            self.node_id[self.instance.customers[i].id] = i + 1 + offset
        self.dist_matrix = self.instance.dist_matrix

    def battery_to_nearest_rs(self, node):
        """Precompute distance (as fuel) from each customer to its nearest charging station."""
        self.nearest_station = {self.instance.depot.id: 0}
        self.nearest_station_idx = {}

        for i in range(len(self.instance.stations)):
            self.nearest_station[self.instance.stations[i].id] = 0

        for i in range(len(self.instance.customers)):
            pos = np.array([self.instance.customers[i].x, self.instance.customers[i].y])
            distances = np.linalg.norm(self.recharging_stations - pos, axis=1)
            nearest_station = self.instance.stations[np.argmin(distances)]
            self.nearest_station_idx[self.instance.customers[i].id] = nearest_station.id
            self.nearest_station[self.instance.customers[i].id] = self.instance.vehicle_params['consump_rate'] * np.min(distances) / self.instance.vehicle_params['velocity']

    def fuel_consumption(self, node1, node2):
        return self.instance.vehicle_params['consump_rate'] * self.time_cost(node1, node2)

    def solve(self):
        """VNS/TS framework based on Figure 1 in the paper."""
        S = self.initial_solution()
        self.global_solution = S[:]
        κ = 1   # current neighborhood index
        i = 0   # iteration counter
        feasibilityPhase = True  # start in feasible phase
        best_solution = copy.deepcopy(S)
        best_value = 1e10
        
        # Initialize tqdm (rough upper bound on steps)
        pbar = tqdm(total=self.η_dist + self.η_feas)
        while feasibilityPhase or (not feasibilityPhase and i < self.η_dist):
            start_time = time.time()
            S_prime = self.vns_perturb(S, κ)
            S_double_prime = self.apply_tabu_search(S_prime)

            if self.accept_sa(S_double_prime, S):
                S = S_double_prime[:]
                κ = 1
            else:
                κ = (κ % self.k_max) + 1

            S_value = self.generalized_cost(S, penalty_value=False, p_div_value=False, allow_infeasible=False)
            if self.is_solution_feasible(S) and S_value < best_value:
                best_solution = copy.deepcopy(S)
                best_value = S_value
                if self.global_value > best_value:
                    self.global_value = best_value
                    self.global_solution = best_solution

            if feasibilityPhase:
                if not self.is_solution_feasible(S):
                    if i == self.η_feas:
                        S = self.add_vehicle(S)
                        i -= 1
                else:
                    feasibilityPhase = False
                    i = 0  # switch to non-feasible phase
                    pbar.reset(total=self.η_dist)
            self.update_penalty_weights(S, i)
            i += 1
            pbar.update(1)

        pbar.close()
        return self.global_solution 

    def apply_tabu_search(self, S_prime):
        """Apply tabu search to S'."""
        return self._tabu_search(S_prime)

    def accept_sa(self, S_double_prime, S):
        """SA acceptance criterion."""
        cost_diff = self.generalized_cost(S_double_prime, penalty_value=False, p_div_value=False, allow_infeasible=False) - self.generalized_cost(S, penalty_value=False, p_div_value=False, allow_infeasible=False)

        if cost_diff <= 0:
            return True  # accept better solution

        if self.temp == -1 and cost_diff > 0:
            # Paper's heuristic: initialize temperature at first worsening move
            self.temp = -cost_diff / math.log(0.5)
            self.cooling = (1 - self.delta_sa)
        else:
            # linear cooling
            self.temp *= self.cooling  

        # SA acceptance
        return random.random() < math.exp(-cost_diff / self.temp)

    def add_vehicle(self, S):
        """Add an additional vehicle by splitting infeasible routes."""
        new_route = []
        candidate_customer = []
        for route in S:
            if self.is_route_feasible(route):
                new_route.append(route)
                continue

            route_add = self.copy_route(route)
            route_length = len(route_add.nodes)
            search_idx = 1

            while search_idx < route_length - 1 and not self.is_route_feasible(route_add):
                current_node = route_add.nodes[search_idx]
                if self.violates_constraints(route_add, search_idx):
                    if current_node.type == 'c':
                        candidate_customer.append(current_node)
                        route_add.nodes.pop(search_idx)
                        route_length -= 1
                    else:
                        route_add.nodes.pop(search_idx)
                else:                    
                    search_idx += 1

            if len(route_add.nodes) > 2:
                new_route.append(route_add)
            
        route_add = self.create_new_route()
        route_add.nodes.append(self.instance.depot)
        candidate_routes = []
        # breakpoint()
        while len(candidate_customer) > 0:
            current_node = candidate_customer.pop()
            
            if not candidate_routes:
                route_add.nodes.insert(-1, current_node)
                candidate_routes.append(route_add)
            else:
                insert_success = False
                for route in candidate_routes:
                    for insert_pos in reversed(range(1, len(route.nodes) - 1)):
                        route.nodes.insert(insert_pos, current_node)
                        if self.is_route_feasible(route):
                            insert_success = True
                            break
                        else:
                            route.nodes.pop(insert_pos)  # revert insertion
                    if insert_success:
                        break
                
                if not insert_success:
                    route_add = self.create_new_route()
                    route_add.nodes.append(current_node)
                    route_add.nodes.append(self.instance.depot)
                    candidate_routes.append(route_add)
        
        new_route.extend(candidate_routes)
        return new_route

    def violates_constraints(self, route, search_idx):
        """Check whether the route still violates constraints after removing node at search_idx."""
        new_route = self.copy_route(route)
        new_route.nodes = new_route.nodes[:search_idx + 1]
        if new_route.nodes[-1].type != 'd':
            new_route.nodes.append(self.instance.depot)
        return self.battery_violation(new_route) or self.time_violation(new_route) or self.load_violation(new_route)

    def battery_violation(self, route, node=None):
        """Check whether battery constraints are violated (optionally with an additional node)."""
        current_fuel = self.instance.vehicle_params['fuel_cap']

        for i in range(len(route.nodes) - 1):
            from_node = route.nodes[i]
            to_node = route.nodes[i + 1]
            fuel_needed = self.fuel_consumption(from_node, to_node)

            current_fuel -= fuel_needed
            if current_fuel < 0:
                return True  # insufficient battery

            # recharge at station
            if to_node.type == 'f':
                current_fuel = self.instance.vehicle_params['fuel_cap']

        # check for an extra node
        if node is not None:
            fuel_needed = self.fuel_consumption(route.nodes[-1], node)
            current_fuel -= fuel_needed
            if current_fuel < 0:
                return True

        return False

    def battery_penalty(self, route):
        """Compute cumulative battery overflow along the route (used as penalty)."""
        current_fuel = self.instance.vehicle_params['fuel_cap']
        battery_penalty_value = 0
        gamma_in = 0
        for i in range(len(route.nodes) - 1):
            from_node = route.nodes[i]
            to_node = route.nodes[i + 1]
            fuel_needed = self.fuel_consumption(from_node, to_node)

            gamma_in += fuel_needed
            battery_penalty_value += max(0, gamma_in - self.instance.vehicle_params['fuel_cap'])
                
            # recharge at station
            if to_node.type == 'f':
                gamma_in = 0
        return battery_penalty_value

    def load_violation(self, route, node=None):
        """Check vehicle capacity constraint (optionally with an additional node)."""
        if node is not None:
            return sum(node.demand for node in route.nodes) + node.demand > self.instance.vehicle_params['load_cap']
        return sum(node.demand for node in route.nodes) > self.instance.vehicle_params['load_cap']
    
    def load_penalty(self, route):
        """Capacity overflow used as penalty."""
        return max(0, sum(node.demand for node in route.nodes) - self.instance.vehicle_params['load_cap'])

    def time_violation(self, route, node=None):
        """Check time-window feasibility (optionally with an additional node)."""
        current_time = 0
        battery_use = 0
        vehicle_params = self.instance.vehicle_params
        charge_rate = vehicle_params['charge_rate']
        
        for i in range(len(route.nodes)):
            node_i = route.nodes[i]
            
            # arrival time cannot be earlier than ready time
            arrival_time = max(current_time, node_i.ready)
            
            # time-window violation
            if arrival_time > node_i.due:
                return True
            
            # battery update or charging
            if node_i.type == "c":  # customer
                if i < len(route.nodes) - 1:
                    battery_use += self.dist_matrix[self.node_id[node_i.id], self.node_id[route.nodes[i-1].id]]
                current_time = arrival_time + node_i.service

            elif node_i.type == "f":  # charging station
                battery_use += self.dist_matrix[self.node_id[node_i.id], self.node_id[route.nodes[i-1].id]]
                current_time += battery_use / charge_rate
                battery_use = 0  # reset after charge

            # travel time to next node
            if i < len(route.nodes) - 1:
                current_time += self.time_cost(node_i, route.nodes[i + 1])

        # optionally check adding a new node
        if node is not None:
            last_node = route.nodes[-1] if route.nodes else None
            if last_node:
                projected_arrival = current_time + self.time_cost(last_node, node)
                if projected_arrival > node.due:
                    return True

        return False

    def time_penalty(self, route):
        """Compute time-window violation penalty including charging time at stations."""
        time_penalty_value = 0
        current_time = 0
        battery_use = 0
        vehicle_params = self.instance.vehicle_params
        charge_rate = vehicle_params['charge_rate']

        for i in range(len(route.nodes)):
            node_i = route.nodes[i]

            # cannot arrive earlier than ready time
            arrival_time = max(current_time, node_i.ready)

            # accumulate lateness penalty
            if arrival_time > node_i.due:
                time_penalty_value += max(0, arrival_time - node_i.due)
                arrival_time = node_i.due  # assume arriving at due time

            # handle customer
            if node_i.type == "c":  
                current_time = arrival_time + node_i.service
                if i < len(route.nodes) - 1:
                    next_node = route.nodes[i + 1]
                    battery_use += self.dist_matrix[self.node_id[node_i.id], self.node_id[next_node.id]]

            # handle station
            elif node_i.type == "f":
                if battery_use > 0:
                    charge_time = battery_use / charge_rate
                    current_time += charge_time
                    battery_use = 0

            # travel time to next
            if i < len(route.nodes) - 1:
                current_time += self.time_cost(node_i, route.nodes[i + 1])

        return time_penalty_value

    def update_penalty_weights(self, solution, step):
        """Update α, β, γ adaptively following the paper's rule."""
        delta = 1.2  # growth factor used in experiments
        penalty_update_interval = 2  # τ_penalty = 2

        # compute current violations
        load_violation = sum(self.load_penalty(route) for route in solution)
        tw_violation = sum(self.time_penalty(route) for route in solution)
        batt_violation = sum(self.battery_penalty(route) for route in solution)

        # flags for whether there is a violation
        self.load_update = load_violation > 0
        self.tw_update = tw_violation > 0
        self.batt_update = batt_violation > 0

        # update only at scheduled steps
        if step % penalty_update_interval == 0:
            # α: capacity violation penalty
            if self.load_update:
                self.alpha = min(self.alpha * delta, self.alpha_max)
            else:
                self.alpha = max(self.alpha / delta, self.alpha_min)
            
            # β: time-window violation penalty
            if self.tw_update:
                self.beta = min(self.beta * delta, self.beta_max)
            else:
                self.beta = max(self.beta / delta, self.beta_min)

            # γ: battery violation penalty
            if self.batt_update:
                self.gamma = min(self.gamma * delta, self.gamma_max)
            else:
                self.gamma = max(self.gamma / delta, self.gamma_min)

            self.update_reset()

    def update_reset(self):
        """Reset violation flags."""
        self.load_update = False
        self.batt_update = False
        self.tw_update = False

    def initial_solution(self):
        """Initial solution based on Schneider et al. (2014): polar-angle sort + greedy insertion + batch TW ordering."""
        depot = self.instance.depot
        customers = self.instance.customers

        # choose a random point for polar-angle sorting
        random_point = random.choice(customers)
        customers_sorted = sorted(customers, key=lambda c: self.polar_angle(c, depot, random_point))

        # number of predefined routes (from best-known solutions)
        predefined_routes = self.predefine_route_number

        routes = []
        current_route = self.create_new_route()
        current_route.nodes.append(depot)

        last_route = self.create_new_route()
        unassigned_customers = []  # customers that cannot fit within predefined_routes

        for customer in customers_sorted:
            best_position = None
            min_extra_cost = float('inf')

            # find best insertion position in current route (min marginal cost)
            for i in range(1, len(current_route.nodes)):  # do not insert at the start
                temp_route = self.copy_route(current_route)
                temp_route.nodes.insert(i, customer)

                if not self.load_violation(temp_route, customer) and not self.time_violation(temp_route, customer):
                    extra_cost = self.generalized_cost([temp_route], penalty_value=False, p_div_value=False, allow_infeasible=True)
                    if extra_cost < min_extra_cost:
                        min_extra_cost = extra_cost
                        best_position = i

            # insert or open a new route
            if best_position is not None:
                current_route.nodes.insert(best_position, customer)
            else:
                if len(routes) < predefined_routes:
                    # close current route and open a new one
                    routes.append(current_route)

                    current_route = self.create_new_route()
                    current_route.nodes.append(customer)
                    current_route.nodes.append(depot)  # start of the new route
                else:
                    # exceed route-limit; postpone to last_route
                    unassigned_customers.append(customer)

        if len(current_route.nodes) > 2:
            routes.append(current_route) 

        # batch-insert to last_route ordered by ready time
        if len(unassigned_customers) > 0:
            unassigned_customers.sort(key=lambda c: c.ready)
            last_route.nodes.extend(unassigned_customers)
            last_route.nodes.append(depot)
            routes.append(last_route)

        while len(routes) < predefined_routes:
            current_route = self.create_new_route()
            current_route.nodes.append(depot)
            routes.append(current_route)

        return routes

    def vns_perturb(self, solution, k):
        """Neighborhood perturbation for VNS (following Table 2 in the paper)."""
        neighborhood_structure = {
            1: (2, 1),  2: (2, 2),  3: (2, 3),  4: (2, 4),  5: (2, 5),
            6: (3, 1),  7: (3, 2),  8: (3, 3),  9: (3, 4), 10: (3, 5),
            11: (4, 1), 12: (4, 2), 13: (4, 3), 14: (4, 4), 15: (4, 5)
        }

        if k not in neighborhood_structure:
            return solution
        if len(solution) == 1:
            if random.random() < 0.3:
                return self.extra_exchange(solution)
            return solution

        if len(solution) < neighborhood_structure[k][0]:
            return solution

        num_routes, max_nodes = neighborhood_structure[k]
        return self.cyclic_exchange(solution, num_routes, max_nodes)

    def cyclic_exchange(self, solution, num_routes, max_nodes):
        """Cyclic-Exchange across num_routes routes (paper's κ-neighborhood)."""
        if len(solution) < num_routes:
            return solution

        # Optimization 1: copy structure only where modified
        selected_routes_idx = random.sample(range(len(solution)), num_routes)
        new_solution = [solution[i] for i in range(len(solution))]

        selected_routes = [solution[i] for i in selected_routes_idx]
        segments, start_positions, end_positions = [], [], []

        # Optimization 2: avoid repeated attribute access
        for route in selected_routes:
            nodes = route.nodes
            num_nodes = len(nodes)
            if num_nodes < 3:  # need at least depot + one customer + depot
                return solution  

            start = random.randint(1, num_nodes - 2)
            max_chain_length = min(max_nodes, num_nodes - 2)  
            chain_length = random.randint(0, max_chain_length)
            end = min(start + chain_length, num_nodes - 1)

            segments.append(nodes[start:end])
            start_positions.append(start)
            end_positions.append(end)

        # Optimization 3: in-place splice
        for i in range(num_routes):
            next_i = (i + 1) % num_routes
            route = new_solution[selected_routes_idx[next_i]]
            route.nodes[start_positions[next_i]:end_positions[next_i]] = segments[i]

        return new_solution

    def extra_exchange(self, solution):
        """Extract a customer from the first route and open a new route for diversification."""
        node_idx = random.randint(1, len(solution[0].nodes) - 1)
        while solution[0].nodes[node_idx].type != "c":
            node_idx = random.randint(1, len(solution[0].nodes) - 1)
        node = solution[0].nodes.pop(node_idx)
        route_add = self.create_new_route()
        route_add.nodes.append(node)
        route_add.nodes.append(self.instance.depot)
        solution.append(route_add)
        return solution

    def _tabu_search(self, S):
        """Tabu search."""
        best_solution = copy.deepcopy(S)
        current_solution = copy.deepcopy(S)
        tabu_list = deque(maxlen=self.tabu_tenure)
        for iter in range(self.tabu_iter):
            # generate candidate neighborhood
            self.route_info = [self.print_route(r) for r in current_solution]
            two_opt_start, two_opt_route_info = self._two_opt(current_solution)
            relocate_start, relocate_route_info = self._relocate(current_solution)
            exchange_start, exchange_route_info = self._exchange(current_solution)
            station_in_re_start, station_in_re_route_info = self._station_insertion(current_solution)

            zip_neighborhood = [two_opt_start, relocate_start, exchange_start, station_in_re_start]
            neighborhood = [item for sublist in zip_neighborhood for item in sublist]
            zip_infos = [two_opt_route_info, relocate_route_info, exchange_route_info, station_in_re_route_info]
            tabu_infos = [item for sublist in zip_infos for item in sublist]

            # pick the best candidate
            current_candidate = min(neighborhood, key=self.generalized_cost)
            current_candidate_info = tabu_infos[neighborhood.index(current_candidate)]
            current_candidate = self.solution_fix(current_candidate)
            best_solution = min(
                neighborhood, 
                key=lambda sol: self.generalized_cost(
                    sol, penalty_value=False, p_div_value=False, allow_infeasible=False
                )
            )
            best_solution_value = self.generalized_cost(best_solution, penalty_value=False, p_div_value=False, allow_infeasible=False)
            best_solution = self.solution_fix(best_solution)

            if best_solution_value < self.global_value:
                global_value = best_solution_value  # NOTE: this sets a local variable in original code
                self.golbal_solution = best_solution  # NOTE: original typo preserved

            if current_candidate_info not in tabu_list:
                depot_to_depot = [r for r in current_candidate if len(r.nodes) == 2]
                if len(depot_to_depot) > 1:
                    regular_routes = [r for r in current_candidate if len(r.nodes) > 2]
                    current_candidate = regular_routes.extend(depot_to_depot[0])
                
                current_solution = current_candidate
                tabu_list.append(current_candidate_info)

                # update historical route-structure frequency
                self.update_diversification_history(current_solution)

                # update best solution
                if self.generalized_cost(current_solution, penalty_value=False, p_div_value=False, allow_infeasible=False) < self.generalized_cost(best_solution, penalty_value=False, p_div_value=False, allow_infeasible=False):
                    best_solution = copy.deepcopy(current_solution)
        test_end = time.time()
        return best_solution

    def update_diversification_history(self, S):
        """Update historical frequency of route structures."""
        for k, route in enumerate(S):
            for i in range(1, len(route.nodes) - 1):  # iterate over customers
                u = route.nodes[i].id
                mu = route.nodes[i - 1].id
                zeta = route.nodes[i + 1].id
                self.attribute_frequency[(u, k, mu, zeta)] += 1
                self.attribute_total += 1

    def generalized_cost(self, S, penalty_value=True, p_div_value=True, allow_infeasible=True, tabu_search=False):
        """Unified objective function: distance + penalties (and optional diversification penalty)."""

        if not allow_infeasible and not self.is_solution_feasible(S):
            infeasible_cost = 1e10
            return (infeasible_cost, infeasible_cost) if tabu_search else infeasible_cost

        total_distance = sum(
            self.dist_matrix[self.node_id[route.nodes[i].id]][self.node_id[route.nodes[i + 1].id]]
            for route in S for i in range(len(route.nodes) - 1)
        )

        total_penalty = sum(
            self.alpha * self.load_violation(route) +
            self.beta * self.time_penalty(route) +
            self.gamma * self.battery_penalty(route)
            for route in S
        ) if penalty_value else 0

        p_div_penalty = 0
        if p_div_value:
            num_customers = sum(len(route.nodes) - 2 for route in S)
            num_vehicles = len(S)

            penalty_sum = sum(
                self.attribute_frequency.get((route.nodes[i].id, k, route.nodes[i - 1].id, route.nodes[i + 1].id), 0)
                for k, route in enumerate(S)
                for i in range(1, len(route.nodes) - 1)
            )

            p_div_penalty = (self.lambda_div * total_distance * penalty_sum *
                            ((num_customers * num_vehicles) ** 0.5) / (1e-10 + self.attribute_total))

        total_cost = total_distance + total_penalty + p_div_penalty

        if tabu_search:
            # for tabu search return both cost and distance
            return total_cost, total_distance
        else:
            return total_cost

    def print_route(self, route):
        """Return a string identifier for a route."""
        route_info = []
        for node in route.nodes:
            route_info.append(node.id)
        return '->'.join(route_info)

    def _two_opt(self, solution):
        """2-opt* move between two routes."""
        two_opt_solution = []
        two_opt_tabu_list = []
        
        for i in range(len(solution) - 1):
            for j in range(i + 1, len(solution)):
                for split_1 in range(1, len(solution[i].nodes)-1):
                    for split_2 in range(1, len(solution[j].nodes)-1):
                        # shallow copy solution
                        new_solution = solution[:]

                        # deep copy only modified routes
                        new_solution[i] = copy.deepcopy(solution[i])
                        new_solution[j] = copy.deepcopy(solution[j])

                        # swap tails
                        segment1_head, segment1_tail = new_solution[i].nodes[:split_1], new_solution[i].nodes[split_1:]
                        segment2_head, segment2_tail = new_solution[j].nodes[:split_2], new_solution[j].nodes[split_2:]

                        new_solution[i].nodes = segment1_head + segment2_tail
                        new_solution[j].nodes = segment2_head + segment1_tail

                        # record tabu info
                        route_info = (['Two_opt', self.route_info[i] + str(split_1), self.route_info[j] + str(split_2)])

                        two_opt_solution.append(new_solution)
                        two_opt_tabu_list.append(route_info)
        
        return two_opt_solution, two_opt_tabu_list

    def _relocate(self, solution):
        """Relocate a customer from one route to another."""
        relocate_solution = []
        relocate_tabu_list = []

        for i in range(len(solution)):
            for j in range(len(solution)):
                for split_pos in range(1, len(solution[i].nodes)-1):
                    for insert_pos in range(1, len(solution[j].nodes)):
                        # shallow copy
                        new_solution = solution[:]
                        # deep copy only modified routes
                        new_solution[i] = copy.deepcopy(solution[i])
                        new_solution[j] = copy.deepcopy(solution[j])

                        route_info = (['Relocate', self.route_info[i] + str(split_pos), self.route_info[j] + str(insert_pos)])

                        # execute relocate
                        if i == j and insert_pos > split_pos:
                            insert_pos -= 1  # adjust target index

                        node = new_solution[i].nodes.pop(split_pos)
                        new_solution[j].nodes.insert(insert_pos, node)

                        relocate_solution.append(new_solution)
                        relocate_tabu_list.append(route_info)
                        my_debug = False
                        if my_debug:
                            self.generalized_cost(new_solution, False, False, False)

                    # relocate to a new route (open one)
                    new_solution = copy.deepcopy(solution)
                    new_solution[i] = copy.deepcopy(solution[i])
                    node = new_solution[i].nodes.pop(split_pos)

                    new_route = self.create_new_route()
                    new_route.nodes.append(node)
                    new_route.nodes.append(self.instance.depot)

                    new_solution.append(new_route)
                    route_info = (['Relocate', self.route_info[i] + str(split_pos)])
                    relocate_solution.append(new_solution)
                    relocate_tabu_list.append(route_info)

        return relocate_solution, relocate_tabu_list

    def _exchange(self, solution):
        """Exchange two customers across two routes."""
        exchange_solution = []
        exchange_tabu_list = []

        for i in range(len(solution)):
            for j in range(len(solution)):
                for split_pos1 in range(1, len(solution[i].nodes)-1):
                    if solution[i].nodes[split_pos1].type != 'c':
                        continue
                    for split_pos2 in range(1, len(solution[j].nodes)-1):
                        if solution[j].nodes[split_pos2].type != 'c' or (i == j and split_pos1 == split_pos2):
                            continue

                        # shallow copy
                        new_solution = solution[:]
                        # deep copy only modified routes
                        new_solution[i] = copy.deepcopy(solution[i])
                        new_solution[j] = copy.deepcopy(solution[j])

                        # swap
                        new_solution[i].nodes[split_pos1], new_solution[j].nodes[split_pos2] = (
                            new_solution[j].nodes[split_pos2],
                            new_solution[i].nodes[split_pos1],
                        )

                        route_info = (['Exchange', self.route_info[i] + str(split_pos1), self.route_info[j] + str(split_pos2)])

                        exchange_solution.append(new_solution)
                        exchange_tabu_list.append(route_info)

        return exchange_solution, exchange_tabu_list

    def _station_insertion(self, solution):
        """StationReIn: insert/remove charging stations with a local tabu mechanism."""

        # initialize local tabu list if needed
        if not hasattr(self, 'StationReIn_tabu_list'):
            self.StationReIn_tabu_list = {}

        station_in_re_solution = []
        station_in_re_tabu_list = []

        for i in range(len(solution)):
            for insert_pos in range(1, len(solution[i].nodes)):
                node = solution[i].nodes[insert_pos]

                # remove a station
                if node.type == 'f':
                    μ, ζ = solution[i].nodes[insert_pos-1], solution[i].nodes[insert_pos+1]
                    arc = (μ.id, ζ.id)  # record deleted arc

                    # shallow copy
                    new_solution = solution[:]
                    # deep copy only this route
                    new_solution[i] = copy.deepcopy(solution[i])

                    # remove the station
                    new_solution[i].nodes = new_solution[i].nodes[:insert_pos] + new_solution[i].nodes[insert_pos+1:]

                    route_info = ['StationInReRemove', self.route_info[i] + "|" + str(insert_pos)]

                    # update local tabu list
                    tabu_tenure = random.randint(15, 30)
                    self.StationReIn_tabu_list[arc] = tabu_tenure

                    station_in_re_solution.append(new_solution)
                    station_in_re_tabu_list.append(route_info)

                # insert a station
                else:
                    for station in self.instance.stations:
                        if solution[i].nodes[insert_pos-1].id == station.id:
                            continue  # avoid immediate repetition
                        
                        μ, ζ = solution[i].nodes[insert_pos-1], station
                        arc = (μ.id, ζ.id)

                        # check tabu
                        if arc in self.StationReIn_tabu_list and self.StationReIn_tabu_list[arc] > 0:
                            continue

                        # shallow copy
                        new_solution = solution[:]
                        # deep copy only this route
                        new_solution[i] = copy.deepcopy(solution[i])

                        # insert the station
                        new_solution[i].nodes.insert(insert_pos, station)

                        route_info = ['StationInReInsert', self.route_info[i] + "|" + str(insert_pos)]

                        station_in_re_solution.append(new_solution)
                        station_in_re_tabu_list.append(route_info)

        # decrease tabu tenure and clean expired entries
        for arc in list(self.StationReIn_tabu_list.keys()):
            if self.StationReIn_tabu_list[arc] > 0:
                self.StationReIn_tabu_list[arc] -= 1
            if self.StationReIn_tabu_list[arc] == 0:
                del self.StationReIn_tabu_list[arc]

        return station_in_re_solution, station_in_re_tabu_list

    def adjacent_check(self, solutions, name=None, checkpoint_mode=True):
        """Sanity check for adjacent duplicate nodes."""
        for solution in solutions:
            for route in solution:
                for i in range(1, len(route.nodes) - 1):
                    if route.nodes[i] == route.nodes[i-1]:
                        print("{} Adjacent Check Failed".format(name))
                        if checkpoint_mode:
                            breakpoint()
                        else:
                            return False
        return True

    def is_solution_feasible(self, solution):
        """Check the feasibility of the whole solution."""
        served_customers = set()
        for route in solution:
            if not self.is_route_feasible(route):
                return False

            # ensure each customer is served exactly once
            for node in route.nodes:
                if node.type == 'c':
                    if node.id in served_customers:
                        return False
                    served_customers.add(node.id)
        
        # ensure all customers are served
        all_customers = {customer.id for customer in self.instance.customers}
        if served_customers != all_customers:
            return False

        return True

    def is_route_feasible(self, route, new_node=None):
        """Check feasibility of a single route."""
        return not (self.load_violation(route) or self.time_violation(route) or self.battery_violation(route))

    def create_new_route(self):
        """Create a new empty route starting at depot."""
        return Route([self.instance.depot])

    def solution_fix(self, solution):
        """Light cleanup of a solution: remove immediate duplicates and empty routes."""
        S = []
        for route in solution:
            # 1) remove immediate duplicates
            route.nodes = [route.nodes[i] for i in range(len(route.nodes)) if i == 0 or route.nodes[i].id != route.nodes[i - 1].id]

            # 2) keep routes that contain at least one customer
            if any(node.type == "c" for node in route.nodes):
                S.append(route)

        return S

    def polar_angle(self, customer, depot, random_point):
        """Compute the polar angle of a customer relative to the depot and a random reference point."""
        # Vector 1: depot -> random point
        dx1, dy1 = random_point.x - depot.x, random_point.y - depot.y
        # Vector 2: depot -> customer
        dx2, dy2 = customer.x - depot.x, customer.y - depot.y

        angle1 = math.atan2(dy1, dx1)
        angle2 = math.atan2(dy2, dx2)

        relative_angle = (angle2 - angle1) % (2 * math.pi) 
        return relative_angle

    def copy_solution(self, solution):
        """Deep copy a solution (list of routes)."""
        return [self.copy_route(r) for r in solution]

    def copy_route(self, route):
        """Deep copy a single route."""
        new_route = Route()
        new_route.nodes = copy.deepcopy(route.nodes)
        new_route.load = route.load
        new_route.time = route.time
        new_route.fuel = route.fuel
        return new_route

    def print_solution(self, solution):
        """Pretty-print the solution as route sequences."""
        res = []
        for routes in solution:
            route = []
            for node in routes.nodes:
                route.append(node.id)
            res.append(' -> '.join(route))
        res.sort()
        print(' | '.join(res))
