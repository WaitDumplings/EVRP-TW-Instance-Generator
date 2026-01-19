import numpy as np
# Utility function to find and print all routes from start nodes to an end node
def find_and_print_routes(sequence, end_node_idx=5):
    routes = []  # Store all complete routes
    visited = set()  # Track visited nodes

    # Recursive function to find paths
    def find_route(current_node, current_route):
        i, j = map(int, current_node.split('_')[1:])  # Extract node indices i and j
        current_route.append(current_node)  # Add the current node to the current route

        # If we reach the end node x_j_0, the route is complete
        if j == end_node_idx:
            routes.append(current_route[:])  # Save the complete route
            return

        # Look for the next connected node that continues the path
        for next_node in sequence:
            if next_node not in visited:
                next_i, next_j = map(int, next_node.split('_')[1:])
                # Ensure the path continuity: x_i_j -> x_j_m
                if next_i == j:
                    visited.add(next_node)  # Mark node as visited
                    find_route(next_node, current_route)  # Recur for the next node
                    visited.remove(next_node)  # Backtrack to explore other possibilities
        current_route.pop()  # Backtrack to the previous node

    # Iterate over the sequence and find all routes starting with x_0_k
    for start_node in sequence:
        if start_node.startswith('x_0_') and start_node not in visited:
            visited.add(start_node)
            find_route(start_node, [])  # Start finding routes from each valid start node
            visited.remove(start_node)  # Remove after route is completed

    # Print all complete routes
    print(f"Total number of complete routes: {len(routes)}")
    for idx, route in enumerate(routes):
        print(f"Route {idx + 1}: {' -> '.join(route)}")


# Function to calculate Euclidean distance between two nodes
def euclidean_distance(pos1, pos2):
    """
    Calculate the Euclidean distance between two nodes.
    
    Args:
        pos1 (list): Position [x, y] of the first node.
        pos2 (list): Position [x, y] of the second node.
    
    Returns:
        float: Euclidean distance between the two nodes.
    """
    return np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
