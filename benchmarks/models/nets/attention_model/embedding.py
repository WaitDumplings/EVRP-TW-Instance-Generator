import torch
import torch.nn as nn

def AutoEmbedding(problem_name, config):
    """
    Automatically select the corresponding module according to ``problem_name``
    """
    mapping = {
        "evrptw": EVRPTWEmbedding,
        "evrptw_graph": EVRPTWGraphEmbedding,
    }
    embeddingClass = mapping[problem_name]
    embedding = embeddingClass(**config)
    return embedding

# Embedding Layer
class EVRPTWEmbedding(nn.Module):
    """
    Embedding for the capacitated vehicle routing problem.
    The shape of tensors in ``input`` is summarized as following:

    +-------------+-----------------------------+
    | Key         | Size of Tensor             |
    +=============+=============================+
    | 'cus_loc'   | [batch, n_customer, 2]     |
    +-------------+-----------------------------+
    | 'depot_loc' | [batch, 2]                 |
    +-------------+-----------------------------+
    | 'rs_loc'    | [batch, n_rs, 2]           |
    +-------------+-----------------------------+
    | 'demand'    | [batch, n_customer, 1]     |
    +-------------+-----------------------------+
    | 'time_window' | [batch, n_customer, 1]   |
    +-------------+-----------------------------+

    Args:
        embedding_dim: dimension of output
    Inputs: input
        * **input** : dict of ['cus_loc', 'depot_loc', 'demand', 'time_window']
    Outputs: out
        * **out** : [batch, n_customer+n_rs+1, embedding_dim]
    """
        
    def __init__(self, embedding_dim, extra_dim = 3):
        super().__init__()
        self.depot_embedding = nn.Linear(2, embedding_dim)
        self.nodes_embedding = nn.Linear(2 + extra_dim, embedding_dim)
        self.rs_embedding = nn.Linear(2, embedding_dim)
        self.context_dim = embedding_dim + 3  # Embedding of last node + remaining_capacity + remaining_battery + current_time
        self.embedmodel = EVRPTWGraphEmbedding(5, 128, 128)

    def forward(self, x):
        # demand = depot demand (0) + rs_demand (0) + cus_demand 
        # cus_demand
        # res = self.embedmodel(x)
        cus_demand = x['demand'][:, 1 + x['rs_loc'].size()[1]:].unsqueeze(-1)
        cus_time_window = x['time_window'][:, 1 + x['rs_loc'].size()[1]:]
        cus_nodes = self.nodes_embedding(torch.cat((
            x['cus_loc'], cus_demand, cus_time_window
        ), dim=-1))
        rs_nodes = self.rs_embedding(x['rs_loc'])
        depot_node = self.depot_embedding(x['depot_loc'].unsqueeze(1))

        return torch.cat((depot_node, rs_nodes, cus_nodes), dim=-2)

class EVRPTWGraphEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the EVRPTWGraphEmbedding model.
        
        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden layer features.
            output_dim (int): Dimension of output node features.
        """
        from torch_geometric.nn import GCNConv
        super(EVRPTWGraphEmbedding, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # First GCN layer
        self.conv2 = GCNConv(hidden_dim, output_dim)  # Second GCN layer
        self.relu = nn.ReLU(True)  # ReLU activation
        # Learnable type-specific features (0: depot, 1: recharging station, 2: customer)
        self.nodes_type_features = nn.Parameter(torch.randn(3, output_dim))
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (dict): Dictionary containing node feature tensors for depot, recharging stations, and customers.
        
        Returns:
            torch.Tensor: Node embeddings with added type-specific features.
        """
        # Fuse node features for depot, recharging stations, and customers
        x, rs_nodes, cus_nodes = self._fusion(x)

        # Get batch size, number of nodes, and feature dimension
        batch_size, num_nodes, num_features = x.shape

        # Flatten node features and create a fully connected edge index
        x = x.view(-1, num_features)
        edge_index = self._create_fully_connected_edge_index(num_nodes, batch_size)

        # Apply the first GCN layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        # Apply the second GCN layer
        x = self.conv2(x, edge_index)

        # Reshape back to batch format
        x = x.view(batch_size, num_nodes, -1)

        # Add type-specific features to the corresponding nodes
        x[:, 0, :] += self.nodes_type_features[0]  # Depot node
        x[:, 1:rs_nodes+1, :] += self.nodes_type_features[1]  # Recharging station nodes
        x[:, rs_nodes+1:, :] += self.nodes_type_features[2]  # Customer nodes

        # Ensure output shape matches expected dimensions
        assert x.shape == (batch_size, num_nodes, self.conv2.out_channels), \
            f"Output shape mismatch: expected ({batch_size}, {num_nodes}, {self.conv2.out_channels}), got {x.shape}"

        return x
    
    # def _create_fully_connected_edge_index(num_nodes, batch_size):
    #     """
    #     Create a fully connected edge index for batched graphs.
        
    #     Args:
    #         num_nodes (int): Number of nodes in a single graph.
    #         batch_size (int): Number of graphs in the batch.
        
    #     Returns:
    #         torch.Tensor: Fully connected edge index for the batched graphs.
    #     """
    #     # Create edge index for a single graph (all pairs of nodes)
    #     row = torch.arange(num_nodes).repeat_interleave(num_nodes)
    #     col = torch.arange(num_nodes).repeat(num_nodes)
    #     edge_index = torch.stack([row, col], dim=0)  # Shape: [2, num_nodes * num_nodes]
        
    #     # Number of edges in a single graph
    #     num_edges = num_nodes * num_nodes
        
    #     # Repeat edge index for each graph in the batch
    #     batch_offsets = torch.arange(batch_size).repeat_interleave(num_edges) * num_nodes
    #     batch_offsets = batch_offsets.to(edge_index.device)  # Ensure same device as edge_index

    #     # Expand batch_offsets to match edge_index shape
    #     expanded_edge_index = edge_index.repeat(1, batch_size)  # Repeat edge_index for batch
    #     expanded_edge_index[0] += batch_offsets  # Adjust row indices for batch
    #     expanded_edge_index[1] += batch_offsets  # Adjust col indices for batch
        
    #     return expanded_edge_index


    def _fusion(self, x):
        """
        Fuse node features for depot, recharging stations, and customers.
        
        Args:
            x (dict): Dictionary containing the following keys:
                - 'depot_loc': Location of the depot.
                - 'rs_loc': Locations of recharging stations.
                - 'cus_loc': Locations of customers.
                - 'demand': Customer demand values.
                - 'time_window': Time window constraints for customers.
        
        Returns:
            tuple: Fused node features, number of recharging station nodes, number of customer nodes.
        """
        # Customer nodes: location, demand, and time window
        cus_info = torch.cat((
            x['cus_loc'],
            x['demand'][:, 1 + x['rs_loc'].size(1):].unsqueeze(-1),
            x['time_window'][:, 1 + x['rs_loc'].size(1):]
        ), dim=-1)

        # Recharging station nodes: location, dummy demand, dummy time window, type indicator
        device = cus_info.device
        rs_info = torch.cat((
            x['rs_loc'],
            torch.zeros(x['rs_loc'].shape[0], x['rs_loc'].shape[1], 1).to(device),
            torch.zeros(x['rs_loc'].shape[0], x['rs_loc'].shape[1], 1).to(device),
            torch.ones(x['rs_loc'].shape[0], x['rs_loc'].shape[1], 1).to(device),
        ), dim=-1)

        # Depot node: location, dummy demand, dummy time window, type indicator
        depot_info = torch.cat((
            x['depot_loc'].unsqueeze(1),
            torch.zeros(x['depot_loc'].shape[0], 1, 1).to(device),
            torch.zeros(x['depot_loc'].shape[0], 1, 1).to(device),
            torch.ones(x['depot_loc'].shape[0], 1, 1).to(device),
        ), dim=-1)

        # Concatenate all nodes
        return torch.cat((depot_info, rs_info, cus_info), dim=-2), x['rs_loc'].shape[1], x['cus_loc'].shape[1]
