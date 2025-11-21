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
class EVRPTWEmbedding_Legacy(nn.Module):
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

class EVRPTWEmbedding(nn.Module):
    """
    统一对 depot / RS / customer 做 embedding，
    静态特征：x, y, demand, tw_start, tw_end, service_time
    再加一个 node_type embedding 区分三种节点。
    """

    def __init__(self, embedding_dim = 128):
        super().__init__()
        self.embed_dim = embedding_dim

        # 2(x,y) + 1(demand) + 2(tw_start, tw_end) + 1(service_time) = 6
        self.static_proj = nn.Linear(6, embedding_dim)

        # 0: depot, 1: RS, 2: customer
        self.type_embed = nn.Embedding(3, embedding_dim)

    def forward(self, x):
        """
        期望输入：
        x['depot_loc']     : [B, 2]
        x['rs_loc']        : [B, n_rs, 2]
        x['cus_loc']       : [B, n_cus, 2]
        x['cus_demand']    : [B, n_cus, 1]   # 已经 / capacity
        x['cus_tw']        : [B, n_cus, 2]   # [start, end] in [0,1]
        x['cus_service']   : [B, n_cus, 1]   # normalized
        """

        B = x['depot_loc'].size(0)
        device = x['depot_loc'].device

        n_rs  = x['rs_loc'].size(1)
        n_cus = x['cus_loc'].size(1)

        # ----- depot -----
        depot_loc = x['depot_loc']               # [B,1,2]
        depot_demand = torch.zeros(B, 1, 1, device=device)      # 0
        depot_tw = torch.zeros(B, 1, 2, device=device)          # [0,1]
        depot_tw[..., 1] = 1.0
        depot_service = torch.zeros(B, 1, 1, device=device)     # 0
        depot_feat = torch.cat(
            [depot_loc, depot_demand, depot_tw, depot_service], dim=-1
        )  # [B,1,6]

        # ----- RS -----
        rs_loc = x['rs_loc']                                    # [B,n_rs,2]
        rs_demand = torch.zeros(B, n_rs, 1, device=device)
        rs_tw = torch.zeros(B, n_rs, 2, device=device)
        rs_tw[..., 1] = 1.0
        rs_service = torch.zeros(B, n_rs, 1, device=device)

        rs_feat = torch.cat(
            [rs_loc, rs_demand, rs_tw, rs_service], dim=-1
        )  # [B,n_rs,6]

        # ----- customers -----
        cus_loc = x['cus_loc']                                       # [B,n_cus,2]
        cus_demand = x['demand'][:, 1:1+n_cus,:]                     # [B,n_cus,1]
        cus_tw = x['time_window'][:, 1:1+n_cus,:]                    # [B,n_cus,2]
        cus_service = x['service_time'][:, 1:1+n_cus,:]              # [B,n_cus,1]

        cus_feat = torch.cat(
            [cus_loc, cus_demand, cus_tw, cus_service], dim=-1
        )  # [B,n_cus,6]

        # 拼成 [B, 1+n_rs+n_cus, 6]
        static_feat = torch.cat([depot_feat, rs_feat, cus_feat], dim=1)

        # 节点类型：0 depot, 1 RS, 2 customer
        depot_type = torch.zeros(B, 1, dtype=torch.long, device=device)
        rs_type = torch.ones(B, n_rs, dtype=torch.long, device=device)
        cus_type = torch.full((B, n_cus), 2, dtype=torch.long, device=device)
        node_type = torch.cat([depot_type, rs_type, cus_type], dim=1)  # [B,N]

        # 做 embedding
        h_static = self.static_proj(static_feat)                # [B,N,d]
        h_type = self.type_embed(node_type)                     # [B,N,d]

        node_emb = h_static + h_type                            # [B,N,d]

        return node_emb  # 直接交给后面的 encoder / GNN / transformer

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
