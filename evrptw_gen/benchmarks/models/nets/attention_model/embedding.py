import torch
import torch.nn as nn

def AutoEmbedding(problem_name, config):
    """
    Automatically select the corresponding module according to ``problem_name``
    """
    mapping = {
        "evrptw": EVRPTWEmbedding,
        # "evrptw_graph": EVRPTWGraphEmbedding,
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
        with torch.no_grad():
            self.type_embed.weight.zero_()
            # self.type_embed.weight.mul_(1e-3)

    def forward(self, x):
        """
        期望输入：
        x['depot_loc']     : [B, 2]
        x['rs_loc']        : [B, n_rs, 2]
        x['cus_loc']       : [B, n_cus, 2]

        x['demand']        : [B, 1+n_cus+n_rs, 1]
        x['time_window']   : [B, 1+n_cus+n_rs, 2]
        x['service_time']  : [B, 1+n_cus+n_rs, 1]

        其中 dim=1 的顺序为: [depot, customers, RS]
        """

        B = x['depot_loc'].size(0)
        device = x['depot_loc'].device

        n_cus = x['cus_loc'].size(1)
        n_rs  = x['rs_loc'].size(1)

        # ----- depot -----
        depot_loc = x['depot_loc']                         # [B,1,2]
        depot_demand  = x['demand'][:, :1, :]              # [B,1,1] = 0
        depot_tw      = x['time_window'][:, :1, :]         # [B,1,2] = [0,1]
        depot_service = x['service_time'][:, :1, :]        # [B,1,1] = 0

        depot_feat = torch.cat(
            [depot_loc, depot_demand, depot_tw, depot_service], dim=-1
        )  # [B,1,6]

        # ----- customers -----
        cus_loc     = x['cus_loc']                         # [B,n_cus,2]
        cus_demand  = x['demand'][:, 1:1+n_cus, :]         # [B,n_cus,1]
        cus_tw      = x['time_window'][:, 1:1+n_cus, :]    # [B,n_cus,2]
        cus_service = x['service_time'][:, 1:1+n_cus, :]   # [B,n_cus,1]

        cus_feat = torch.cat(
            [cus_loc, cus_demand, cus_tw, cus_service], dim=-1
        )  # [B,n_cus,6]

        # ----- RS -----
        rs_loc = x['rs_loc']                               # [B,n_rs,2]
        rs_demand  = x['demand'][:, 1+n_cus:, :]           # [B,n_rs,1] （可以全 0）
        rs_tw      = x['time_window'][:, 1+n_cus:, :]      # [B,n_rs,2] （一般 [0,1]）
        rs_service = x['service_time'][:, 1+n_cus:, :]     # [B,n_rs,1] （一般 0）

        rs_feat = torch.cat(
            [rs_loc, rs_demand, rs_tw, rs_service], dim=-1
        )  # [B,n_rs,6]

        # 拼成 [B, 1+n_cus+n_rs, 6]
        static_feat = torch.cat([depot_feat, cus_feat, rs_feat], dim=1)

        # 节点类型：0 depot, 1 RS, 2 customer
        depot_type = torch.zeros(B, 1, dtype=torch.long, device=device)
        cus_type   = torch.full((B, n_cus), 2, dtype=torch.long, device=device)
        rs_type    = torch.ones(B, n_rs, dtype=torch.long, device=device)

        node_type = torch.cat([depot_type, cus_type, rs_type], dim=1)  # [B,N]

        # 做 embedding
        h_static = self.static_proj(static_feat)           # [B,N,d]
        h_type   = self.type_embed(node_type)              # [B,N,d]

        node_emb = h_static + h_type                       # [B,N,d]

        return node_emb