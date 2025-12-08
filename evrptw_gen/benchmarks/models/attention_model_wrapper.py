import torch
import torch.nn as nn
from .nets.attention_model.attention_model import *

class Problem:
    def __init__(self, name):
        self.NAME = name

class Backbone(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        problem_name="evrptw",
        n_encode_layers=3,
        tanh_clipping=15.0,
        n_heads=16,
        device="cpu",
        use_graph_token=False,
    ):
        super().__init__()
        self.device = device
        self.problem = Problem(problem_name)

        self.embedding = AutoEmbedding(self.problem.NAME, {"embedding_dim": embedding_dim})
        self.use_graph_token = use_graph_token
        self.graph_token = nn.Parameter(torch.empty(1, 1, embedding_dim))
        nn.init.xavier_uniform_(self.graph_token)

        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_encode_layers,
        )

        self.decoder = Decoder(embedding_dim = embedding_dim,
                               step_context_dim = embedding_dim + 5, # since we need to concat feature with (battery, loading, time). 
                               n_heads = n_heads,
                               problem = self.problem,
                               tanh_clipping = tanh_clipping,
                               use_graph_token = use_graph_token)

    def _apply_graph_token(self, embedding, mask):
        """在 [B,N,D] 的 embedding 前面拼一个 [GRAPH] token，并同步更新 mask。"""
        if not self.use_graph_token:
            return embedding, mask

        B = embedding.size(0)
        device = embedding.device

        graph_tok = self.graph_token.expand(B, 1, -1)  # [B,1,D]
        embedding = torch.cat([graph_tok, embedding], dim=1)  # [B,N+1,D]

        if mask is not None:
            pad = torch.zeros(B, 1, dtype=mask.dtype, device=device)
            mask = torch.cat([pad, mask], dim=1)  # [B,N+1]

        return embedding, mask

    def forward(self, obs, use_mask=False):
        mask = obs["instance_mask"] if use_mask else None

        state = stateWrapper(obs, device=self.device, problem=self.problem.NAME)
        inputs = state.states["observations"]         # Input
        embedding = self.embedding(inputs)            # [B,N,D]

        embedding, mask = self._apply_graph_token(embedding, mask)

        encoded_inputs = self.encoder(embedding, mask=mask)

        # decoding
        cached_embeddings = self.decoder._precompute(encoded_inputs, mask=mask)
        logits, glimpse = self.decoder.advance(cached_embeddings, state, node_mask=mask)

        return logits, glimpse

    def encode(self, obs, use_mask=False):
        mask = obs["instance_mask"] if use_mask else None
        
        state = stateWrapper(obs, device=self.device, problem=self.problem.NAME)
        inputs = state.states["observations"]

        embedding = self.embedding(inputs)

        embedding, mask = self._apply_graph_token(embedding, mask)

        encoded_inputs = self.encoder(embedding, mask=mask)

        cached_embeddings = self.decoder._precompute(encoded_inputs, mask=mask)
        return cached_embeddings

    def decode(self, obs, cached_embeddings):
        state = stateWrapper(obs, device=self.device, problem=self.problem.NAME)
        logits, glimpse = self.decoder.advance(cached_embeddings, state)
        return logits, glimpse

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
    
    def forward(self, x):
        logits = x[0]
        return logits

def orthogonal_init(layer, gain=1.0):
    """ Orthogonal 初始化 + 零偏置 """
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

class Critic(nn.Module):
    def __init__(self, hidden_size):
        super(Critic, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # 第一层
            nn.LayerNorm(hidden_size),  # ✅ 归一化防止梯度爆炸
            nn.SiLU(),

            nn.Linear(hidden_size, hidden_size // 2),  # 第二层
            nn.SiLU(),

            nn.Linear(hidden_size // 2, 1)  # 输出 V(s)
        )

        # ✅ 应用 Orthogonal 初始化
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer, gain=0.01)  # 低 gain 让 V(s) 训练更平滑

    def forward(self, x):
        return self.mlp(x[1])  

class Agent(nn.Module):
    def __init__(self, embedding_dim=256, tanh_clipping =15, n_encode_layers = 3, device="cpu", name="evrptw", use_graph_token = False):
        super(Agent, self).__init__()
        self.backbone = Backbone(embedding_dim = embedding_dim,
                                 device = device,
                                 tanh_clipping = tanh_clipping,
                                 n_encode_layers = n_encode_layers,
                                 problem_name = name,
                                 use_graph_token = use_graph_token)
        self.critic = Critic(hidden_size = embedding_dim)
        self.actor = Actor()

    def forward(self, x, use_mask=False):
        x = self.backbone(x, use_mask)
        logits = self.actor(x)
        action = logits.max(2)[1]
        return action, logits

    def get_value(self, x):
        x = self.backbone(x)
        return self.critic(x)
    
    def get_acction_and_value(self, x, action=None):
        x = self.backbone(x)
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_value_cached(self, x, state):
        x = self.backbone.decode(x, state)
        return self.critic(x)
    
    def get_action_and_value_cached(self, x, action=None, state=None):
        # breakpoint()
        if state is None:
            state = self.backbone.encode(x)
            x = self.backbone.decode(x, state)
        else:
            x = self.backbone.decode(x, state)
        logits = self.actor(x)

        # logits_inf = (logits == -torch.inf)
        # for i in range(logits_inf.shape[0]):
        #     for j in range(logits_inf.shape[1]):
        #         if logits_inf[i, j, :].all():
        #             breakpoint()
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        # value_state = self.critic(x)
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), state

class stateWrapper:
    def __init__(self, states, device, problem="evrptw"):
        self.device = device
        self.states = {k: torch.tensor(v, device=self.device) for k, v in states.items()}
        if problem == "evrptw":
            input = {
                "depot_loc": self.states["depot_loc"],
                "cus_loc": self.states["cus_loc"],
                "rs_loc": self.states["rs_loc"],
                "time_window": self.states["time_window"],
                "demand": self.states["demand"].unsqueeze(-1),
                "service_time": self.states["service_time"].unsqueeze(-1)
            }

            self.states["observations"] = input
            self.VEHICLE_CAPACITY = self.states['loading_capacity']
            self.VEHICLE_BATTERY = self.states['battery_capacity']
            self.used_capacity = self.states["current_load"]
            self.used_battery  = self.states["current_battery"]
            self.current_time = self.states["current_time"]
            self.visited_customers_raio = self.states["visited_customers_raio"]
            self.remain_feasible_customers_raio = self.states["remain_feasible_customers_raio"]


    def get_current_node(self):
        return self.states["last_node_idx"]

    def get_mask(self):
        return (1 - self.states["action_mask"]).to(torch.bool)