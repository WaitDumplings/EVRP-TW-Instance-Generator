import torch
import torch.nn as nn
from .nets.attention_model.attention_model import *

class Problem:
    def __init__(self, name):
        self.NAME = name

class Backbone(nn.Module):
    def __init__(self,
                 embedding_dim=128,
                 problem_name="evrptw",
                 n_encode_layers=3,
                 tanh_clipping=15.0,
                 n_heads=16,
                 device="cpu",
    ):
        super(Backbone, self).__init__()
        self.device = device
        self.problem = Problem(problem_name)
        self.embedding = AutoEmbedding(self.problem.NAME, {"embedding_dim": embedding_dim})

        self.encoder = GraphAttentionEncoder(
            n_heads = n_heads,
            embed_dim = embedding_dim,
            n_layers = n_encode_layers
        )

        self.decoder = Decoder(
            embedding_dim, self.embedding.context_dim, n_heads, self.problem, tanh_clipping
        )
    
    def forward(self, obs, use_mask=False):
        if not use_mask:
            mask = None
        else:
            mask = obs['instance_mask']
        state = stateWrapper(obs, device=self.device, problem=self.problem.NAME)
        input = state.states["observations"]
        embedding = self.embedding(input)
        encoded_inputs, _ = self.encoder(embedding, mask = mask)

        # decoding
        cached_embeddings = self.decoder._precompute(encoded_inputs, mask = mask)
        logits, glimpse = self.decoder.advance(cached_embeddings, state, node_mask=mask)

        return logits, glimpse

    def encode(self, obs, use_mask=False):
        if use_mask:
            mask = obs['instance_mask']
        else:
            mask = None
        state = stateWrapper(obs, device=self.device, problem=self.problem.NAME)
        input = state.states["observations"]
        embedding = self.embedding(input)
        encoded_inputs, _ = self.encoder(embedding, mask = mask)
        cached_embeedings = self.decoder._precompute(encoded_inputs)
        return cached_embeedings
    
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
        
# class Critic(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super(Critic, self).__init__()
#         hidden_size = kwargs["hidden_size"]
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.LeakyReLU(0.1), 
#             nn.Linear(hidden_size, 1)
#         )
    
#     def forward(self, x):
#         out = self.mlp(x[1])
#         return out

class Agent(nn.Module):
    def __init__(self, embedding_dim=256, tanh_clipping =15, n_encode_layers = 3, device="cpu", name="evrptw"):
        super(Agent, self).__init__()
        self.backbone = Backbone(embedding_dim = embedding_dim,
                                 device = device,
                                 tanh_clipping = tanh_clipping,
                                 n_encode_layers = n_encode_layers,
                                 problem_name = name)
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
        if state is None:
            state = self.backbone.encode(x)
            x = self.backbone.decode(x, state)
        else:
            x = self.backbone.decode(x, state)
        logits = self.actor(x)

        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
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
                "demand": self.states["demand"],
            }
            self.states["observations"] = input
            self.VEHICLE_CAPACITY = 0
            self.VEHICLE_BATTERY = 0
            self.used_capacity = -self.states["current_load"]
            self.used_battery  = -self.states["current_battery"]
            self.current_time = self.states["current_time"]


    def get_current_node(self):
        return self.states["last_node_idx"]

    def get_mask(self):
        return (1 - self.states["action_mask"]).to(torch.bool)