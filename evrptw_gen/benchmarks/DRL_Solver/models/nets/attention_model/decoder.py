   
import torch
import torch.nn as nn
import math

import torch
from torch import nn

from ...nets.attention_model.context import AutoContext
from ...nets.attention_model.dynamic_embedding import AutoDynamicEmbedding, AutoDynamicContextEmbedding
from ...nets.attention_model.multi_head_attention import (
    AttentionScore,
    MultiHeadAttention,
)

################################ Decoder ################################

class Decoder(nn.Module):
    """
    Pointer-style decoder with:
      - Global graph context (from graph token)
      - Dynamic state context (time, battery, load)
      - Multi-head glimpse
      - Tanh-clipped pointer

    embeddings: [B, N+1, D] if use_graph_token=True (0-th is [GRAPH])
                [B, N,   D] if use_graph_token=False
    """

    def __init__(
        self,
        embedding_dim,
        step_context_dim,
        n_heads,
        problem,
        tanh_clipping,
        use_graph_token: bool = False,
    ):
        super().__init__()

        # K/V/logit_K for each node: 3 * D
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)

        # Project graph embedding to fixed context (global query bias)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Project dynamic step context (prev node + state)
        # self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        
        self.project_step_context = nn.Sequential(
            nn.LayerNorm(step_context_dim),        # step_context_dim → stabilize
            nn.Linear(step_context_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Context & dynamic embedding factories (你原来的 AutoContext / AutoDynamicEmbedding)
        self.context = AutoContext(problem.NAME, {"context_dim": step_context_dim})
        self.dynamic_embedding = AutoDynamicEmbedding(
            problem.NAME, {"embedding_dim": embedding_dim}
        )
        self.dynamic_context_embedding = AutoDynamicContextEmbedding(
            problem.NAME, {"embedding_dim": embedding_dim}
        )

        # Glimpse MHA and pointer
        self.glimpse = MultiHeadAttention(embedding_dim=embedding_dim, n_heads=n_heads)
        self.pointer = AttentionScore(
            use_tanh=True,
            C=tanh_clipping,
            learn_scale=True,
            learn_C=False,   # 推荐 baseline
        )

        self.decode_type = None
        self.problem = problem
        self.use_graph_token = use_graph_token

    # ----------------- High-level API -----------------
    def set_decode_type(self, decode_type):
        assert decode_type in ["greedy", "sampling"]
        self.decode_type = decode_type

    # ----------------- Precompute K/V/logit_K -----------------

    def _precompute(self, embeddings, mask=None):
        if mask is not None:
            raise NotImplementedError  # (你以后要做 padding 的话再扩展)

        if self.use_graph_token:
            # embeddings: [B, N+1, D], 0-th is [GRAPH]
            graph_embed = embeddings[:, 0, :]    # [B,D]
            node_embed  = embeddings[:, 1:, :]   # [B,N,D]
        else:
            # No explicit graph token → fall back to mean pooling
            graph_embed = embeddings.mean(1)     # [B,D]
            node_embed  = embeddings             # [B,N,D]

        # Global graph context: [B,1,D]
        graph_context = self.project_fixed_context(graph_embed).unsqueeze(-2)

        # Node projections for glimpse & pointer
        glimpse_key, glimpse_val, logit_key = self.project_node_embeddings(node_embed).chunk(
            3, dim=-1
        )  # [B,N,D] each

        cache = (node_embed, graph_context, glimpse_key, glimpse_val, logit_key)
        return cache

    # ----------------- One decoding step -----------------

    def advance(self, cached_embeddings, state, node_mask=None):
        node_embeddings, graph_context, glimpse_K, glimpse_V, logit_K = cached_embeddings

        # Context(s,h): prev node embedding + (time, battery_used, load_used) etc.
        # AutoContext 内部已经在做 prev-node + state 的融合，你前面说的 concat + Linear 就是这里的实现
        context = self.context(node_embeddings, state)       # [B,1,step_context_dim]
        step_context = self.project_step_context(context)    # [B,1,D]
        
        # Query = global graph context + step context
        query = graph_context + step_context + self.dynamic_context_embedding(state)     # [B,1,D]

        # 动态 embedding（目前可以先用零向量，留接口）
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.dynamic_embedding(state)
        glimpse_K = glimpse_K + glimpse_key_dynamic
        glimpse_V = glimpse_V + glimpse_val_dynamic
        logit_K   = logit_K   + logit_key_dynamic

        # Feasibility mask from state
        mask = state.get_mask()  # [B,N]

        if node_mask is not None:
            # optional extra mask from outside (e.g., padded nodes)
            node_mask = node_mask.to(mask.device)
            mask = mask | node_mask

        logits, glimpse = self.calc_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        return logits, glimpse

    def calc_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        # glimpse: [B,1,D]
        glimpse = self.glimpse(query, glimpse_K, glimpse_V, mask)
        logits  = self.pointer(glimpse, logit_K, mask)   # [B,1,N]
        return logits, glimpse

    # ----------------- Decode strategy -----------------

    def decode(self, probs, mask):
        """
        probs: [B,N], mask: [B,N] (True = infeasible)
        """
        assert (probs == probs).all(), "Probs should not contain NaNs"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(-1)).data.any(), \
                "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print("Sampled bad values, resampling!")
                selected = probs.multinomial(1).squeeze(1)
        else:
            raise ValueError(f"Unknown decode type: {self.decode_type}")

        return selected
