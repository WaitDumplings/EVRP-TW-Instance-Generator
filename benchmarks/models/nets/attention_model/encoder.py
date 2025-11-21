import torch
from torch import nn
import torch.nn.functional as F
from ...nets.attention_model.multi_head_attention import MultiHeadAttentionProj


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self,
        n_heads: int,
        embedding_dim: int,
        feed_forward_hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = MultiHeadAttentionProj(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
        )
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_hidden, embedding_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask=None, attn_bias=None):
        # x: [B, N, D]
        # mask: [B, N]  True = masked
        # attn_bias: [B, N, N] or [B, 1, N, N] (可选，用于 EVRPTW edge bias)

        # Self-Attention block (Pre-LN)
        h = self.norm1(x)
        h = self.attn(h, mask=mask, attn_bias=attn_bias)   # 你需要在 MultiHeadAttentionProj 里接 attn_bias
        x = x + h

        # FFN block (Pre-LN)
        h = self.norm2(x)
        h = self.ff(h)
        x = x + h
        return x


class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        feed_forward_hidden: int = 512,
        dropout: float = 0.1,
        use_graph_token: bool = False,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                MultiHeadAttentionLayer(
                    n_heads,
                    embed_dim,
                    feed_forward_hidden,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        self.use_graph_token = use_graph_token
        if use_graph_token:
            self.graph_token = nn.Parameter(torch.empty(1, 1, embed_dim))
            nn.init.xavier_uniform_(self.graph_token)
        else:
            self.graph_token = None

    def forward(self, x, mask=None, attn_bias=None):
        """
        x: [B, N, D]
        mask: [B, N]  True = masked (比如 padding / 不存在的节点)
        attn_bias: [B, N, N] (可选，将来可以放 EVRPTW 距离/时间窗 bias)
        """

        # 2) 扩展 attn_bias（如果有）
        if attn_bias is not None and self.graph_token is not None:
            # 原 attn_bias: [B, N, N]
            # 扩到 [B, N+1, N+1]，graph token 那一行/列先设为 0
            attn_bias = F.pad(attn_bias, (1, 0, 1, 0), value=0.0)

        # 3) 多层 attention
        for layer in self.layers:
            x = layer(x, mask=mask, attn_bias=attn_bias)

        # 4) final LN
        x = self.final_norm(x)  # [B, N(+1), D]

        # if self.graph_token is not None:
        #     graph_emb = x[:, 0, :]   # [B, D]
        #     node_features = x[:, 1:, :]
        # else:
        #     graph_emb = self._masked_mean(x, mask)  # [B,D]
        #     node_features = x

        return x

    @staticmethod
    def _masked_mean(x, mask):
        """当没有 graph token 时，用 mask 做平均池化"""
        if mask is None:
            return x.mean(dim=1)

        # mask: True = masked → valid_mask: 1 for valid nodes
        valid_mask = (~mask).unsqueeze(-1).type_as(x)   # [B,N,1]
        x_sum = (x * valid_mask).sum(dim=1)             # [B,D]
        counts = valid_mask.sum(dim=1).clamp(min=1e-6)  # [B,1]
        return x_sum / counts
