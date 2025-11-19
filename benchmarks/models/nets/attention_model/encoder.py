import torch
from torch import nn
from ...nets.attention_model.multi_head_attention import MultiHeadAttentionProj

class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, x):  # x: [B, N, D]
        return self.normalizer(x)

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_heads, embedding_dim, feed_forward_hidden=512):
        super(MultiHeadAttentionLayer, self).__init__()

        self.norm1 = Normalization(embedding_dim)
        self.attn = MultiHeadAttentionProj(embedding_dim=embedding_dim, n_heads=n_heads)

        self.norm2 = Normalization(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(feed_forward_hidden, embedding_dim),
        )
        self.norm3 = Normalization(embedding_dim)

    def forward(self, x, mask=None):
        x = self.norm1(x)
        x = x + self.attn(x, mask = mask)
        x = self.norm2(x)
        x = x + self.ff(x)
        x = self.norm3(x)
        return x

class GraphAttentionEncoder(nn.Module):
    def __init__(self, n_heads, embed_dim, n_layers, feed_forward_hidden=512):
        super(GraphAttentionEncoder, self).__init__()

        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden)
            for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask = mask)

        if mask is not None:
            # mask: [batch_size, graph_size] -> bool tensor where True = masked
            # valid_mask: True where NOT masked
            valid_mask = torch.tensor(~mask).to(x.device)  # shape: [B, N]
            valid_mask = valid_mask.unsqueeze(-1).type_as(x)  # [B, N, 1] and float

            x_sum = (x * valid_mask).sum(dim=1)  # [B, D]
            counts = valid_mask.sum(dim=1).clamp(min=1e-6)  # avoid division by zero
            x_mean = x_sum / counts  # [B, D]
        else:
            x_mean = x.mean(dim=1)  # [B, D]

        return x, x_mean
