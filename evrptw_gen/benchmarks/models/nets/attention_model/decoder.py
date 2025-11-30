   
import torch
import torch.nn as nn
import math

import torch
from torch import nn

from ...nets.attention_model.context import AutoContext
from ...nets.attention_model.dynamic_embedding import AutoDynamicEmbedding
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
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)

        # Context & dynamic embedding factories (你原来的 AutoContext / AutoDynamicEmbedding)
        self.context = AutoContext(problem.NAME, {"context_dim": step_context_dim})
        self.dynamic_embedding = AutoDynamicEmbedding(
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

    def forward(self, input, embeddings):
        """
        input: problem-specific raw obs
        embeddings: encoder output
            - if use_graph_token: [B, N+1, D] (0-th is graph token)
            - else: [B, N, D]
        """
        outputs = []
        sequences = []

        state = self.problem.make_state(input)

        cached_embeddings = self._precompute(embeddings)

        while not state.all_finished():
            log_p, mask = self.advance(cached_embeddings, state)
            action = self.decode(log_p.exp(), mask)
            state = state.update(action)

            outputs.append(log_p)
            sequences.append(action)

        return torch.stack(outputs, 1), torch.stack(sequences, 1)

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
        query = graph_context + step_context                 # [B,1,D]

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

class Decoder2(nn.Module):
    r"""
    The decoder of the Attention Model.

    .. math::
        \{\log(\pmb{p}_t)\},\pi =  \mathrm{Decoder}(s, \pmb{h})

    First of all, precompute the keys and values for the embedding :math:`\pmb{h}`:

    .. math::
        \pmb{k}, \pmb{v}, \pmb{k}^\prime = W^K\pmb{h}, W^V\pmb{h}, W^{K^\prime}\pmb{h}
    and the projection of the graph embedding:

    .. math::
         W_{gc}\bar{\pmb{h}} \quad \text{ for } \bar{\pmb{h}} = \frac{1}{N}\sum\nolimits_i \pmb{h}_i.

    Then, the decoder iterates the decoding process autoregressively.
    In each decoding step, we perform multiple attentions to get the logits for each node.

    .. math::
        \begin{aligned}
        \pmb{h}_{(c)} &= [\bar{\pmb{h}}, \text{Context}(s,\pmb{h})]                                                 \\
        q & = W^Q \pmb{h}_{(c)} = W_{gc}\bar{\pmb{h}} + W_{sc}\text{Context}(s,\pmb{h})                               \\
        q_{gl} &= \mathrm{MultiHeadAttention}(q,\pmb{k},\pmb{v},\mathrm{mask}_t)                                    \\
        \pmb{p}_t &= \mathrm{Softmax}(\mathrm{AttentionScore}_{\text{clip}}(q_{gl},\pmb{k}^\prime, \mathrm{mask}_t))\\
        \pi_{t} &= \mathrm{DecodingStartegy}(\pmb{p}_t)                                                             \\
        \mathrm{mask}_{t+1} &= \mathrm{mask}_t.update(\pi_t).
        \end{aligned}



    .. note::
        If there are dynamic node features specified by :mod:`.dynamic_embedding` ,
        the keys and values projections are updated in each decoding step by

        .. math::
            \begin{aligned}
            \pmb{k}_{\text{dynamic}}, \pmb{v}_{\text{dynamic}}, \pmb{k}^\prime_{\text{dynamic}} &= \mathrm{DynamicEmbedding}(s)\\
            \pmb{k} &= \pmb{k} + \pmb{k}_{\text{dynamic}}\\
            \pmb{v} &= \pmb{v} +\pmb{v}_{\text{dynamic}} \\
            \pmb{k}^\prime &= \pmb{k}^\prime +\pmb{k}^\prime_{\text{dynamic}}.
            \end{aligned}
    .. seealso::
        * The :math:`\text{Context}` is defined in the :mod:`.context` module.
        * The :math:`\text{AttentionScore}` is defined by the :class:`.AttentionScore` class.
        * The :math:`\text{MultiHeadAttention}` is defined by the :class:`.MultiHeadAttention` class.

    Args:
         embedding_dim : the dimension of the embedded inputs
         step_context_dim : the dimension of the context :math:`\text{Context}(\pmb{x})`
         n_heads: number of heads in the :math:`\mathrm{MultiHeadAttention}`
         problem: an object defining the state and the mask updating rule of the problem
         tanh_clipping : the clipping scale of the pointer (attention layer before output)
    Inputs: input, embeddings
        * **input** : dict of inputs, for example ``{'loc': tensor, 'depot': tensor, 'demand': tensor}`` for CVRP.
        * **embeddings**: [batch, graph_size, embedding_dim]
    Outputs: log_ps, pi
        * **log_ps**: [batch, graph_size, T]
        * **pi**: [batch, T]

    """

    def __init__(self, embedding_dim, step_context_dim, n_heads, problem, tanh_clipping, use_graph_token = False):
        super(Decoder, self).__init__()
        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)

        self.context = AutoContext(problem.NAME, {"context_dim": step_context_dim})
        self.dynamic_embedding = AutoDynamicEmbedding(
            problem.NAME, {"embedding_dim": embedding_dim}
        )
        self.glimpse = MultiHeadAttention(embedding_dim=embedding_dim, n_heads=n_heads)
        self.pointer = AttentionScore(use_tanh=True, C=tanh_clipping)

        self.decode_type = None
        self.problem = problem
        self.use_graph_token = use_graph_token

    def forward(self, input, embeddings):
        outputs = []
        sequences = []

        state = self.problem.make_state(input)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        cached_embeddings = self._precompute(embeddings)

        # Perform decoding steps
        while not (state.all_finished()):

            log_p, mask = self.advance(cached_embeddings, state)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            # Squeeze out steps dimension
            action = self.decode(log_p.exp(), mask)
            state = state.update(action)

            # Collect output of step
            outputs.append(log_p)
            sequences.append(action)

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def set_decode_type(self, decode_type):
        r"""
        Currently support

        .. code-block:: python

            ["greedy", "sampling"]

        """
        self.decode_type = decode_type

    def decode(self, probs, mask):
        r"""
        Execute the decoding strategy specified by ``self.decode_type``.

        Inputs:
            * **probs**: [batch_size, graph_size]
            * **mask** (bool): [batch_size, graph_size]
        Outputs:
            * **idxs** (int): index of action chosen. [batch_size]
        """
        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(
                1, selected.unsqueeze(-1)
            ).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print("Sampled bad values, resampling!")
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, mask=None):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        if mask is not None:
            raise NotImplementedError
        
        if self.use_graph_token:
            # Use [GRAPH] token
            graph_embed = embeddings[:, 0, :]
            node_embed = embeddings[:, 1:, :]
        else:
            # AVG Pooling
            graph_embed = embeddings.mean(1)
            node_embed = embeddings

        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        graph_context = self.project_fixed_context(graph_embed).unsqueeze(-2)
        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key, glimpse_val, logit_key = self.project_node_embeddings(node_embed).chunk(
            3, dim=-1
        )

        cache = (
            node_embed,
            graph_context,
            glimpse_key,
            glimpse_val,
            logit_key,
        )  # single head for the final logit
        return cache

    def advance(self, cached_embeddings, state, node_mask=None):

        node_embeddings, graph_context, glimpse_K, glimpse_V, logit_K = cached_embeddings

        # Compute context node embedding: [graph embedding| prev node| problem-state-context]
        # [batch, 1, context dim]

        context = self.context(node_embeddings, state)
        step_context = self.project_step_context(context)  # [batch, 1, embed_dim]
        query = graph_context + step_context  # [batch, 1, embed_dim]

        # glimpse_key_dynamic, glimpse_val_dynamic and glimpse_val_dynamic are "0" for the furture improvements
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.dynamic_embedding(state)
        glimpse_K = glimpse_K + glimpse_key_dynamic
        glimpse_V = glimpse_V + glimpse_val_dynamic
        logit_K = logit_K + logit_key_dynamic
        # Compute the mask
        mask = state.get_mask()

        # if node_mask is not None:
        #     mask |= torch.tensor(node_mask).to(mask.device)
            
        # Compute logits (unnormalized log_p)
        logits, glimpse = self.calc_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        return logits, glimpse

    def calc_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        # Compute glimpse with multi-head-attention.
        # Then use glimpse as a query to compute logits for each node

        # [batch, 1, embed dim]
        glimpse = self.glimpse(query, glimpse_K, glimpse_V, mask)

        logits = self.pointer(glimpse, logit_K, mask)
        # (Pdb) logits.shape
        # torch.Size([1, 1, 49, 121])
        # (Pdb) glimpse.shape
        # torch.Size([1, 49, 256])
        return logits, glimpse