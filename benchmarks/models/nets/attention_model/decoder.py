   
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

class Decoder(nn.Module):
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

    def __init__(self, embedding_dim, step_context_dim, n_heads, problem, tanh_clipping):
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
        if mask is None:
            graph_embed = embeddings.mean(1)
        else:
            valid_mask = torch.tensor(~mask).to(embeddings.device)           
            valid_mask = valid_mask.unsqueeze(-1) 

            embeddings_valid = embeddings * valid_mask  
            sum_embeddings = embeddings_valid.sum(dim=1)  
            count_valid = valid_mask.sum(dim=1).clamp(min=1e-6)  

            graph_embed = sum_embeddings / count_valid

        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        graph_context = self.project_fixed_context(graph_embed).unsqueeze(-2)
        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key, glimpse_val, logit_key = self.project_node_embeddings(embeddings).chunk(
            3, dim=-1
        )

        cache = (
            embeddings,
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

        return logits, glimpse


# embedding_dim, self.embedding.context_dim, n_heads, self.problem, tanh_clipping

class Decoder_My(nn.Module):
    def __init__(self, embedding_dim, step_context_dim, decoder_head, problem, tanh_clipping, mask_logits = True, temp = 1.0, extra_dim = 1):
        super().__init__()
        self.decoder_head = decoder_head
        self.step_context_dim = step_context_dim
        self.norm_factor = 1 / math.sqrt(step_context_dim//self.decoder_head)
        self.tanh_clipping = tanh_clipping
        self.problem = problem
        self.mask_logits = mask_logits
        self.temp = temp
        self.decode_type = None

        self.time_embedding = nn.Linear(1, step_context_dim // decoder_head, bias=False)
        self.context_project = nn.Linear(embedding_dim, step_context_dim, bias=False)
        self.kvlogit_project = nn.Linear(embedding_dim, step_context_dim * 3, bias=False)
        self.project_step_context = nn.Linear(embedding_dim + extra_dim, step_context_dim, bias=False)
        self.project_out = nn.Linear(step_context_dim, step_context_dim, bias=False)

    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any nans"
        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _calc_log_likelihood(self, _log_p, a, mask):
        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask=None):
        batch_size, _, embed_dim = query.size()
        key_size = val_size = embed_dim // self.decoder_head

        # Get Glimpse Q
        glimpse_Q = query.view(batch_size, self.decoder_head, 1, key_size)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        if mask is not None:
            compatibility[mask[:, None, :, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, embedding_dim)
        glimpse = self.project_out(
            heads.transpose(1, 2).contiguous().view(-1, 1, self.decoder_head * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)

        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits and mask is not None:
            logits[mask.squeeze(1)] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_log_p(self, embeddings, context_node, glimpse_K, glimpse_V, logit_K, state, normalize=True, time=None):
        batch_size = glimpse_K.shape[0]
        current_node = state.get_current_node()
        next_nodes = torch.cat((
            torch.gather(embeddings, 1, current_node.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, embeddings.size(-1))).view(batch_size, 1, embeddings.size(-1)),
              self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, None], 
              self.problem.BATTERY_CAPACITY - state.used_battery[:, :, None], 
              state.current_time[:, :, None]),
        -1)

        # Compute query = context node embedding
        query = context_node + \
                self.project_step_context(next_nodes)

        # Compute the mask
        mask = state.get_mask_evrp()
        # mask_ = state.get_mask_()

        # if not (mask[0, :, 3:] == mask_[0, :, 3:]).all():
        #     breakpoint()
        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        # breakpoint()
        if normalize:
            log_p1 = torch.log_softmax(log_p / self.temp, dim=-1)
        if torch.isnan(log_p1).any():
            breakpoint()
            mask = state.get_mask()
        assert not torch.isnan(log_p1).any()

        return log_p1, mask

    def forward(self, input, embedding, mask=None, return_pi=False):
        outputs = []
        sequences = []
        chg_energy_list = []
        state = self.problem.make_state(input)
        # avg hidden states to get graph embedding
        # BR1
        graph_embedding = embedding.mean(1).unsqueeze(1)
        fixed_context = self.context_project(graph_embedding)

        # Get glimps KV, logits_k
        # BR2
        glimps_k, glimps_v, logits_k = self.kvlogit_project(embedding).chunk(3, dim=-1)
        batch_size, seq_len, hidden_dim = glimps_k.size()

        assert (hidden_dim%self.decoder_head)==0, "The dim of decoder cannot match its decoder head!"
        glimps_k = glimps_k.view(batch_size, seq_len, self.decoder_head, hidden_dim//self.decoder_head).contiguous().transpose(1, 2)
        glimps_v = glimps_v.view(batch_size, seq_len, self.decoder_head, hidden_dim//self.decoder_head).contiguous().transpose(1, 2)
        logits_k = logits_k.contiguous()

        # Perform decoding steps
        i = 0
        while not state.all_finished():
            # if self.decode_type == "greedy":
            log_p, mask = self._get_log_p(embedding, fixed_context, glimps_k, glimps_v, logits_k, state)
            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected = self._select_node(log_p.exp(), mask[:,0,:])  # Squeeze out steps dimension
            state = state.update(selected)
            energy = state.chrg_battery
            # energy_mask = (selected <= (input['loc'].shape[1] - input['demand'].shape[1])) & (selected > 0)

            # Collect output of step
            outputs.append(log_p)
            sequences.append(selected)
            chg_energy_list.append(energy)
            # print(torch.stack(sequences, 1)[0], state.used_capacity[0])
            # print(torch.stack(sequences, 1)[1], state.used_capacity[1])
            i += 1

        # Collected lists, return Tensor
        _log_p, pi = torch.stack(outputs, 1), torch.stack(sequences, 1)

        # k = 1
        # processed_chg_energy_list = [torch.exp(k * x) - 1 for x in chg_energy_list]
        # remain_energy = torch.sum(torch.stack(processed_chg_energy_list, dim=1), dim=1)
        # remain_energy = torch.sum(torch.stack(chg_energy_list, 1),dim=1)

        cost, mask = self.problem.get_costs(input, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)

        if return_pi:
            return cost, ll, pi

        return cost, ll, None