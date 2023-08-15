import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, std=0.02)
#         if isinstance(m, nn.Linear) and m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.LayerNorm):
#         nn.init.constant_(m.bias, 0)
#         nn.init.constant_(m.weight, 1.0)
#     elif isinstance(m, nn.Conv2d):
#         nn.init.kaiming_uniform_(m.weight, a=1)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    else:
        for p in m.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

def PosEncoding(feats, pos):
    if pos is not None:
        return feats + pos
    else:
        return feats

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None, act='gelu'):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU() if act =='gelu' else nn.ReLU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for both image and text
    """

    def __init__(self, q_dim, k_dim, embed_dim, num_heads, dropout=0.1, 
        clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_dim = q_dim
        self.k_dim = k_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(self.q_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.q_dim)
        self.clamp_min_for_underflow = clamp_min_for_underflow
        self.clamp_max_for_overflow = clamp_max_for_overflow

        self._reset_parameters()

    def extra_repr(self):
        return 'num_heads={}'.format(self.num_heads)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        # nn.init.xavier_uniform_(self.q_proj.weight)
        # nn.init.xavier_uniform_(self.k_proj.weight)
        # nn.init.xavier_uniform_(self.v_proj.weight)
        # nn.init.xavier_uniform_(self.out_proj.weight)
        trunc_normal_(self.q_proj.weight)
        trunc_normal_(self.k_proj.weight)
        trunc_normal_(self.v_proj.weight)
        trunc_normal_(self.out_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)
        self.out_proj.bias.data.fill_(0)

    def forward(self, q, k, v, attention_mask=None, return_attention=False):
        bsz, tgt_len, embed_dim = q.size()

        query_states = self.q_proj(q) * self.scale
        key_states = self._shape(self.k_proj(k), -1, bsz) # [B, num_heads, src_len, head_dim]
        value_states = self._shape(self.v_proj(v), -1, bsz) # [B, num_heads, src_len, head_dim]

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape) # [B*num_heads, tgt_len, head_dim]
        key_states = key_states.view(*proj_shape) # [B*num_heads, src_len, head_dim]
        value_states = value_states.view(*proj_shape) # [B*num_heads, src_len, head_dim]

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2)) # [B*num_heads, tgt_len, src_len]

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        if attention_mask is not None:
            # [bsz, src_len]
            assert (attention_mask.dim() == 2)
            
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1) # [B, 1, 1, src_len]
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            # attention_mask = attention_mask.masked_fill(attention_mask == 0, float("-inf"))
            # attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if return_attention:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = self.dropout(attn_weights)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        # return attn_output, attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        return attn_output, attn_weights
    
class MultiScaleAttention(nn.Module):
    """
    Multi-scale multi-head attention module for both image and text
    """

    def __init__(self, q_dim, k_dim, embed_dim, num_heads, num_levels, dropout=0.1, 
        clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        super(MultiScaleAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.head_dim = embed_dim // num_heads
        self.q_dim = q_dim
        self.k_dim = k_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(self.q_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.q_dim)
        self.level_weights = nn.Parameter(torch.ones((num_levels)), requires_grad=True)
        self.clamp_min_for_underflow = clamp_min_for_underflow
        self.clamp_max_for_overflow = clamp_max_for_overflow

        self._reset_parameters()

    def extra_repr(self):
        return 'num_heads={}'.format(self.num_heads)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        # nn.init.xavier_uniform_(self.q_proj.weight)
        # nn.init.xavier_uniform_(self.k_proj.weight)
        # nn.init.xavier_uniform_(self.v_proj.weight)
        # nn.init.xavier_uniform_(self.out_proj.weight)
        trunc_normal_(self.q_proj.weight)
        trunc_normal_(self.k_proj.weight)
        trunc_normal_(self.v_proj.weight)
        trunc_normal_(self.out_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)
        self.out_proj.bias.data.fill_(0)

    def ss_attention(self, q, k, v, attention_mask=None, return_attention=False):
        bsz, tgt_len, embed_dim = q.size()
        key_states = self._shape(k, -1, bsz) # [B, num_heads, src_len, head_dim]
        value_states = self._shape(v, -1, bsz) # [B, num_heads, src_len, head_dim]

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        q_states = self._shape(q, tgt_len, bsz).view(*proj_shape) # [B*num_heads, tgt_len, head_dim]
        key_states = key_states.view(*proj_shape) # [B*num_heads, src_len, head_dim]
        value_states = value_states.view(*proj_shape) # [B*num_heads, src_len, head_dim]

        src_len = key_states.size(1)
        attn_weights = torch.bmm(q_states, key_states.transpose(1, 2)) # [B*num_heads, tgt_len, src_len]

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        if attention_mask is not None:
            # [bsz, src_len]
            assert (attention_mask.dim() == 2)
            
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1) # [B, 1, 1, src_len]
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            # attention_mask = attention_mask.masked_fill(attention_mask == 0, float("-inf"))
            # attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if return_attention:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = self.dropout(attn_weights)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        return attn_output, attn_weights
    
    def ms_attention(self, q, k, v, k_spatial_shapes, attention_mask=None, return_attention=False):
        bsz, tgt_len, embed_dim = q.size()
        key_states = self._shape(k, -1, bsz) # [B, num_heads, src_len, head_dim]
        value_states = self._shape(v, -1, bsz) # [B, num_heads, src_len, head_dim]

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        q_states = self._shape(q, tgt_len, bsz).view(*proj_shape) # [B*num_heads, tgt_len, head_dim]
        key_states = key_states.view(*proj_shape) # [B*num_heads, src_len, head_dim]
        value_states = value_states.view(*proj_shape) # [B*num_heads, src_len, head_dim]

        src_len = key_states.size(1)
        attn_weights = torch.bmm(q_states, key_states.transpose(1, 2)) # [B*num_heads, tgt_len, src_len]

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        if attention_mask is not None:
            # [bsz, src_len]
            assert (attention_mask.dim() == 2)
            
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1) # [B, 1, 1, src_len]
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            # attention_mask = attention_mask.masked_fill(attention_mask == 0, float("-inf"))
            # attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_list = attn_weights.split([H_ * W_ for H_, W_ in k_spatial_shapes], dim=-1)
        attn_weights = torch.cat([nn.functional.softmax(attn_weights_list[i], dim=-1) * self.level_weights[i] for i in range(self.num_levels)], dim=-1)
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if return_attention:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = self.dropout(attn_weights)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        return attn_output, attn_weights
    
    def forward(self, q, k, v, k_spatial_shapes, attention_mask=None, return_attention=False):
        q = self.q_proj(q) * self.scale
        k = self.k_proj(k)
        v = self.v_proj(v)

        # k_list = k.split([H_ * W_ for H_, W_ in k_spatial_shapes],
        #                     dim=1)
        # v_list = v.split([H_ * W_ for H_, W_ in k_spatial_shapes],
        #                     dim=1)
        # mask_list = attention_mask.split([H_ * W_ for H_, W_ in k_spatial_shapes],
        #                     dim=1)
        # assert len(k_list) == len(v_list) == len(mask_list) == self.num_levels

        # attn_output_list = []
        # attn_weights_list = []
        # for level in range(self.num_levels):
        #     attn_output, attn_weights = self.ss_attention(q, k_list[level], v_list[level], mask_list[level], return_attention)
        #     attn_output_list.append(attn_output)
        #     attn_weights_list.append(attn_weights)
        # attn_output = torch.stack(attn_output_list, dim=-1) # [B, tgt_len, C, num_lvls]
        # attn_output = (attn_output * self.level_weights).sum(-1)  # [B, tgt_len, C]
        attn_output, attn_weights = self.ms_attention(q, k, v, k_spatial_shapes, attention_mask, return_attention)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

class MaskBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        '''
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm')
        '''
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout, act='relu')

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, proto, feat, proto_pos, feat_pos):
        proto2 = self.self_attn(query=proto+proto_pos, key=proto+proto_pos, value=proto)[0]
        proto = proto + self.dropout1(proto2)
        proto = self.norm1(proto)

        proto2 = self.cross_attn(query=proto+proto_pos, key=feat+feat_pos, value=feat)[0]
        proto = proto + self.dropout2(proto2)
        proto = self.norm2(proto)

        proto2 = self.mlp(proto)
        proto = proto + self.dropout3(proto2)
        proto = self.norm3(proto)

        return proto

class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim, k_dim, v_dim, heads, mlp_dim, dropout=0., norm_layer=nn.LayerNorm):
        '''
            operation_order=('cross_attn', 'norm', 'self_attn', 'norm', 
                                 'ffn', 'norm')
        '''
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, kdim=k_dim, vdim=v_dim)
        self.mlp = FeedForward(dim, mlp_dim, dropout, act='relu')

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, query, key, value, query_mask=None, key_mask=None):
        '''
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - query_mask/key_mask: :math:`(N, L/S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position
            with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        '''
        query2 = self.cross_attn(query=query, key=key, value=value, key_padding_mask=key_mask)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        query2 = self.self_attn(query=query, key=query, value=query, key_padding_mask=query_mask)[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.mlp(query)
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        return query

class TransformerDecoderLayer2(nn.Module):
    def __init__(self, dim, k_dim, v_dim, heads, mlp_dim, dropout, norm_layer=nn.LayerNorm):
        '''
            operation_order=('cross_attn', 'norm', 'ffn', 'norm')
        '''
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, kdim=k_dim, vdim=v_dim)
        self.mlp = FeedForward(dim, mlp_dim, dropout, act='relu')

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, value, query_mask=None, key_mask=None):
        '''
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - query_mask/key_mask: :math:`(N, L/S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position
            with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        '''
        query2 = self.cross_attn(query=query, key=key, value=value, key_padding_mask=key_mask)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        query2 = self.mlp(query)
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        return query

class TransformerDecoderLayer3(nn.Module):
    def __init__(self, q_dim, k_dim, embed_dim, heads, mlp_expand=2, dropout=0., norm_layer=nn.LayerNorm, with_gamma=False):
        '''
            operation_order=('self_attn', 'norm', 
                                 'ffn', 'norm')
        '''
        super().__init__()
        self.self_attn = MultiHeadAttention(q_dim, q_dim, embed_dim, heads, dropout=dropout, clamp_max_for_overflow=True, clamp_min_for_underflow=True)
        self.mlp = FeedForward(q_dim, q_dim*mlp_expand, dropout, act='relu')

        self.norm1 = norm_layer(q_dim)
        self.norm2 = norm_layer(q_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, query, query_mask=None, key_mask=None, query_pos=None, key_pos=None):
        '''
            - query: :math:`(N, L, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(N, S, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(N, S, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - query_mask/key_mask: :math:`(N, L/S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the zero positions will be ignored while the position
            with the non-zero positions will be unchanged.
        '''
        query2 = self.self_attn(q=PosEncoding(query, query_pos), k=PosEncoding(query, query_pos), v=query, attention_mask=query_mask)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        query2 = self.mlp(query)
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        return query

class TransformerDecoderLayer4(nn.Module):
    def __init__(self, q_dim, k_dim, embed_dim, heads, mlp_expand=2, dropout=0., norm_layer=nn.LayerNorm, with_gamma=False):
        '''
            operation_order=('cross_attn', 'norm', 'self_attn', 'norm', 
                                 'ffn', 'norm')
        '''
        super().__init__()
        self.self_attn = MultiHeadAttention(q_dim, q_dim, embed_dim, heads, dropout=dropout, clamp_max_for_overflow=True, clamp_min_for_underflow=True)
        self.cross_attn = MultiHeadAttention(q_dim, k_dim, embed_dim, heads, dropout=dropout, clamp_max_for_overflow=True, clamp_min_for_underflow=True)
        self.mlp = FeedForward(q_dim, q_dim*mlp_expand, dropout, act='relu')
        self.with_gamma = with_gamma
        if with_gamma:
            self.gamma = nn.Parameter(torch.ones((1, 1, q_dim)), requires_grad=True)

        self.norm1 = norm_layer(q_dim)
        self.norm2 = norm_layer(q_dim)
        self.norm3 = norm_layer(q_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.init_weights()

    def extra_repr(self):
        return 'with_gamma={}'.format(self.with_gamma)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, query, key, value, query_mask=None, key_mask=None, query_pos=None, key_pos=None):
        '''
            - query: :math:`(N, L, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(N, S, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(N, S, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - query_mask/key_mask: :math:`(N, L/S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the zero positions will be ignored while the position
            with the non-zero positions will be unchanged.
        '''
        query2 = self.cross_attn(q=PosEncoding(query, query_pos), k=PosEncoding(key, key_pos), v=value, attention_mask=key_mask)[0]
        if self.with_gamma:
            query2 = query2 * self.gamma
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        query2 = self.self_attn(q=PosEncoding(query, query_pos), k=PosEncoding(query, query_pos), v=query, attention_mask=query_mask)[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.mlp(query)
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        return query

class TransformerDecoderLayer5(nn.Module):
    def __init__(self, q_dim, k_dim, embed_dim, heads, mlp_expand=2, dropout=0., norm_layer=nn.LayerNorm, with_gamma=False, with_mlp=True):
        '''
            operation_order=('cross_attn', 'norm',
                                 'ffn', 'norm')
        '''
        super().__init__()
        self.cross_attn = MultiHeadAttention(q_dim, k_dim, embed_dim, heads, dropout=dropout, clamp_max_for_overflow=True, clamp_min_for_underflow=True)
        self.norm1 = norm_layer(q_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.with_gamma = with_gamma
        if with_gamma:
            self.gamma = nn.Parameter(torch.ones((1, 1, q_dim)), requires_grad=True)

        self.with_mlp = with_mlp
        if with_mlp:
            self.mlp = FeedForward(q_dim, q_dim*mlp_expand, dropout, act='relu')
            self.norm2 = norm_layer(q_dim)
            self.dropout2 = nn.Dropout(dropout)

        self.init_weights()

    def extra_repr(self):
        return 'with_gamma={}'.format(self.with_gamma)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, query, key, value, query_mask=None, key_mask=None, query_pos=None, key_pos=None):
        '''
            - query: :math:`(N, L, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(N, S, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(N, S, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - query_mask/key_mask: :math:`(N, L/S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the zero positions will be ignored while the position
            with the non-zero positions will be unchanged.
        '''
        query2 = self.cross_attn(q=PosEncoding(query, query_pos), k=PosEncoding(key, key_pos), v=value, attention_mask=key_mask)[0]
        if self.with_gamma:
            query2 = query2 * self.gamma
        query = query + self.dropout1(query2)
        query = self.norm1(query)
        
        if self.with_mlp:
            query2 = self.mlp(query)
            query = query + self.dropout2(query2)
            query = self.norm2(query)

        return query

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).view(x.size(0), -1, self.num_pos_feats*2) # [B,(HxW),C]
        return pos

class MaskFormerDecoder(nn.Module):
    def __init__(
        self,
        d_in,
        n_layers=6,
        n_heads=8,
        d_model=256,
        d_ff=2048,
        dropout=0.1,
        return_intermediate=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.return_intermediate = return_intermediate
        if d_in != d_model:
            self.decoder_input_proj = nn.Conv2d(d_in, d_model, kernel_size=1)
        else:
            self.decoder_input_proj = nn.Identity()

        self.blocks = nn.ModuleList(
            [MaskBlock(d_model, n_heads, d_ff, dropout) for i in range(n_layers)]
        )
        self.decoder_norm = nn.LayerNorm(d_model)
        self.pe_layer = PositionEmbeddingSine(d_model//2, normalize=True)
        self.apply(init_weights)

    def forward(self, feats, query_emb, cls_emb=None):
        '''
        Args:
            feats: [B, C_in, h, w]
            query_emb: [B, N, C] where N = num_queries
            cls_emb: None or [B, N, C]
        Returns:
            cls_out: [l, B, N, C]
        '''
        feats = self.decoder_input_proj(feats)
        B, C, h, w = feats.shape

        pos_emb = self.pe_layer(feats).transpose(0,1)
        feats = feats.view(B, C, -1).permute(2, 0, 1) # [(hw)xBxC]
        
        query_emb = query_emb.transpose(0,1) # [NxBxC]
        if cls_emb is not None:
            cls_out = cls_emb.transpose(0,1) # [NxBxC]
        else:
            cls_out = torch.zeros_like(query_emb) # [NxBxC]

        intermediate = []
        for blk in self.blocks:
            cls_out = blk(cls_out, feats, query_emb, pos_emb)
            if self.return_intermediate:
                intermediate.append(self.decoder_norm(cls_out))

        if self.return_intermediate:
            return torch.stack(intermediate, dim=0).transpose(1,2) # [l,B,N,C]

        cls_out = self.decoder_norm(cls_out)
        cls_out = cls_out.transpose(0,1).unsqueeze(0) # [1,B,N,C]
        return cls_out

class GridTransformerDecoderLayer(nn.Module):
    def __init__(self, num_grids, dim, k_dim, v_dim, heads, mlp_dim, dropout, norm_layer=nn.LayerNorm, mode='v2l'):
        '''
            operation_order=('cross_attn', 'norm', 'self_attn', 'norm', 
                                 'ffn', 'norm')
        '''
        super().__init__()
        assert mode == 'v2l' or mode == 'l2v'
        self.mode = mode
        d_vis = k_dim if mode == 'v2l' else dim
        self.pool = nn.AdaptiveAvgPool2d((num_grids, num_grids))
        self.absolute_pos_embed = nn.Parameter(torch.zeros(num_grids*num_grids, 1, d_vis))
        trunc_normal_(self.absolute_pos_embed, std=.02)

        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, kdim=k_dim, vdim=v_dim)
        self.mlp = FeedForward(dim, mlp_dim, dropout, act='relu')

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward_l2v(self, query, key, query_mask=None, key_mask=None):
        '''
            query: visual features [B, C, H, W]
            key/value: language features [L, B, C]
            key_mask: language mask [B, L]

            return: visual features [B, C, H, W]
        '''
        B, C, H, W = query.shape
        identity = query
        query = self.pool(query) # [B, C, G, G]
        query = query.flatten(2).permute(2,0,1) # [G^2, B, C]

        query2 = self.cross_attn(query=query+self.absolute_pos_embed, key=key, value=key, key_padding_mask=key_mask)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        identity = identity.flatten(2).permute(2,0,1) # [HW, B, C]
        query2 = self.self_attn(query=identity, key=query+self.absolute_pos_embed, value=query)[0]
        query = identity + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.mlp(query)
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        query = query.permute(1,2,0).reshape(B,C,H,W)
        return query

    def forward_v2l(self, query, key, query_mask=None, key_mask=None):
        '''
            query: language features [L, B, C]
            key/value: visual features [B, C, H, W]
            query_mask: language mask [B, L]

            return: language features [L, B, C]
        '''
        key = self.pool(key) # [B, C, G, G]
        key = key.flatten(2).permute(2,0,1) # [G^2, B, C]

        query2 = self.cross_attn(query=query, key=key+self.absolute_pos_embed, value=key)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        query2 = self.self_attn(query=query, key=query, value=query, key_padding_mask=query_mask)[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.mlp(query)
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        return query

    def forward(self, query, key, query_mask=None, key_mask=None):
        '''
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - query_mask/key_mask: :math:`(N, L/S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position
            with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        '''
        if self.mode == 'v2l':
            return self.forward_v2l(query, key, query_mask, key_mask)
        else:
            return self.forward_l2v(query, key, query_mask, key_mask)
