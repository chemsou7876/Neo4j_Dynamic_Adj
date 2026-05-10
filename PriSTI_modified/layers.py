import copy
import torch.nn as nn
import torch.nn.functional as F
import math
from generate_adj import *


def Attn_tem(heads=8, layers=1, channels=64):
    encoder_layer = TransformerEncoderLayer_QKV(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return TransformerEncoder_QKV(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer_QKV(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer_QKV, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer_QKV, self).__setstate__(state)

    def forward(self, query, key, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(query, key, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder_QKV(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder_QKV, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, key, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(query, key, output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


# ════════════════════════════════════════════════════════════
# MODIFIÉ : AdaptiveGCN accepte maintenant des adjacences
#           dynamiques (B, K, K) au lieu de statiques (K, K)
# ════════════════════════════════════════════════════════════
class AdaptiveGCN(nn.Module):
    def __init__(self, channels, order=2, include_self=True, device=None, is_adp=True, adj_file=None):
        super().__init__()
        self.order = order
        self.include_self = include_self
        c_in = channels
        c_out = channels

        # Toujours 2 supports (forward + backward)
        # La branche adaptive nn.Parameter est remplacée par A_dynamic
        self.support_len = 2

        c_in = (order * self.support_len + (1 if include_self else 0)) * c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x, base_shape, support_adp):
        B, channel, K, L = base_shape
        if K == 1:
            return x

        support = support_adp  # liste de (B, K, K)

        # Reshape : (B, C, K*L) → (B*L, C, K)
        x = x.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        x = x.unsqueeze(-1)  # (B*L, C, K, 1)

        out = [x] if self.include_self else []

        for a in support:
            # Expand adj pour chaque timestep : (B,K,K) → (B*L,K,K)
            a_expanded = a.unsqueeze(1).expand(-1, L, -1, -1).reshape(B * L, K, K)

            # Graph convolution : x @ a^T
            x_sq = x.squeeze(-1)                                    # (B*L, C, K)
            x1 = torch.bmm(x_sq, a_expanded.transpose(1, 2))       # (B*L, C, K)
            x1 = x1.unsqueeze(-1)                                   # (B*L, C, K, 1)
            out.append(x1)

            for k in range(2, self.order + 1):
                x1_sq = x1.squeeze(-1)
                x2 = torch.bmm(x1_sq, a_expanded.transpose(1, 2))
                x2 = x2.unsqueeze(-1)
                out.append(x2)
                x1 = x2

        out = torch.cat(out, dim=1)
        out = self.mlp(out)
        out = out.squeeze(-1)
        out = out.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return out


class TemporalLearning(nn.Module):
    def __init__(self, channels, nheads, is_cross=True):
        super().__init__()
        self.is_cross = is_cross
        self.time_layer = Attn_tem(heads=nheads, layers=1, channels=channels)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)

    def forward(self, y, base_shape, itp_y=None):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        v = y.permute(2, 0, 1)
        if self.is_cross:
            itp_y = itp_y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
            q = itp_y.permute(2, 0, 1)
            y = self.time_layer(q, q, v).permute(1, 2, 0)
        else:
            y = self.time_layer(v, v, v).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y


class SpatialLearning(nn.Module):
    def __init__(self, channels, nheads, target_dim, order, include_self, device, is_adp, adj_file, proj_t, is_cross):
        super().__init__()
        self.is_cross = is_cross
        self.feature_layer = SpaDependLearning(channels, nheads=nheads, order=order, target_dim=target_dim,
                                      include_self=include_self, device=device, is_adp=is_adp, adj_file=adj_file,
                                      proj_t=proj_t, is_cross=is_cross)

    def forward(self, y, base_shape, support, itp_y=None):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = self.feature_layer(y, base_shape, support, itp_y)
        return y


class SpaDependLearning(nn.Module):
    def __init__(self, channels, nheads, target_dim, order, include_self, device, is_adp, adj_file, proj_t, is_cross=True):
        super().__init__()
        self.is_cross = is_cross
        self.GCN = AdaptiveGCN(channels, order=order, include_self=include_self, device=device, is_adp=is_adp, adj_file=adj_file)
        self.attn = Attn_spa(dim=channels, seq_len=target_dim, k=proj_t, heads=nheads)
        self.cond_proj = Conv1d_with_init(2 * channels, channels, 1)
        self.norm1_local = nn.GroupNorm(4, channels)
        self.norm1_attn = nn.GroupNorm(4, channels)
        self.ff_linear1 = nn.Linear(channels, channels * 2)
        self.ff_linear2 = nn.Linear(channels * 2, channels)
        self.norm2 = nn.GroupNorm(4, channels)

    def forward(self, y, base_shape, support, itp_y=None):
        B, channel, K, L = base_shape
        y_in1 = y

        y_local = self.GCN(y, base_shape, support)
        y_local = y_in1 + y_local
        y_local = self.norm1_local(y_local)
        y_attn = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_cross:
            itp_y_attn = itp_y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
            y_attn = self.attn(y_attn.permute(0, 2, 1), itp_y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y_attn = self.attn(y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        y_attn = y_attn.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)

        y_attn = y_in1 + y_attn
        y_attn = self.norm1_attn(y_attn)

        y_in2 = y_local + y_attn
        y = F.relu(self.ff_linear1(y_in2.permute(0, 2, 1)))
        y = self.ff_linear2(y).permute(0, 2, 1)
        y = y + y_in2

        y = self.norm2(y)
        return y


class GuidanceConstruct(nn.Module):
    def __init__(self, channels, nheads, target_dim, order, include_self, device, is_adp, adj_file, proj_t):
        super().__init__()
        self.GCN = AdaptiveGCN(channels, order=order, include_self=include_self, device=device, is_adp=is_adp, adj_file=adj_file)
        self.attn_s = Attn_spa(dim=channels, seq_len=target_dim, k=proj_t, heads=nheads)
        self.attn_t = Attn_tem(heads=nheads, layers=1, channels=channels)
        self.norm1_local = nn.GroupNorm(4, channels)
        self.norm1_attn_s = nn.GroupNorm(4, channels)
        self.norm1_attn_t = nn.GroupNorm(4, channels)
        self.ff_linear1 = nn.Linear(channels, channels * 2)
        self.ff_linear2 = nn.Linear(channels * 2, channels)
        self.norm2 = nn.GroupNorm(4, channels)

    def forward(self, y, base_shape, support):
        B, channel, K, L = base_shape
        y_in1 = y

        y_local = self.GCN(y, base_shape, support)
        y_local = y_in1 + y_local
        y_local = self.norm1_local(y_local)

        y_attn_s1 = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y_attn_s = self.attn_s(y_attn_s1.permute(0, 2, 1)).permute(0, 2, 1)
        y_attn_s = y_attn_s.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        y_attn_s = y_in1 + y_attn_s
        y_attn_s = self.norm1_attn_s(y_attn_s)

        y_attn_t1 = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        v = y_attn_t1.permute(2, 0, 1)
        y_attn_t = self.attn_t(v, v, v).permute(1, 2, 0)
        y_attn_t = y_attn_t.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        y_attn_t = y_in1 + y_attn_t
        y_attn_t = self.norm1_attn_t(y_attn_t)

        y_in2 = y_local + y_attn_s + y_attn_t
        y = F.relu(self.ff_linear1(y_in2.permute(0, 2, 1)))
        y = self.ff_linear2(y).permute(0, 2, 1)
        y = y + y_in2

        y = self.norm2(y)
        return y


def default(val, default_val):
    return val if val is not None else default_val


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class Attn_spa(nn.Module):
    def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, itp_x=None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        v_len = n if itp_x is None else itp_x.shape[1]
        assert v_len == self.seq_len, f'the sequence length of the values must be {self.seq_len} - {v_len} given'

        q_input = x if itp_x is None else itp_x
        queries = self.to_q(q_input)
        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        k_input = x if itp_x is None else itp_x
        v_input = x

        keys = self.to_k(k_input)
        values = self.to_v(v_input) if not self.share_kv else keys
        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)