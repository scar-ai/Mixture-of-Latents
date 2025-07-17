import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint

class LearnablePosEmb(nn.Module):
    def __init__(self, dim, max_positions=10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

        self.weights = nn.Parameter(torch.zeros(1, dim))
        self.bias = nn.Parameter(torch.zeros(1, dim))
        
        self.register_buffer('pos_cache', torch.zeros(max_positions, dim))
        self.cache_filled = False

        self._init_with_sinusoidal()

    def _init_with_sinusoidal(self):
        half_dim = self.dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        pos = torch.arange(self.max_positions, dtype=torch.float).unsqueeze(1)
        emb = pos * emb.unsqueeze(0)
        sinusoidal = torch.cat((emb.sin(), emb.cos()), dim=-1)

        with torch.no_grad():
            self.weights.data.copy_(sinusoidal.mean(0, keepdim=True))
            self.bias.data.copy_(sinusoidal.std(0, keepdim=True))

    def _fill_cache(self, device):
        if not self.cache_filled or self.pos_cache.device != device:
            half_dim = self.dim // 2
            emb = math.log(self.max_positions) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float) * -emb)
            pos = torch.arange(self.max_positions, device=device, dtype=torch.float).unsqueeze(1)
            emb = pos * emb.unsqueeze(0)
            sinusoidal = torch.cat((emb.sin(), emb.cos()), dim=-1)
            with torch.no_grad():
                self.pos_cache[:self.max_positions].copy_(sinusoidal * self.weights + self.bias)
            self.cache_filled = True

    def forward(self, x):
        self._fill_cache(x.device)
        return self.pos_cache.index_select(0, x.long())

class RotaryPositionEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for RoPE"
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        self.register_buffer('pos_emb_cache', torch.zeros(0, dim))
        self.max_cached_len = 0

    def forward(self, seq_len, device):
        if seq_len > self.max_cached_len or self.pos_emb_cache.device != device:
            t = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.pos_emb_cache = emb.unsqueeze(0)
            self.max_cached_len = seq_len
        return self.pos_emb_cache

@torch.jit.script
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

@torch.jit.script
def apply_rotary_pos_emb(x, pos_emb):
    cos = pos_emb.cos()
    sin = pos_emb.sin()
    return (x * cos) + (rotate_half(x) * sin)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, d_compr=256, dropout=0.0, d_rope=128):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert d_rope % 2 == 0, "d_rope must be even for RoPE"
        
        self.embed_dim = embed_dim
        self.latent_dim = d_compr
        self.num_heads = num_heads
        self.head_dim = self.latent_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
        
        self.C_Q = nn.Linear(embed_dim, d_compr)
        self.C_Q_C = nn.Linear(d_compr, d_compr - d_rope)
        self.C_Q_R = nn.Linear(embed_dim, d_rope)
        
        self.C_KV = nn.Linear(embed_dim, d_compr)
        self.C_K_R = nn.Linear(embed_dim, d_rope)
        self.C_K_C = nn.Linear(d_compr, d_compr - d_rope)
        self.C_V = nn.Linear(d_compr, d_compr)

        self.activation = nn.GELU()

        self.rotary_pos_enc = RotaryPositionEncoding(d_rope)

        self.out_proj = nn.Linear(d_compr, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer('causal_mask_cache', torch.zeros(0, 0, dtype=torch.bool))
        self.max_cached_seq_len = 0

    def _get_causal_mask(self, seq_len, device):
        if seq_len > self.max_cached_seq_len or self.causal_mask_cache.device != device:
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
            self.causal_mask_cache = mask
            self.max_cached_seq_len = seq_len
        return self.causal_mask_cache[:seq_len, :seq_len]

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        if mask is None:
            mask = self._get_causal_mask(seq_len, x.device)

        C_Q_proj = self.activation(self.C_Q(x))
        C_Q_C_proj = self.C_Q_C(C_Q_proj)
        C_Q_R_proj = self.C_Q_R(x)
        
        C_KV_proj = self.activation(self.C_KV(x))
        C_K_R_proj = self.C_K_R(x)
        C_K_C_proj = self.C_K_C(C_KV_proj)
        

        pos_emb = self.rotary_pos_enc(seq_len, x.device)
        C_Q_R_proj = apply_rotary_pos_emb(C_Q_R_proj, pos_emb)
        C_K_R_proj = apply_rotary_pos_emb(C_K_R_proj, pos_emb)

        Q = self.activation(torch.cat((C_Q_C_proj, C_Q_R_proj), dim=-1))
        K = self.activation(torch.cat((C_K_C_proj, C_K_R_proj), dim=-1))
        V = self.C_V(C_KV_proj)


        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)


        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.latent_dim)
        
        return self.out_proj(attn_output)


class GatingNetwork(nn.Module):
    def __init__(self, embed_dim, num_experts, top_k, noise_std=1.0):
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.W_g = nn.Parameter(torch.randn(embed_dim, num_experts))
        self.W_noise = nn.Parameter(torch.randn(embed_dim, num_experts))

    def forward(self, x):
        logits = x @ self.W_g
        noise = torch.randn_like(logits) * self.noise_std
        noise = noise * F.softplus(x @ self.W_noise).clamp(max=10)
        noisy_logits = logits + noise


        top_k_values, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_values.float(), dim=-1).type_as(top_k_values)
        top_k_gates = torch.nan_to_num(top_k_gates, nan=0.0, posinf=0.0, neginf=0.0)
        return top_k_gates, top_k_indices
    
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim, dropout, num_experts, topk_experts):
        super().__init__()
        self.embed_dim = embed_dim


        self.attn = MultiHeadAttention(embed_dim, num_heads, latent_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.dr1 = nn.Dropout(dropout)


        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.dr2 = nn.Dropout(dropout)
        self.gating = GatingNetwork(embed_dim=embed_dim, num_experts=num_experts, top_k=topk_experts, noise_std=0.1)
        self.experts = nn.ModuleList([FeedForward(embed_dim, dropout) for _ in range(num_experts)])

    def forward(self, x):

        residual = x
        
        norm_x = self.ln1(x)
        attn_out = self.dr1(self.attn(norm_x))
        x = attn_out + residual
        
        residual = x
        norm_x = self.ln2(x)
        top_k_gates, top_k_indices = self.gating(norm_x)
        
        batch, seq, _ = norm_x.shape
        flattened_x = norm_x.view(-1, self.embed_dim)
        flattened_gates = top_k_gates.view(-1, top_k_gates.size(-1))
        flattened_indices = top_k_indices.view(-1, top_k_indices.size(-1))

        output = torch.zeros_like(flattened_x)
        for expert_idx, expert in enumerate(self.experts):
            mask = (flattened_indices == expert_idx)
            rows = torch.any(mask, dim=1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
                
            token_gates = flattened_gates[rows] * mask[rows].float()
            token_gates = token_gates.sum(dim=1)
            
            expert_out = expert(flattened_x[rows])
            output[rows] += expert_out * token_gates.unsqueeze(-1)
        
        experts_output = output.view(batch, seq, self.embed_dim)
        x = self.dr2(experts_output) + residual
        return x


class TheTransformer(nn.Module):
    def __init__(self, vocab_size, num_heads, n_layers, d_model, latent_dim, ignore_index, dropout, num_experts, topk_experts):
        super().__init__()
        self.ignore_index = ignore_index
        self.embed = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = LearnablePosEmb(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, latent_dim, dropout, num_experts, topk_experts)
            for _ in range(n_layers)
        ])
         
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, targets=None, mask=None):
        embeddings = self.embed(x)
        b, l, h = embeddings.shape
        seq_inx = torch.arange(x.size(1), device=x.device)
        positional_encoding = self.positional_encoding(seq_inx).reshape(1, l, h).expand(b, l, h)
        x = embeddings + positional_encoding

        for layer in self.layers:
            x = layer(x)

        logits = self.fc(x)
        logits = logits.clamp(min=-50, max=50)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        if targets is not None:
            loss = (F.cross_entropy(
                logits.transpose(1, 2).float(), targets, ignore_index=self.ignore_index) * mask).sum()/mask.sum()
            if not torch.isfinite(loss):
                loss = torch.tensor(0.0, device=logits.device)
            return logits, loss
        return logits
