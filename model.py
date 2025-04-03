import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class LearnablePosEmb(nn.Module):
    def __init__(self, dim, max_positions=10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

        self.weights = nn.Parameter(torch.zeros(1, dim))
        self.bias = nn.Parameter(torch.zeros(1, dim))

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

    def forward(self, x):
        positions = x.unsqueeze(-1)

        half_dim = self.dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = positions * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb * self.weights + self.bias


class RotaryPositionEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for RoPE"
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.unsqueeze(0)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

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
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)


        self.C_Q = nn.Linear(embed_dim, d_compr)
        self.C_Q_C = nn.Linear(d_compr, embed_dim - d_rope)
        self.C_Q_R = nn.Linear(embed_dim, d_rope)
        
        self.C_KV = nn.Linear(embed_dim, d_compr)
        self.C_K_R = nn.Linear(embed_dim, d_rope)
        self.C_K_C = nn.Linear(d_compr, embed_dim - d_rope)
        self.C_V = nn.Linear(d_compr, embed_dim)

        self.activation = nn.GELU()

        self.rotary_pos_enc = RotaryPositionEncoding(d_rope)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        return mask

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        if mask is None:
            mask = self.generate_mask(seq_len, x.device)

        C_Q_proj = self.activation(self.C_Q(x))
        C_Q_C_proj = self.C_Q_C(C_Q_proj)
        C_Q_R_proj = self.C_Q_R(x)
        
        C_KV_proj = self.activation(self.C_KV(x))
        C_K_R_proj = self.C_K_R(x)
        C_K_C_proj = self.C_K_C(C_KV_proj)
        V = self.C_V(C_KV_proj).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        pos_emb = self.rotary_pos_enc(seq_len, x.device)
        C_Q_R_proj = apply_rotary_pos_emb(C_Q_R_proj, pos_emb)
        C_K_R_proj = apply_rotary_pos_emb(C_K_R_proj, pos_emb)

        Q = self.activation(torch.cat((C_Q_C_proj, C_Q_R_proj), dim=-1))
        K = self.activation(torch.cat((C_K_C_proj, C_K_R_proj), dim=-1))

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(attn_output)




class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, latent_dim, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, latent_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        norm_x = self.norm1(x)
        x = self.attn(norm_x) + x
        norm_x = self.norm2(x)
        x = self.ff(x) + x
        return x


class TheTransformer(nn.Module):
    def __init__(self, vocab_size, num_heads, n_layers, d_model, latent_dim, ignore_index, dropout):
        super().__init__()
        self.ignore_index = ignore_index
        self.embed = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = LearnablePosEmb(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, latent_dim, dropout)
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

        if targets is not None:
            loss = (F.cross_entropy(
                logits.transpose(1, 2), targets, ignore_index=self.ignore_index) * mask).sum()/mask.sum()
            return logits, loss
        return logits
