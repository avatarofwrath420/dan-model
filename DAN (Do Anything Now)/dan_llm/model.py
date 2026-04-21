from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DANConfig:
    vocab_size: int = 32000
    context_len: int = 2048
    n_layers: int = 12
    d_model: int = 720
    n_heads: int = 12
    d_ff: int = 2880
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    """
    RMSNorm with a single learned weight (no bias), matching common LLaMA-style usage.
    Uses a small epsilon for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


_TorchRMSNorm = getattr(nn, "RMSNorm", None)
Norm = _TorchRMSNorm if _TorchRMSNorm is not None else RMSNorm


class SwiGLU(nn.Module):
    def __init__(self, config: DANConfig):
        super().__init__()
        self.W1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.W2 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.W3 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W3(self.W1(x) * self.act(self.W2(x)))


def _rotate_half_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (..., d_head)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    y = torch.empty_like(x)
    y[..., 0::2] = y1
    y[..., 1::2] = y2
    return y


class RoPE(nn.Module):
    def __init__(self, config: DANConfig):
        super().__init__()
        self.rope_theta = float(config.rope_theta)
        self.max_seq = int(config.context_len)
        self.d_head = int(config.d_model // config.n_heads)
        if self.d_head % 2 != 0:
            raise ValueError("RoPE requires even d_head")

        t = torch.arange(self.max_seq, dtype=torch.float32)
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head))
        angles = t[:, None] * inv_freq[None, :]
        angles = angles.view(1, 1, self.max_seq, self.d_head // 2)  # (1,1,T,Dh/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def apply(self, q: torch.Tensor, k: torch.Tensor, T: int) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos[:, :, :T, :].to(device=q.device, dtype=q.dtype)
        sin = self.sin[:, :, :T, :].to(device=q.device, dtype=q.dtype)
        return _rotate_half_pairs(q, cos, sin), _rotate_half_pairs(k, cos, sin)


class TransformerBlock(nn.Module):
    def __init__(self, config: DANConfig, rope: RoPE):
        super().__init__()
        self.config = config
        self.rope = rope

        self.att_norm = Norm(config.d_model)
        self.ffn_norm = Norm(config.d_model)

        self.wq = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wk = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wv = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)

        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.config.n_heads
        Dh = D // H

        p = self.att_norm(x)
        q = self.wq(p).view(B, T, H, Dh).transpose(1, 2)
        k = self.wk(p).view(B, T, H, Dh).transpose(1, 2)
        v = self.wv(p).view(B, T, H, Dh).transpose(1, 2)
        q, k = self.rope.apply(q, k, T)

        att = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        att = att.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.wo(att)

        p = self.ffn_norm(x)
        x = x + self.ffn(p)
        return x


class DANModel(nn.Module):
    def __init__(self, config: DANConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RoPE(config)
        self.blocks = nn.ModuleList([TransformerBlock(config, self.rope) for _ in range(config.n_layers)])
        self.norm = Norm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.embed.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.lm_head(x)

