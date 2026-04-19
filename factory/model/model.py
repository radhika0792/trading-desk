"""
Decoder-only transformer for Born Traders.
All hyperparameters come from config.yaml — nothing is hardcoded.

Architecture:
- RoPE positional encoding
- GELU activation
- Weight tying between input embeddings and output projection
- Pre-norm (LayerNorm before attention and FFN)
"""

import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int
    n_layers: int
    n_heads: int
    hidden_dim: int
    ffn_dim: int
    dropout: float
    activation: str
    positional_encoding: str
    context_window: int
    weight_tying: bool

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)

    @property
    def head_dim(self) -> int:
        assert self.hidden_dim % self.n_heads == 0
        return self.hidden_dim // self.n_heads


def precompute_rope_freqs(head_dim: int, context_window: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE rotation frequencies — shape (context_window, head_dim/2, 2)."""
    assert head_dim % 2 == 0
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(context_window).float()
    freqs = torch.outer(positions, theta)  # (context_window, head_dim/2)
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)  # (context_window, head_dim/2, 2)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to query or key tensor.
    x: (batch, n_heads, seq_len, head_dim)
    freqs: (seq_len, head_dim/2, 2)
    """
    seq_len = x.shape[2]
    freqs = freqs[:seq_len]  # (seq_len, head_dim/2, 2)
    x_r = x.reshape(*x.shape[:-1], -1, 2)  # (batch, heads, seq, head_dim/2, 2)
    cos = freqs[..., 0]  # (seq_len, head_dim/2)
    sin = freqs[..., 1]
    x0, x1 = x_r[..., 0], x_r[..., 1]
    rotated = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return rotated.reshape(x.shape)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: (B, n_heads, T, head_dim)

        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, n_heads, T, T)
        attn = attn + mask[:T, :T]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_dim, cfg.ffn_dim, bias=False)
        self.fc2 = nn.Linear(cfg.ffn_dim, cfg.hidden_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.activation == "gelu":
            self.act = nn.GELU()
        elif cfg.activation == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = MultiHeadAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs, mask)
        x = x + self.ffn(self.norm2(x))
        return x


class BornTrader(nn.Module):
    """Decoder-only transformer. Load from config.yaml, never hardcode parameters."""

    def __init__(self, cfg: ModelConfig, gradient_checkpointing: bool = False):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = gradient_checkpointing
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

        if cfg.weight_tying:
            self.lm_head.weight = self.embedding.weight

        # Causal mask and RoPE frequencies — registered as buffers (not parameters)
        mask = torch.full((cfg.context_window, cfg.context_window), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask)

        freqs = precompute_rope_freqs(cfg.head_dim, cfg.context_window)
        self.register_buffer("rope_freqs", freqs)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch, seq_len)
        returns logits: (batch, seq_len, vocab_size)
        """
        x = self.drop(self.embedding(input_ids))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, self.rope_freqs, self.causal_mask, use_reentrant=False)
            else:
                x = block(x, self.rope_freqs, self.causal_mask)
        x = self.norm(x)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, config_path: str) -> "BornTrader":
        cfg = ModelConfig.from_yaml(config_path)
        return cls(cfg)


def memory_estimate(model: BornTrader) -> dict:
    """Estimate memory in MB for model weights and optimizer states."""
    param_count = model.count_parameters()
    weights_mb = param_count * 4 / 1024 / 1024  # float32
    optimizer_mb = param_count * 8 / 1024 / 1024  # AdamW: 2 states per param
    return {
        "parameters": param_count,
        "weights_mb": round(weights_mb, 1),
        "optimizer_mb": round(optimizer_mb, 1),
        "total_mb": round(weights_mb + optimizer_mb, 1),
    }


if __name__ == "__main__":
    config_path = Path(__file__).parent / "config.yaml"
    model = BornTrader.from_config(str(config_path))
    mem = memory_estimate(model)

    print(f"Parameters : {mem['parameters']:,}")
    print(f"Weights    : {mem['weights_mb']} MB (float32)")
    print(f"Optimizer  : {mem['optimizer_mb']} MB (AdamW)")
    print(f"Total      : {mem['total_mb']} MB")

    # Dummy forward pass on CPU (MPS not needed for validation)
    dummy = torch.randint(0, model.cfg.vocab_size, (1, 16))
    with torch.no_grad():
        logits = model(dummy)
    print(f"Forward pass OK — output shape: {logits.shape}")

    # MPS (Apple Silicon) check
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model_mps = model.to(device)
        dummy_mps = dummy.to(device)
        with torch.no_grad():
            out = model_mps(dummy_mps)
        print(f"MPS forward pass OK — shape: {out.shape}")
    else:
        print("MPS not available — CPU only")
