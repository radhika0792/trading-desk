"""
Trading Council — Chief of Staff (CoS) Model Definition.

Decoder-only transformer built from scratch in PyTorch.
Architecture: 24-layer, 768 hidden dim, 12 heads, RoPE, GELU, weight tying.
All hyperparameters loaded from config.yaml — nothing hardcoded.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


@dataclass
class TransformerConfig:
    """Model configuration loaded from config.yaml."""

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
    def from_yaml(cls, path: str) -> "TransformerConfig":
        """Load config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


def precompute_rope_freqs(
    head_dim: int,
    context_window: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE cosine and sine frequency tables.

    Returns cos and sin tensors of shape (context_window, head_dim // 2),
    cached once at model init and reused every forward pass.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    theta = 1.0 / (
        10000.0
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    positions = torch.arange(context_window, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, theta)  # (context_window, head_dim // 2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embedding to a query or key tensor.

    Args:
        x:   (batch, n_heads, seq_len, head_dim)
        cos: (context_window, head_dim // 2)
        sin: (context_window, head_dim // 2)

    Returns:
        Tensor of same shape as x with positions encoded.
    """
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim // 2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    # Split head_dim into two halves, rotate each pair
    x1, x2 = x.chunk(2, dim=-1)  # each: (batch, n_heads, seq_len, head_dim // 2)
    return torch.cat(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
        dim=-1,
    )


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention with RoPE positional encoding."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.hidden_dim % config.n_heads == 0, (
            f"hidden_dim ({config.hidden_dim}) must be divisible by n_heads ({config.n_heads})"
        )
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_dim // config.n_heads

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (batch, seq_len, hidden_dim)
            cos:  (context_window, head_dim // 2)
            sin:  (context_window, head_dim // 2)
            mask: (seq_len, seq_len) causal mask, 1 = attend, 0 = mask out

        Returns:
            (batch, seq_len, hidden_dim)
        """
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to queries and keys only (not values)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, n_heads, T, T)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)          # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.fc2 = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: (batch, seq_len, hidden_dim) — Returns: same shape."""
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block: Pre-LN attention + Pre-LN FFN with residuals."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pre-LayerNorm residual connections (more stable than post-LN)."""
        x = x + self.attn(self.ln1(x), cos, sin, mask)
        x = x + self.ffn(self.ln2(x))
        return x


class TradingCouncilModel(nn.Module):
    """
    Trading Council language model — 189M parameter decoder-only transformer.

    Built from scratch. No pretrained weights. All config from config.yaml.
    Shared by all 14 agents — each agent loads its own trained weights.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_final = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying: output projection shares input embedding matrix.
        # Reduces parameters by vocab_size * hidden_dim (~19M) with no quality loss.
        if config.weight_tying:
            self.lm_head.weight = self.token_embedding.weight

        # Precompute and register RoPE frequency tables as non-trainable buffers.
        # Buffers move with the model (CPU/GPU) and are saved in checkpoints.
        head_dim = config.hidden_dim // config.n_heads
        cos, sin = precompute_rope_freqs(
            head_dim, config.context_window, device=torch.device("cpu")
        )
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        # Weight initialisation (GPT-2 style)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise linear and embedding weights with small normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token indices
            mask:      (seq_len, seq_len) attention mask — causal mask built
                       automatically if not provided

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = input_ids.shape
        assert T <= self.config.context_window, (
            f"Sequence length {T} exceeds context window {self.config.context_window}"
        )

        x = self.embedding_dropout(self.token_embedding(input_ids))

        # Build causal mask if not supplied: lower triangular, 1 = attend
        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=input_ids.device))

        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin, mask)

        x = self.ln_final(x)
        return self.lm_head(x)  # (B, T, vocab_size)

    def count_parameters(self) -> int:
        """Total parameter count including shared (weight-tied) parameters."""
        return sum(p.numel() for p in self.parameters())

    def count_unique_parameters(self) -> int:
        """Unique parameter count — deduplicated for weight-tied tensors."""
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total
