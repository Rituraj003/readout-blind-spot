"""Residual MLP refiner — a minimal non-transformer architecture for the
architecture-generality ablation.

Same interface as RefinementTransformer: forward(), token_latents(), decode().
Uses a simple residual MLP (linear → GELU → linear + skip) applied to each
position independently, with no attention. This isolates the LayerNorm/readout
effect from any transformer-specific dynamics.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.refiner import ScalarTimeEmbedding


class ResidualMLPBlock(nn.Module):
    def __init__(self, d_model: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * ff_multiplier)
        self.fc2 = nn.Linear(d_model * ff_multiplier, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc2(self.dropout(F.gelu(self.fc1(x))))
        return residual + x


class RefinementMLP(nn.Module):
    """Position-wise residual MLP with the same interface as RefinementTransformer."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        num_heads: int,       # ignored, kept for config compatibility
        num_layers: int,
        ff_multiplier: int,
        dropout: float,
        prefix_length: int,
        suffix_length: int,
        pad_id: int,
        use_decode_norm: bool = True,
        decode_norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self.total_length = prefix_length + suffix_length
        self.pad_id = pad_id
        self.use_decode_norm = use_decode_norm

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(self.total_length, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.time_embedding = ScalarTimeEmbedding(d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(d_model, ff_multiplier, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.update_head = nn.Linear(d_model, d_model)
        if decode_norm_type == "rmsnorm":
            from models.refiner import RMSNorm
            self.decode_norm = RMSNorm(d_model)
        else:
            self.decode_norm = nn.LayerNorm(d_model)

    def token_latents(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(token_ids)

    def decode(self, suffix_latents: torch.Tensor) -> torch.Tensor:
        if self.use_decode_norm:
            suffix_latents = self.decode_norm(suffix_latents)
        return F.linear(suffix_latents, self.token_embedding.weight)

    def forward(
        self,
        *,
        prefix_tokens: torch.Tensor,
        suffix_latents: torch.Tensor,
        step_fraction: torch.Tensor,
        noise_level: torch.Tensor,
        prefix_padding_mask: torch.Tensor | None = None,
        suffix_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = prefix_tokens.size(0)
        prefix_length = prefix_tokens.size(1)
        suffix_length = suffix_latents.size(1)

        prefix_latents = self.token_embedding(prefix_tokens)
        hidden = torch.cat([prefix_latents, suffix_latents], dim=1)

        positions = torch.arange(self.total_length, device=hidden.device)
        position_bias = self.position_embedding(positions).unsqueeze(0)
        segment_ids = torch.cat(
            [
                torch.zeros(prefix_length, dtype=torch.long, device=hidden.device),
                torch.ones(suffix_length, dtype=torch.long, device=hidden.device),
            ]
        )
        segment_bias = self.segment_embedding(segment_ids).unsqueeze(0)
        time_bias = self.time_embedding(step_fraction, noise_level).view(batch_size, 1, self.d_model)
        hidden = self.input_dropout(hidden + position_bias + segment_bias + time_bias)

        # Position-wise MLP — no attention, no cross-position interaction
        for block in self.blocks:
            hidden = block(hidden)

        hidden = self.final_norm(hidden)
        suffix_hidden = hidden[:, prefix_length:, :]
        return self.update_head(suffix_hidden)
