"""LoopLM-style refiner — causal decoder with RMSNorm and SwiGLU.

Structurally matches Ouro/LoopLM: shared causal transformer layers
applied K times, RMSNorm throughout, SwiGLU activation.
Same interface as RefinementTransformer for drop-in compatibility.
"""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from models.refiner import ScalarTimeEmbedding, RMSNorm, TaperRMSNorm, AlphaReadout


def build_rms_norm(
    d_model: int,
    *,
    norm_type: str = "rmsnorm",
    taper_ema_decay: float = 0.99,
    alpha_init: float = 0.5,
    alpha_fixed: float | None = None,
    alpha_ema_decay: float = 0.99,
    alpha_s_ref_mode: str = "ema",
    alpha_init_log_s_ref: float = 0.0,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(d_model)
    if norm_type == "tapernorm":
        return TaperRMSNorm(d_model, ema_decay=taper_ema_decay)
    if norm_type == "alpha_readout":
        return AlphaReadout(
            d_model,
            ema_decay=alpha_ema_decay,
            init_alpha=alpha_init,
            fixed_alpha=alpha_fixed,
            s_ref_mode=alpha_s_ref_mode,
            init_log_s_ref=alpha_init_log_s_ref,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
    raise ValueError(f"unknown norm_type: {norm_type}")


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)  # gate
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, d = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, L, H, D]
        q = q.transpose(1, 2)  # [B, H, L, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product with causal + padding mask
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        # Use a large negative instead of -inf to avoid NaN when all keys are masked
        # (happens for early positions under causal mask when prefix is left-padded)
        NEG_INF = torch.finfo(attn.dtype).min
        if causal_mask is not None:
            attn = attn.masked_fill(causal_mask[:L, :L].unsqueeze(0).unsqueeze(0), NEG_INF)
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), NEG_INF)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, d)
        return self.out_proj(out)


class LoopDecoderLayer(nn.Module):
    """Pre-norm causal decoder layer with RMSNorm and SwiGLU."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_multiplier: int,
        dropout: float,
        *,
        norm_type: str = "rmsnorm",
        taper_ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.attn_norm = build_rms_norm(
            d_model,
            norm_type=norm_type,
            taper_ema_decay=taper_ema_decay,
        )
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.ffn_norm = build_rms_norm(
            d_model,
            norm_type=norm_type,
            taper_ema_decay=taper_ema_decay,
        )
        self.ffn = SwiGLU(d_model, d_model * ff_multiplier, dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), causal_mask=causal_mask, key_padding_mask=key_padding_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class RefinementLoopLM(nn.Module):
    """LoopLM-style refiner with same interface as RefinementTransformer."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ff_multiplier: int,
        dropout: float,
        prefix_length: int,
        suffix_length: int,
        pad_id: int,
        use_decode_norm: bool = True,
        decode_norm_type: str = "rmsnorm",
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

        self.layers = nn.ModuleList([
            LoopDecoderLayer(d_model, num_heads, ff_multiplier, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.update_head = nn.Linear(d_model, d_model)

        if decode_norm_type == "rmsnorm":
            self.decode_norm = RMSNorm(d_model)
        elif decode_norm_type == "layernorm":
            self.decode_norm = nn.LayerNorm(d_model)
        else:
            self.decode_norm = nn.Identity()

        # Register causal mask
        mask = torch.triu(torch.ones(self.total_length, self.total_length, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

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
        segment_ids = torch.cat([
            torch.zeros(prefix_length, dtype=torch.long, device=hidden.device),
            torch.ones(suffix_length, dtype=torch.long, device=hidden.device),
        ])
        segment_bias = self.segment_embedding(segment_ids).unsqueeze(0)
        time_bias = self.time_embedding(step_fraction, noise_level).view(batch_size, 1, self.d_model)
        hidden = self.input_dropout(hidden + position_bias + segment_bias + time_bias)

        # Build key padding mask: [B, total_length], True = padded
        key_padding_mask = None
        if prefix_padding_mask is not None or suffix_padding_mask is not None:
            if prefix_padding_mask is None:
                prefix_padding_mask = torch.zeros(
                    batch_size, prefix_length, dtype=torch.bool, device=hidden.device)
            if suffix_padding_mask is None:
                suffix_padding_mask = torch.zeros(
                    batch_size, suffix_length, dtype=torch.bool, device=hidden.device)
            key_padding_mask = torch.cat([prefix_padding_mask, suffix_padding_mask], dim=1)

        # Causal decoder layers
        for layer in self.layers:
            hidden = layer(hidden, causal_mask=self.causal_mask, key_padding_mask=key_padding_mask)

        hidden = self.final_norm(hidden)
        suffix_hidden = hidden[:, prefix_length:, :]
        return self.update_head(suffix_hidden)
