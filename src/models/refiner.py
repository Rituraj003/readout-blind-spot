from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ScalarTimeEmbedding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, step_fraction: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        if step_fraction.ndim == 0:
            step_fraction = step_fraction.expand_as(noise_level)
        inputs = torch.stack([step_fraction, noise_level], dim=-1)
        return self.net(inputs)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (scale-invariant, like LayerNorm)."""
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class TaperRMSNorm(nn.Module):
    """RMSNorm that can taper into a sample-independent scaling branch.

    During gate warmup, ``gate=1`` and this is exactly RMSNorm. The training loop
    then calibrates a fixed scaling branch from EMA statistics and cosine-decays
    the gate to zero:

        gate * RMSNorm(x) + (1 - gate) * c * x * scale_weight

    This implements the RMSNorm variant of TaperNorm from Kanavalau et al. The
    The sample-independent branch has its own trainable gain, initialized from
    the RMSNorm gain at calibration, so it can keep adapting during tapering.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.ema_decay = ema_decay

        self.register_buffer("gate", torch.tensor(1.0))
        self.scale_weight = nn.Parameter(torch.ones(d_model))
        self.register_buffer("scale_coeff", torch.tensor(1.0))
        self.register_buffer("ema_num", torch.tensor(0.0))
        self.register_buffer("ema_den", torch.tensor(0.0))
        self.register_buffer("ema_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("calibrated", torch.tensor(False))
        self._calibrated = False

    def set_gate(self, gate: float) -> None:
        self.gate.fill_(float(max(0.0, min(1.0, gate))))

    def calibrate(self) -> None:
        self.scale_weight.data.copy_(self.weight.detach())
        if self.ema_count.item() > 0 and self.ema_den.item() > 0:
            coeff = (self.ema_num / self.ema_den).clamp(min=1e-6, max=1e6)
            self.scale_coeff.copy_(coeff)
        else:
            self.scale_coeff.fill_(1.0)
        self.calibrated.fill_(True)
        self._calibrated = True

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self._calibrated = bool(self.calibrated.item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = x / rms * self.weight

        if self.training and not self._calibrated:
            with torch.no_grad():
                x_float = x.detach().float()
                weight_float = self.weight.detach().float()
                weighted_sq = (x_float * weight_float).pow(2).sum(dim=-1)
                rms_float = torch.sqrt(x_float.pow(2).mean(dim=-1) + self.eps)
                num = (weighted_sq / rms_float).mean()
                den = weighted_sq.mean()
                if self.ema_count.item() == 0:
                    self.ema_num.copy_(num)
                    self.ema_den.copy_(den)
                else:
                    decay = self.ema_decay
                    self.ema_num.mul_(decay).add_(num, alpha=1.0 - decay)
                    self.ema_den.mul_(decay).add_(den, alpha=1.0 - decay)
                self.ema_count.add_(1)

        gate = self.gate.to(dtype=x.dtype, device=x.device)
        scaled = x * self.scale_weight.to(dtype=x.dtype, device=x.device) * self.scale_coeff.to(
            dtype=x.dtype,
            device=x.device,
        )
        return gate * normed + (1.0 - gate) * scaled


class AlphaReadout(nn.Module):
    """Power-modulated readout: gamma(s) = s_ref^(1-alpha) * s^alpha.

    Provides an exact scalar continuum between RMSNorm-style and raw readout:
        alpha = 0: gamma = s_ref (constant scalar)  ->  RMSNorm-without-per-channel-gain
        alpha = 1: gamma = s                        ->  raw readout (h)
        0 < alpha < 1: partial radial sensitivity

    Output is gamma * u where s = RMS(h) and u = h / s.
    The downstream lm_head then maps this to logits.

    Implementation notes:
    - s_ref can be an EMA, fixed scalar, or learnable scalar. EMA is kept as the
      default for checkpoint/config compatibility, but fixed/learned anchors are
      safer for terminal-only supervision because alpha=0 then truly gives a
      constant-magnitude output rather than following hidden-state norm drift.
    - Uses RMS(h), not full norm: dimension-independent across model sizes.
    - alpha is parameterized as sigmoid(alpha_logit), shared across calls.
    - Computation in log-space for numerical stability.
    - No per-channel gain (clean math: alpha=1 exactly equals raw readout).
    """

    def __init__(
        self,
        d_model: int,
        ema_decay: float = 0.99,
        init_alpha: float = 0.5,
        fixed_alpha: float | None = None,
        s_ref_mode: str = "ema",
        init_log_s_ref: float = 0.0,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.ema_decay = ema_decay
        self.fixed_alpha = fixed_alpha
        self.s_ref_mode = s_ref_mode
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

        if s_ref_mode not in {"ema", "fixed", "learned"}:
            raise ValueError(f"s_ref_mode must be one of ema/fixed/learned, got {s_ref_mode}")
        if not (0.0 <= self.alpha_min < self.alpha_max <= 1.0):
            raise ValueError(
                f"alpha_min/alpha_max must satisfy 0 <= min < max <= 1, "
                f"got {self.alpha_min}/{self.alpha_max}"
            )

        if fixed_alpha is not None:
            assert 0.0 <= fixed_alpha <= 1.0, f"fixed_alpha must be in [0, 1], got {fixed_alpha}"
            self.register_buffer("alpha_value", torch.tensor(float(fixed_alpha)))
            self.register_buffer("alpha_logit", torch.tensor(0.0))
        else:
            init_alpha_clamped = float(min(max(init_alpha, self.alpha_min + 1e-4), self.alpha_max - 1e-4))
            init_unit = (init_alpha_clamped - self.alpha_min) / (self.alpha_max - self.alpha_min)
            init_unit = float(min(max(init_unit, 1e-4), 1.0 - 1e-4))
            init_logit = float(torch.logit(torch.tensor(init_unit)).item())
            self.alpha_logit = nn.Parameter(torch.tensor(init_logit))

        if s_ref_mode == "learned":
            self.log_s_ref = nn.Parameter(torch.tensor(float(init_log_s_ref)))
        else:
            self.register_buffer("log_s_ref", torch.tensor(float(init_log_s_ref)))

        # Kept for strict loading of older EMA checkpoints. It is only updated in
        # EMA mode, but fixed/learned mode can still load old state_dict keys.
        self.register_buffer("ema_count", torch.tensor(0, dtype=torch.long))

    def get_alpha(self) -> torch.Tensor:
        if self.fixed_alpha is not None:
            return self.alpha_value
        unit_alpha = torch.sigmoid(self.alpha_logit)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * unit_alpha

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [..., d_model]
        s = h.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=self.eps)
        log_s = s.log()
        u = h / s

        if self.training and self.s_ref_mode == "ema":
            with torch.no_grad():
                log_s_mean = log_s.mean().detach()
                if self.ema_count.item() == 0:
                    self.log_s_ref.copy_(log_s_mean)
                else:
                    self.log_s_ref.mul_(self.ema_decay).add_(
                        log_s_mean, alpha=1.0 - self.ema_decay
                    )
                self.ema_count.add_(1)

        alpha = self.get_alpha()
        log_s_ref = self.log_s_ref if self.s_ref_mode == "learned" else self.log_s_ref.detach().clone()
        log_gamma = (1.0 - alpha) * log_s_ref + alpha * log_s
        gamma = log_gamma.exp()
        return gamma * u


class RefinementTransformer(nn.Module):
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
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_model * ff_multiplier,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.update_head = nn.Linear(d_model, d_model)
        if decode_norm_type == "rmsnorm":
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
        if prefix_length != self.prefix_length or suffix_length != self.suffix_length:
            raise ValueError("prefix or suffix length does not match model configuration")

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

        key_padding_mask = None
        if prefix_padding_mask is not None or suffix_padding_mask is not None:
            if prefix_padding_mask is None:
                prefix_padding_mask = torch.zeros(
                    batch_size,
                    prefix_length,
                    dtype=torch.bool,
                    device=hidden.device,
                )
            if suffix_padding_mask is None:
                suffix_padding_mask = torch.zeros(
                    batch_size,
                    suffix_length,
                    dtype=torch.bool,
                    device=hidden.device,
                )
            key_padding_mask = torch.cat([prefix_padding_mask, suffix_padding_mask], dim=1)

        for block in self.blocks:
            hidden = block(hidden, src_key_padding_mask=key_padding_mask)

        hidden = self.final_norm(hidden)
        suffix_hidden = hidden[:, prefix_length:, :]
        return self.update_head(suffix_hidden)
