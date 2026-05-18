"""Autoregressive LoopLM for the non-denoising experiment.

Causal decoder with shared layers applied K times. RMSNorm, SwiGLU.
Returns logits at each loop iteration for per-step vs terminal supervision.

Two variants:
- inter_loop_norm=True: apply norm between loops (Ouro-style)
- inter_loop_norm=False: no norm between loops (raw accumulation)
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.loop_refiner import LoopDecoderLayer, build_rms_norm
from models.refiner import TaperRMSNorm


class AutoregressiveLoopLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ff_multiplier: int,
        dropout: float,
        max_seq_len: int,
        num_loops: int = 4,
        inter_loop_norm: bool = True,
        use_decode_norm: bool = True,
        decode_norm_final_only: bool = False,
        block_norm_type: str = "rmsnorm",
        loop_norm_type: str = "rmsnorm",
        decode_norm_type: str = "rmsnorm",
        taper_ema_decay: float = 0.99,
        alpha_init: float = 0.5,
        alpha_fixed: float | None = None,
        alpha_ema_decay: float = 0.99,
        alpha_s_ref_mode: str = "ema",
        alpha_init_log_s_ref: float = 0.0,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        per_loop_alpha: bool = False,
        use_spectral_damping: bool = False,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_loops = num_loops
        self.inter_loop_norm = inter_loop_norm
        self.use_decode_norm = use_decode_norm
        self.decode_norm_final_only = decode_norm_final_only
        self.use_spectral_damping = use_spectral_damping
        self.block_norm_type = block_norm_type
        self.loop_norm_type = loop_norm_type
        self.decode_norm_type = decode_norm_type

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            LoopDecoderLayer(
                d_model,
                num_heads,
                ff_multiplier,
                dropout,
                norm_type=block_norm_type,
                taper_ema_decay=taper_ema_decay,
            )
            for _ in range(num_layers)
        ])

        self.loop_norm = build_rms_norm(
            d_model,
            norm_type=loop_norm_type,
            taper_ema_decay=taper_ema_decay,
            alpha_init=alpha_init,
            alpha_fixed=alpha_fixed,
            alpha_ema_decay=alpha_ema_decay,
            alpha_s_ref_mode=alpha_s_ref_mode,
            alpha_init_log_s_ref=alpha_init_log_s_ref,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )  # applied between loops if inter_loop_norm
        # If per_loop_alpha and decode is alpha_readout, build K independent
        # AlphaReadout modules so each loop iteration can learn its own alpha.
        # Otherwise, a single shared decode_norm is reused across loops.
        self.per_loop_alpha = bool(per_loop_alpha and decode_norm_type == "alpha_readout"
                                    and use_decode_norm)
        if not use_decode_norm:
            self.decode_norm = nn.Identity()
        elif self.per_loop_alpha:
            self.decode_norm = nn.ModuleList([
                build_rms_norm(
                    d_model,
                    norm_type=decode_norm_type,
                    taper_ema_decay=taper_ema_decay,
                    alpha_init=alpha_init,
                    alpha_fixed=alpha_fixed,
                    alpha_ema_decay=alpha_ema_decay,
                    alpha_s_ref_mode=alpha_s_ref_mode,
                    alpha_init_log_s_ref=alpha_init_log_s_ref,
                    alpha_min=alpha_min,
                    alpha_max=alpha_max,
                )
                for _ in range(num_loops)
            ])
        else:
            self.decode_norm = build_rms_norm(
                d_model,
                norm_type=decode_norm_type,
                taper_ema_decay=taper_ema_decay,
                alpha_init=alpha_init,
                alpha_fixed=alpha_fixed,
                alpha_ema_decay=alpha_ema_decay,
                alpha_s_ref_mode=alpha_s_ref_mode,
                alpha_init_log_s_ref=alpha_init_log_s_ref,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
            )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Learned exit gate (Ouro-style): per-loop scalar probability of exiting
        self.exit_gate = nn.Linear(d_model, 1, bias=True)

        # Spectral damping (OpenMythos-style): constrain recurrent dynamics
        # α = sigmoid(raw_alpha) ensures α ∈ (0, 1), so ρ(A) < 1
        self.raw_alpha = nn.Parameter(torch.tensor(2.0))  # sigmoid(2.0) ≈ 0.88

        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Causal mask
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def taper_norms(self) -> list[TaperRMSNorm]:
        return [m for m in self.modules() if isinstance(m, TaperRMSNorm)]

    def set_taper_gate(self, gate: float) -> None:
        for module in self.taper_norms():
            module.set_gate(gate)

    def calibrate_taper_norms(self) -> None:
        for module in self.taper_norms():
            if not bool(module.calibrated.item()):
                module.calibrate()

    def taper_summary(self) -> dict[str, float | int]:
        modules = self.taper_norms()
        if not modules:
            return {"count": 0, "calibrated": 0, "gate": 1.0, "scale_coeff": 1.0}
        calibrated = sum(int(bool(module.calibrated.item())) for module in modules)
        gate = sum(float(module.gate.item()) for module in modules) / len(modules)
        scale_coeff = sum(float(module.scale_coeff.item()) for module in modules) / len(modules)
        return {
            "count": len(modules),
            "calibrated": calibrated,
            "gate": gate,
            "scale_coeff": scale_coeff,
        }

    def alpha_readouts(self):
        from models.refiner import AlphaReadout
        return [m for m in self.modules() if isinstance(m, AlphaReadout)]

    def alpha_summary(self) -> dict:
        modules = self.alpha_readouts()
        if not modules:
            return {"count": 0, "alpha": -1.0, "log_s_ref": 0.0, "s_ref_mode": "none",
                    "per_loop_alpha": [], "per_loop_log_s_ref": []}
        per_loop_alpha = [float(m.get_alpha().item()) for m in modules]
        per_loop_log_s_ref = [float(m.log_s_ref.item()) for m in modules]
        alpha_mean = sum(per_loop_alpha) / len(per_loop_alpha)
        log_s_ref_mean = sum(per_loop_log_s_ref) / len(per_loop_log_s_ref)
        s_ref_modes = sorted({m.s_ref_mode for m in modules})
        return {
            "count": len(modules),
            "alpha": alpha_mean,
            "log_s_ref": log_s_ref_mean,
            "s_ref_mode": ",".join(s_ref_modes),
            "per_loop_alpha": per_loop_alpha,
            "per_loop_log_s_ref": per_loop_log_s_ref,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        """Forward pass with K loop iterations.

        Returns:
            all_logits: list of K tensors, each [B, L, V]
            all_norms: list of K floats, mean hidden state norm per loop
            final_logits: [B, L, V] from the last loop
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Embed
        positions = torch.arange(L, device=device)
        h = self.token_embedding(input_ids) + self.position_embedding(positions)
        h = self.input_dropout(h)

        causal_mask = self.causal_mask[:L, :L]

        all_logits: list[torch.Tensor] = []
        all_norms: list[float] = []
        all_hidden_norms_sq: list[torch.Tensor] = []  # differentiable for norm penalty
        gate_logits: list[torch.Tensor] = []

        for k in range(self.num_loops):
            # Apply all layers (one loop iteration)
            for layer in self.layers:
                h = layer(h, causal_mask=causal_mask)

            # Measure norm BEFORE any normalization (detached for logging)
            with torch.no_grad():
                norm_k = h.float().norm(dim=-1).mean().item()
                all_norms.append(norm_k)

            # Differentiable squared norm for optional penalty
            all_hidden_norms_sq.append(h.pow(2).mean())

            # Decode at this iteration (BEFORE inter-loop norm to avoid double normalization)
            # If decode_norm_final_only: raw readout at intermediate loops (loss sees norms),
            # decode_norm only at the final loop (good logit scaling for output)
            is_final = (k == self.num_loops - 1)
            if self.use_decode_norm and (is_final or not self.decode_norm_final_only):
                if self.per_loop_alpha:
                    decode_module = self.decode_norm[k]
                else:
                    decode_module = self.decode_norm
                logits_k = self.lm_head(decode_module(h))
            else:
                logits_k = self.lm_head(h)
            all_logits.append(logits_k)

            # Exit gate: scalar per token per loop
            gate_k = self.exit_gate(h).squeeze(-1)  # [B, L]
            gate_logits.append(gate_k)

            # Apply inter-loop norm for next iteration (Ouro-style)
            if self.inter_loop_norm:
                h = self.loop_norm(h)

            # Spectral damping (OpenMythos-style): scale h by α ∈ (0,1)
            if self.use_spectral_damping and k < self.num_loops - 1:
                alpha = torch.sigmoid(self.raw_alpha)
                h = h * alpha

        # Compute exit probability distribution (Ouro-style)
        # p_k = λ_k * remaining_prob; last step gets all remaining
        exit_pdf: list[torch.Tensor] = []
        remaining = torch.ones_like(gate_logits[0])  # [B, L]
        for k, g in enumerate(gate_logits):
            lam = torch.sigmoid(g)
            if k < len(gate_logits) - 1:
                p_k = lam * remaining
                remaining = remaining * (1.0 - lam)
            else:
                p_k = remaining  # last loop gets all remaining mass
            exit_pdf.append(p_k)

        return {
            "all_logits": all_logits,
            "all_norms": all_norms,
            "all_hidden_norms_sq": all_hidden_norms_sq,  # differentiable for norm penalty
            "final_logits": all_logits[-1],
            "exit_pdf": exit_pdf,         # list of K tensors [B, L]
            "gate_logits": gate_logits,   # list of K tensors [B, L]
        }


def build_ar_looplm_from_config(
    config: dict,
    *,
    vocab_size: int,
    seq_len: int | None = None,
    dropout_override: float | None = None,
) -> AutoregressiveLoopLM:
    """Single source-of-truth builder. Reads every architecture knob from the
    nested config dict so train/eval scripts can never silently drop fields.

    Required keys: config["model"]["d_model","num_heads","num_layers",
    "ff_multiplier","num_loops"], config["data"]["seq_len"] (or seq_len arg).

    All optional sub-keys (alpha_readout, taper_norm, model.* extras) fall back
    to module-level defaults that match what training would have used.
    """
    model_config = config["model"]
    if seq_len is None:
        seq_len = config["data"]["seq_len"]

    alpha_config = config.get("alpha_readout", {})
    taper_config = config.get("taper_norm", {})

    return AutoregressiveLoopLM(
        vocab_size=vocab_size,
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        ff_multiplier=model_config["ff_multiplier"],
        dropout=(dropout_override if dropout_override is not None
                 else model_config.get("dropout", 0.1)),
        max_seq_len=seq_len,
        num_loops=model_config["num_loops"],
        inter_loop_norm=model_config.get("inter_loop_norm", True),
        use_decode_norm=model_config.get("use_decode_norm", True),
        decode_norm_final_only=model_config.get("decode_norm_final_only", False),
        block_norm_type=model_config.get("block_norm_type", "rmsnorm"),
        loop_norm_type=model_config.get("loop_norm_type", "rmsnorm"),
        decode_norm_type=model_config.get("decode_norm_type", "rmsnorm"),
        taper_ema_decay=taper_config.get("ema_decay", 0.99),
        alpha_init=alpha_config.get("init_alpha", 0.5),
        alpha_fixed=alpha_config.get("fixed_alpha", None),
        alpha_ema_decay=alpha_config.get("ema_decay", 0.99),
        alpha_s_ref_mode=alpha_config.get("s_ref_mode", "ema"),
        alpha_init_log_s_ref=alpha_config.get("init_log_s_ref", 0.0),
        alpha_min=alpha_config.get("alpha_min", 0.0),
        alpha_max=alpha_config.get("alpha_max", 1.0),
        per_loop_alpha=alpha_config.get("per_loop_alpha", False),
        use_spectral_damping=model_config.get("use_spectral_damping", False),
        tie_weights=model_config.get("tie_weights", True),
    )
