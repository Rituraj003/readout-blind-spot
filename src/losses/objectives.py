from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    weights = valid_mask.float()
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def clause_satisfaction_loss(
    final_logits: torch.Tensor,
    clause_variables: torch.Tensor,
    clause_polarities: torch.Tensor,
    bit_1_token_id: int,
    bit_0_token_id: int,
) -> torch.Tensor:
    """Differentiable clause-satisfaction loss (Phase 2).

    Computes -log P(clause satisfied) averaged over all clauses and the batch.
    Permits the model to find *any* satisfying assignment, not just the
    canonical one.

    Args:
        final_logits: [B, n, V] logits from the terminal state.
        clause_variables: [B, m, k] variable indices (0-based).
        clause_polarities: [B, m, k] polarities (1=positive, 0=negated).
        bit_1_token_id: token ID for BIT_1 in the vocabulary.
        bit_0_token_id: token ID for BIT_0 in the vocabulary.

    Returns:
        Scalar loss.
    """
    # Extract P(bit=1) per variable from the two relevant logit positions.
    bit_logits = torch.stack(
        [final_logits[..., bit_0_token_id], final_logits[..., bit_1_token_id]],
        dim=-1,
    )  # [B, n, 2]
    p_bit1 = F.softmax(bit_logits, dim=-1)[..., 1]  # [B, n]

    # Gather per-literal probabilities: [B, m, k]
    B, m, k = clause_variables.shape
    p_var = torch.gather(
        p_bit1.unsqueeze(1).expand(-1, m, -1),
        dim=2,
        index=clause_variables,
    )  # [B, m, k]

    # Apply polarity: P(literal true) = p if positive, 1-p if negated.
    p_lit_true = torch.where(
        clause_polarities.bool(),
        p_var,
        1.0 - p_var,
    )  # [B, m, k]

    # P(clause unsatisfied) = prod(1 - P(literal true)) over k literals.
    # Work in log-space for stability.
    log_p_lit_false = torch.log1p(-p_lit_true.clamp(min=1e-7, max=1.0 - 1e-7))  # [B, m, k]
    log_p_clause_unsat = log_p_lit_false.sum(dim=-1)  # [B, m]

    # -log P(clause satisfied) = -log(1 - exp(log_p_clause_unsat))
    # Use log1p(-exp(...)) for numerical stability.
    p_clause_unsat = torch.exp(log_p_clause_unsat).clamp(max=1.0 - 1e-7)
    neg_log_p_clause_sat = -torch.log1p(-p_clause_unsat)  # [B, m]

    return neg_log_p_clause_sat.mean()


def compute_losses(
    *,
    outputs: dict[str, Any],
    suffix_tokens: torch.Tensor,
    pad_id: int,
    mode: str,
    endpoint_weight: float,
    local_weight: float,
    calibration_weight: float,
    detach_local_state: bool,
    detach_clean_target: bool,
    normalize_local_loss: bool = False,
    clause_variables: torch.Tensor | None = None,
    clause_polarities: torch.Tensor | None = None,
    clause_sat_weight: float = 0.0,
    bit_1_token_id: int = -1,
    bit_0_token_id: int = -1,
    anchor_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    final_logits = outputs["final_logits"]
    suffix_padding_mask = outputs["suffix_padding_mask"]
    valid_mask = (~suffix_padding_mask).float()

    endpoint_loss = F.cross_entropy(
        final_logits.reshape(-1, final_logits.size(-1)),
        suffix_tokens.reshape(-1),
        ignore_index=pad_id,
    )

    clean_suffix_latents = outputs["clean_suffix_latents"]
    if clean_suffix_latents is None:
        raise ValueError("clean suffix latents are required for loss computation")
    clean_logits = outputs["clean_logits"]
    if clean_logits is None:
        raise ValueError("clean logits are required for calibration loss")

    calibration_loss = F.cross_entropy(
        clean_logits.reshape(-1, clean_logits.size(-1)),
        suffix_tokens.reshape(-1),
        ignore_index=pad_id,
    )

    states = outputs["states"]
    pred_updates = outputs["pred_updates"]
    num_steps = pred_updates.size(0)
    teacher_clean_suffix_latents = clean_suffix_latents.detach() if detach_clean_target else clean_suffix_latents
    local_terms: list[torch.Tensor] = []
    for step in range(num_steps):
        remaining_steps = float(num_steps - step)
        teacher_state = states[step].detach() if detach_local_state else states[step]
        target_update = (teacher_clean_suffix_latents - teacher_state) / remaining_steps
        squared_error = (pred_updates[step] - target_update).pow(2).mean(dim=-1)
        if normalize_local_loss:
            target_scale = target_update.detach().pow(2).mean(dim=-1).clamp(min=1.0)
            squared_error = squared_error / target_scale
        local_terms.append(_masked_mean(squared_error, valid_mask))
    local_loss = torch.stack(local_terms).mean()

    # Clause-satisfaction loss (Phase 2, only when clause data is provided).
    clause_sat_loss = torch.tensor(0.0, device=final_logits.device)
    if clause_sat_weight > 0.0 and clause_variables is not None and clause_polarities is not None:
        if bit_0_token_id < 0 or bit_1_token_id < 0:
            raise ValueError(
                "bit_0_token_id and bit_1_token_id must be set (>= 0) when clause_sat_weight > 0"
            )
        clause_sat_loss = clause_satisfaction_loss(
            final_logits=final_logits,
            clause_variables=clause_variables,
            clause_polarities=clause_polarities,
            bit_1_token_id=bit_1_token_id,
            bit_0_token_id=bit_0_token_id,
        )

    # Latent anchor loss: ||H_K - H*||² / (L*d), penalizes norm drift.
    # Detach the target so gradients don't flow into the embedding table.
    anchor_loss = _masked_mean(
        (outputs["final_state"] - teacher_clean_suffix_latents).pow(2).mean(dim=-1),
        valid_mask,
    )

    if mode == "local-only":
        total_loss = local_weight * local_loss
    elif mode == "endpoint-only":
        total_loss = endpoint_weight * endpoint_loss
    elif mode == "local+endpoint":
        total_loss = endpoint_weight * endpoint_loss + local_weight * local_loss
    else:
        raise ValueError(f"unknown loss mode: {mode}")
    total_loss = total_loss + calibration_weight * calibration_loss
    if anchor_weight > 0.0:
        total_loss = total_loss + anchor_weight * anchor_loss
    if clause_sat_weight > 0.0:
        total_loss = total_loss + clause_sat_weight * clause_sat_loss

    with torch.no_grad():
        predictions = final_logits.argmax(dim=-1)
        correct = predictions.eq(suffix_tokens) & ~suffix_padding_mask
        token_accuracy = correct.sum().float() / valid_mask.sum().clamp_min(1.0)
        clean_predictions = clean_logits.argmax(dim=-1)
        clean_correct = clean_predictions.eq(suffix_tokens) & ~suffix_padding_mask
        clean_token_accuracy = clean_correct.sum().float() / valid_mask.sum().clamp_min(1.0)
        final_latent_error = (outputs["final_state"] - clean_suffix_latents).pow(2).mean(dim=-1)
        latent_mse = _masked_mean(final_latent_error, valid_mask)

    return {
        "total_loss": total_loss,
        "endpoint_loss": endpoint_loss.detach(),
        "local_loss": local_loss.detach(),
        "calibration_loss": calibration_loss.detach(),
        "clause_sat_loss": clause_sat_loss.detach(),
        "anchor_loss": anchor_loss.detach(),
        "token_accuracy": token_accuracy,
        "clean_token_accuracy": clean_token_accuracy,
        "latent_mse": latent_mse,
    }
