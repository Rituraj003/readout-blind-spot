from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.refiner import RefinementTransformer


class RefinementRollout(nn.Module):
    def __init__(
        self,
        *,
        model: RefinementTransformer,
        num_steps: int,
        min_noise: float,
        max_noise: float,
        noise_distribution: str,
        beta_alpha: float,
        beta_beta: float,
        point_mass_prob: float,
        point_mass_value: float,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.noise_distribution = noise_distribution
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.point_mass_prob = point_mass_prob
        self.point_mass_value = point_mass_value
        self.pad_id = pad_id

    def sample_noise_levels(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.min_noise == self.max_noise:
            base = torch.full((batch_size,), self.min_noise, device=device)
        elif self.noise_distribution == "uniform":
            base = torch.rand(batch_size, device=device)
            base = self.min_noise + (self.max_noise - self.min_noise) * base
        elif self.noise_distribution == "beta":
            # MPS does not implement the internal Dirichlet sampler used by
            # torch.distributions.Beta, so sample on CPU and move the result.
            distribution = torch.distributions.Beta(
                torch.tensor(self.beta_alpha, device="cpu"),
                torch.tensor(self.beta_beta, device="cpu"),
            )
            base = distribution.sample((batch_size,)).to(device=device)
            base = self.min_noise + (self.max_noise - self.min_noise) * base
        else:
            raise ValueError(f"unknown noise distribution: {self.noise_distribution}")

        if self.point_mass_prob > 0.0:
            mask = torch.rand(batch_size, device=device) < self.point_mass_prob
            point_value = torch.full_like(base, self.point_mass_value)
            base = torch.where(mask, point_value, base)

        return base.clamp_(0.0, 1.0)

    def make_fixed_noise_levels(self, batch_size: int, device: torch.device, noise_level: float) -> torch.Tensor:
        return torch.full((batch_size,), float(noise_level), device=device)

    def corrupt_suffix(
        self,
        clean_suffix_latents: torch.Tensor,
        noise_levels: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn_like(clean_suffix_latents)
        alpha = (1.0 - noise_levels).view(-1, 1, 1)
        sigma = noise_levels.view(-1, 1, 1)
        return alpha * clean_suffix_latents + sigma * noise

    def rollout(
        self,
        *,
        prefix_tokens: torch.Tensor,
        suffix_tokens: torch.Tensor | None = None,
        initial_suffix_latents: torch.Tensor | None = None,
        noise_levels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if suffix_tokens is None and initial_suffix_latents is None:
            raise ValueError("suffix_tokens or initial_suffix_latents must be provided")

        batch_size = prefix_tokens.size(0)
        device = prefix_tokens.device
        prefix_padding_mask = prefix_tokens.eq(self.pad_id)

        clean_suffix_latents = None
        clean_logits = None
        suffix_padding_mask = None
        if suffix_tokens is not None:
            clean_suffix_latents = self.model.token_latents(suffix_tokens)
            clean_logits = self.model.decode(clean_suffix_latents)
            suffix_padding_mask = suffix_tokens.eq(self.pad_id)
        if initial_suffix_latents is None:
            if clean_suffix_latents is None:
                raise ValueError("clean_suffix_latents are required when initial latents are not provided")
            if noise_levels is None:
                noise_levels = self.sample_noise_levels(batch_size, device)
            initial_suffix_latents = self.corrupt_suffix(clean_suffix_latents, noise_levels)
        elif noise_levels is None:
            noise_levels = torch.ones(batch_size, device=device)

        if suffix_padding_mask is None:
            suffix_padding_mask = torch.zeros(
                batch_size,
                initial_suffix_latents.size(1),
                dtype=torch.bool,
                device=device,
            )

        current = initial_suffix_latents
        pre_update_states: list[torch.Tensor] = []
        pred_updates: list[torch.Tensor] = []
        denominator = max(self.num_steps - 1, 1)

        for step in range(self.num_steps):
            step_fraction = torch.full(
                (batch_size,),
                float(step) / denominator,
                device=device,
            )
            predicted_update = self.model(
                prefix_tokens=prefix_tokens,
                suffix_latents=current,
                step_fraction=step_fraction,
                noise_level=noise_levels,
                prefix_padding_mask=prefix_padding_mask,
                suffix_padding_mask=suffix_padding_mask,
            )
            pre_update_states.append(current)
            pred_updates.append(predicted_update)
            current = current + predicted_update

        return {
            "initial_suffix_latents": initial_suffix_latents,
            "clean_suffix_latents": clean_suffix_latents,
            "clean_logits": clean_logits,
            "noise_levels": noise_levels,
            "states": torch.stack(pre_update_states, dim=0),
            "pred_updates": torch.stack(pred_updates, dim=0),
            "final_state": current,
            "final_logits": self.model.decode(current),
            "suffix_padding_mask": suffix_padding_mask,
        }

    @torch.no_grad()
    def generate(self, prefix_tokens: torch.Tensor, noise_level: float = 1.0) -> dict[str, Any]:
        batch_size = prefix_tokens.size(0)
        initial_suffix_latents = torch.randn(
            batch_size,
            self.model.suffix_length,
            self.model.d_model,
            device=prefix_tokens.device,
        )
        noise_levels = self.make_fixed_noise_levels(batch_size, prefix_tokens.device, noise_level)
        return self.rollout(
            prefix_tokens=prefix_tokens,
            initial_suffix_latents=initial_suffix_latents,
            noise_levels=noise_levels,
        )
