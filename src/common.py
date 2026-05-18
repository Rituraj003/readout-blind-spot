from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from models.refiner import RefinementTransformer
from models.rollout import RefinementRollout


def merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_cli_overrides(config: dict[str, Any], override_specs: list[str] | None) -> dict[str, Any]:
    if not override_specs:
        return config

    merged = dict(config)
    for spec in override_specs:
        if "=" not in spec:
            raise ValueError(f"invalid override `{spec}`; expected dotted.path=value")
        dotted_key, raw_value = spec.split("=", 1)
        keys = [part.strip() for part in dotted_key.split(".") if part.strip()]
        if not keys:
            raise ValueError(f"invalid override `{spec}`; empty key path")
        value = yaml.safe_load(raw_value)
        nested: dict[str, Any] = value
        for key in reversed(keys):
            nested = {key: nested}
        merged = merge_dicts(merged, nested)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base_config = config.pop("base_config", None)
    if base_config is None:
        return config
    candidate = config_path.parent / base_config
    base_path = candidate.resolve() if candidate.exists() else Path(base_config).resolve()
    base = load_config(base_path)
    return merge_dicts(base, config)


def save_config(path: str | Path, config: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def build_model_and_rollout(
    config: dict[str, Any],
    vocab_size: int,
    pad_id: int,
) -> tuple[RefinementTransformer, RefinementRollout]:
    data_config = config["data"]
    model_config = config["model"]
    rollout_config = config["rollout"]

    arch = model_config.get("arch", "transformer")
    if arch == "mlp":
        from models.mlp_refiner import RefinementMLP
        model_class = RefinementMLP
    elif arch == "looplm":
        from models.loop_refiner import RefinementLoopLM
        model_class = RefinementLoopLM
    else:
        model_class = RefinementTransformer

    model = model_class(
        vocab_size=vocab_size,
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        ff_multiplier=model_config["ff_multiplier"],
        dropout=model_config["dropout"],
        prefix_length=data_config["prefix_length"],
        suffix_length=data_config["suffix_length"],
        pad_id=pad_id,
        use_decode_norm=bool(model_config.get("use_decode_norm", True)),
        decode_norm_type=str(model_config.get("decode_norm_type", "layernorm")),
    )
    rollout = RefinementRollout(
        model=model,
        num_steps=rollout_config["steps"],
        min_noise=rollout_config["min_noise"],
        max_noise=rollout_config["max_noise"],
        noise_distribution=rollout_config.get("noise_distribution", "uniform"),
        beta_alpha=rollout_config.get("beta_alpha", 1.0),
        beta_beta=rollout_config.get("beta_beta", 1.0),
        point_mass_prob=rollout_config.get("point_mass_prob", 0.0),
        point_mass_value=rollout_config.get("point_mass_value", 1.0),
        pad_id=pad_id,
    )
    return model, rollout


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    tokenizer_state: dict[str, Any],
    epoch: int,
    global_step: int,
    best_val_loss: float,
    best_pure_noise_acc: float | None = None,
    scaler_state: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "tokenizer": tokenizer_state,
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "best_pure_noise_acc": best_pure_noise_acc,
            "scaler_state": scaler_state,
        },
        destination,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)


def prepare_prefix_batch(
    prefix_texts: list[str],
    tokenizer: Any,
    prefix_length: int,
    device: torch.device,
) -> torch.Tensor:
    rows = [
        tokenizer.fit_length(
            tokenizer.encode(text),
            prefix_length,
            pad_side="left",
            truncate_side="left",
        )
        for text in prefix_texts
    ]
    return torch.tensor(rows, dtype=torch.long, device=device)


def sample_token_ids(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if temperature <= 0.0:
        return logits.argmax(dim=-1)

    scaled = logits / temperature
    if top_k > 0 and top_k < scaled.size(-1):
        values, indices = torch.topk(scaled, k=top_k, dim=-1)
        filtered = torch.full_like(scaled, float("-inf"))
        filtered.scatter_(-1, indices, values)
        scaled = filtered
    probabilities = F.softmax(scaled, dim=-1)
    flat_probabilities = probabilities.view(-1, probabilities.size(-1))
    sampled = torch.multinomial(flat_probabilities, num_samples=1)
    return sampled.view(*probabilities.shape[:-1])
