"""Train an autoregressive LoopLM on WikiText-103.

NO denoising, NO corruption, NO teacher direction.
Just next-token prediction with shared layers applied K times.

Two supervision modes:
- per-step: CE loss averaged across all K loop iterations
- terminal: CE loss only at the final loop iteration

Usage:
    uv run python src/train_ar_looplm.py --config configs/ar_looplm_terminal.yaml
"""

from __future__ import annotations

import argparse
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml


def load_config(path: str) -> dict:
    config_path = Path(path)
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    base = config.pop("base_config", None)
    if base:
        candidate = config_path.parent / base
        base_path = candidate.resolve() if candidate.exists() else Path(base).resolve()
        base_cfg = load_config(str(base_path))
        # merge
        def merge(b, o):
            m = dict(b)
            for k, v in o.items():
                if k in m and isinstance(m[k], dict) and isinstance(v, dict):
                    m[k] = merge(m[k], v)
                else:
                    m[k] = v
            return m
        config = merge(base_cfg, config)
    return config


def build_dataset(config: dict) -> tuple[Any, Any, Any]:
    """Build WikiText-103 train/val datasets as token ID sequences."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data.dataset import load_hf_text_split, join_texts, RegexTokenizer

    data_config = config["data"]
    train_texts = load_hf_text_split(data_config, data_config["hf_train_split"])
    val_texts = load_hf_text_split(data_config, data_config["hf_val_split"])

    pattern = data_config.get("pattern", r"\s+|[A-Za-z0-9_]+|[^\w\s]")
    tokenizer = RegexTokenizer.from_texts(
        train_texts + val_texts,
        pattern,
        max_vocab_size=data_config.get("max_vocab_size", 20000),
        min_token_freq=data_config.get("min_token_freq", 2),
    )

    join_with = data_config.get("join_with", "\n\n")
    train_ids = tokenizer.encode(join_texts(train_texts, join_with))
    val_ids = tokenizer.encode(join_texts(val_texts, join_with))

    return tokenizer, train_ids, val_ids


class TextChunkDataset(torch.utils.data.Dataset):
    """Simple dataset that yields chunks of token IDs for autoregressive training."""

    def __init__(self, token_ids: list[int], seq_len: int, stride: int) -> None:
        self.token_ids = token_ids
        self.seq_len = seq_len
        # We need seq_len + 1 tokens per chunk (input + shifted target)
        max_start = len(token_ids) - seq_len - 1
        self.starts = list(range(0, max(1, max_start + 1), stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.starts[idx]
        chunk = self.token_ids[start: start + self.seq_len + 1]
        return {
            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
            "labels": torch.tensor(chunk[1:], dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def apply_overrides(config: dict, overrides: list[str] | None) -> dict:
    """Apply --set key=value overrides to config."""
    if not overrides:
        return config
    import copy
    config = copy.deepcopy(config)
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"invalid override: {spec}")
        key, raw_val = spec.split("=", 1)
        val = yaml.safe_load(raw_val)
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val
    return config


def _schedule_steps(
    taper_config: dict,
    *,
    total_steps: int,
    step_key: str,
    frac_key: str,
    default_frac: float,
) -> int:
    if step_key in taper_config:
        return max(0, int(taper_config[step_key]))
    frac = float(taper_config.get(frac_key, default_frac))
    return max(0, int(round(total_steps * frac)))


def compute_taper_gate(global_step: int, total_steps: int, taper_config: dict) -> tuple[float, int]:
    """Cosine gate schedule: warmup at 1, then decay to 0."""
    warmup_steps = _schedule_steps(
        taper_config,
        total_steps=total_steps,
        step_key="gate_warmup_steps",
        frac_key="gate_warmup_frac",
        default_frac=0.05,
    )
    default_decay_frac = max(0.0, 1.0 - warmup_steps / max(1, total_steps))
    decay_steps = _schedule_steps(
        taper_config,
        total_steps=total_steps,
        step_key="gate_decay_steps",
        frac_key="gate_decay_frac",
        default_frac=default_decay_frac,
    )
    if global_step < warmup_steps:
        return 1.0, warmup_steps
    if decay_steps <= 0:
        return 0.0, warmup_steps
    progress = min(1.0, max(0.0, (global_step - warmup_steps) / decay_steps))
    gate = 0.5 * (1.0 + math.cos(math.pi * progress))
    return gate, warmup_steps


def update_taper_norm_state(model: torch.nn.Module, global_step: int, total_steps: int, config: dict) -> dict:
    if not hasattr(model, "taper_summary"):
        return {"count": 0, "calibrated": 0, "gate": 1.0, "scale_coeff": 1.0}

    summary = model.taper_summary()
    if int(summary["count"]) == 0:
        return summary

    taper_config = config.get("taper_norm", {})
    gate, warmup_steps = compute_taper_gate(global_step, total_steps, taper_config)
    if global_step >= warmup_steps:
        model.calibrate_taper_norms()
    model.set_taper_gate(gate)
    return model.taper_summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=None,
                        help="Override config: --set train.seed=101")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.overrides)
    train_config = config["train"]
    model_config = config["model"]

    # Seed
    seed = train_config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    device_str = train_config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    precision = train_config.get("precision", "fp32")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Data
    print("Loading data...")
    tokenizer, train_ids, val_ids = build_dataset(config)
    vocab_size = len(tokenizer)
    seq_len = config["data"]["seq_len"]
    stride = config["data"].get("stride", seq_len)

    train_dataset = TextChunkDataset(train_ids, seq_len, stride)
    val_dataset = TextChunkDataset(val_ids, seq_len, stride)
    print(f"vocab={vocab_size} train_chunks={len(train_dataset)} val_chunks={len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"],
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=train_config.get("num_workers", 0))
    val_loader = DataLoader(val_dataset, batch_size=train_config["batch_size"],
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=train_config.get("num_workers", 0))

    # Model
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from models.ar_looplm import AutoregressiveLoopLM, build_ar_looplm_from_config

    model = build_ar_looplm_from_config(config, vocab_size=vocab_size, seq_len=seq_len).to(device)
    _ = AutoregressiveLoopLM  # keep import alive for any external callers

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    taper_summary = update_taper_norm_state(model, 0, max(1, len(train_loader) * train_config["epochs"]), config)
    print(
        f"device={device} params={num_params:,} loops={model.num_loops} "
        f"taper_norms={taper_summary['count']}"
    )

    # Training setup
    supervision = config["supervision"]["mode"]  # "per-step" or "terminal"
    print(f"supervision={supervision}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=train_config["lr"],
                                  weight_decay=train_config.get("weight_decay", 0.01))
    grad_clip = train_config.get("grad_clip", 1.0)
    max_steps_per_epoch = int(train_config.get("max_steps_per_epoch", 0) or 0)
    max_val_batches = int(train_config.get("max_val_batches", 0) or 0)
    steps_per_epoch = min(len(train_loader), max_steps_per_epoch) if max_steps_per_epoch > 0 else len(train_loader)
    total_train_steps = max(1, steps_per_epoch * train_config["epochs"])
    global_step = 0
    completed_epochs = 0

    use_autocast = device.type == "cuda" and precision in ("bf16", "fp16")
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16 if precision == "fp16" else None

    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_path = args.resume or train_config.get("resume")
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        global_step = int(checkpoint.get("global_step", 0))
        completed_epochs = int(checkpoint.get("completed_epochs", checkpoint.get("epoch", 0)))
        update_taper_norm_state(model, global_step, total_train_steps, config)
        print(f"resumed={resume_path} global_step={global_step} completed_epochs={completed_epochs}")

    # Save resolved config
    with (output_dir / "resolved_config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    def save_checkpoint(epoch: int, *, completed_epoch: bool) -> None:
        done_epochs = epoch if completed_epoch else max(0, epoch - 1)
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "tokenizer": tokenizer.to_state(),
            "epoch": epoch,
            "completed_epochs": done_epochs,
            "global_step": global_step,
            "total_train_steps": total_train_steps,
        }, output_dir / "last.pt")

    # Training loop
    for epoch in range(completed_epochs + 1, train_config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        norm_accum = [0.0] * model.num_loops
        norm_count = 0

        iterator = tqdm(train_loader, desc=f"train e{epoch}", disable=False, total=steps_per_epoch)
        for batch_idx, batch in enumerate(iterator):
            if max_steps_per_epoch > 0 and batch_idx >= max_steps_per_epoch:
                break
            taper_summary = update_taper_norm_state(model, global_step, total_train_steps, config)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            ctx = torch.autocast(device_type=device.type, dtype=autocast_dtype) if use_autocast else nullcontext()
            with ctx:
                outputs = model(input_ids)
                all_logits = outputs["all_logits"]

                if supervision == "per-step":
                    # Average CE across all loop iterations
                    loss = torch.tensor(0.0, device=device)
                    for logits_k in all_logits:
                        loss = loss + F.cross_entropy(
                            logits_k.reshape(-1, vocab_size),
                            labels.reshape(-1),
                            ignore_index=tokenizer.pad_id,
                        )
                    loss = loss / len(all_logits)
                elif supervision == "terminal":
                    # CE only at the final loop iteration
                    loss = F.cross_entropy(
                        all_logits[-1].reshape(-1, vocab_size),
                        labels.reshape(-1),
                        ignore_index=tokenizer.pad_id,
                    )
                elif supervision == "adaptive":
                    # Ouro-style: exit-probability-weighted CE + entropy regularization
                    exit_pdf = outputs["exit_pdf"]  # list of K tensors [B, L]
                    entropy_weight = config["supervision"].get("entropy_weight", 0.1)

                    # Weighted CE: Σ_k p_k * CE_k (per-token weighting)
                    loss = torch.tensor(0.0, device=device)
                    flat_labels = labels.reshape(-1)
                    for k, (logits_k, p_k) in enumerate(zip(all_logits, exit_pdf)):
                        # Per-token CE (no reduction)
                        ce_k = F.cross_entropy(
                            logits_k.reshape(-1, vocab_size),
                            flat_labels,
                            ignore_index=tokenizer.pad_id,
                            reduction="none",
                        )  # [B*L]
                        # Weight by exit probability
                        p_flat = p_k.reshape(-1)
                        loss = loss + (p_flat * ce_k).mean()

                    # Entropy regularization on exit distribution
                    # H(p) = -Σ_k p_k log p_k, averaged over tokens
                    # Maximize entropy (add negative entropy as loss)
                    stacked_p = torch.stack(exit_pdf, dim=-1)  # [B, L, K]
                    entropy = -(stacked_p * torch.log(stacked_p.clamp(min=1e-8))).sum(dim=-1)  # [B, L]
                    entropy_loss = -entropy.mean()  # negative because we MAXIMIZE entropy
                    loss = loss + entropy_weight * entropy_loss
                else:
                    raise ValueError(f"unknown supervision mode: {supervision}")

                # Optional per-loop norm penalty: λ * mean(||H_k||²) across loops
                norm_penalty_weight = config.get("supervision", {}).get("norm_penalty_weight", 0.0)
                if norm_penalty_weight > 0.0 and "all_hidden_norms_sq" in outputs:
                    norm_penalty = torch.stack(outputs["all_hidden_norms_sq"]).mean()
                    loss = loss + norm_penalty_weight * norm_penalty

                scale_anchor_weight = config.get("supervision", {}).get("scale_anchor_weight", 0.0)
                if scale_anchor_weight > 0.0 and "all_hidden_norms_sq" in outputs:
                    scale_target = config.get("supervision", {}).get("scale_anchor_target", 1.0)
                    hidden_rms = torch.stack(outputs["all_hidden_norms_sq"]).mean().sqrt()
                    scale_anchor = (hidden_rms - scale_target) ** 2
                    loss = loss + scale_anchor_weight * scale_anchor

            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), grad_clip).item()
            optimizer.step()
            global_step += 1

            bs = input_ids.size(0) * input_ids.size(1)
            total_loss += loss.item() * bs
            total_tokens += bs

            for k, n in enumerate(outputs["all_norms"]):
                norm_accum[k] += n
            norm_count += 1

            iterator.set_description(
                f"train e{epoch} loss={total_loss/total_tokens:.4f} taper_g={taper_summary['gate']:.3f}"
            )

            save_every_steps = int(train_config.get("save_every_steps", 0) or 0)
            if save_every_steps > 0 and global_step % save_every_steps == 0:
                save_checkpoint(epoch, completed_epoch=False)

        train_loss = total_loss / total_tokens
        avg_norms = [n / norm_count for n in norm_accum]

        # Validation
        taper_summary = update_taper_norm_state(model, global_step, total_train_steps, config)
        model.eval()
        val_loss_sum = 0.0
        val_tokens = 0
        val_norms = [0.0] * model.num_loops
        val_norm_count = 0

        with torch.no_grad():
            val_total = min(len(val_loader), max_val_batches) if max_val_batches > 0 else len(val_loader)
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"val e{epoch}", disable=False, total=val_total)):
                if max_val_batches > 0 and batch_idx >= max_val_batches:
                    break
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                ctx = torch.autocast(device_type=device.type, dtype=autocast_dtype) if use_autocast else nullcontext()
                with ctx:
                    outputs = model(input_ids)

                # Compute val loss matching the supervision mode
                if supervision == "adaptive":
                    vloss = torch.tensor(0.0, device=device)
                    flat_labels = labels.reshape(-1)
                    for logits_k, p_k in zip(outputs["all_logits"], outputs["exit_pdf"]):
                        ce_k = F.cross_entropy(logits_k.reshape(-1, vocab_size), flat_labels,
                                               ignore_index=tokenizer.pad_id, reduction="none")
                        vloss = vloss + (p_k.reshape(-1) * ce_k).mean()
                elif supervision == "per-step":
                    vloss = sum(F.cross_entropy(l.reshape(-1, vocab_size), labels.reshape(-1),
                                               ignore_index=tokenizer.pad_id)
                                for l in outputs["all_logits"]) / len(outputs["all_logits"])
                else:
                    vloss = F.cross_entropy(
                        outputs["final_logits"].reshape(-1, vocab_size),
                        labels.reshape(-1),
                        ignore_index=tokenizer.pad_id,
                    )
                bs = input_ids.size(0) * input_ids.size(1)
                val_loss_sum += vloss.item() * bs
                val_tokens += bs

                for k, n in enumerate(outputs["all_norms"]):
                    val_norms[k] += n
                val_norm_count += 1

        val_loss = val_loss_sum / val_tokens
        val_avg_norms = [n / val_norm_count for n in val_norms]

        # Print summary
        norm_str = " ".join(f"||H_{k}||={n:.1f}" for k, n in enumerate(avg_norms))
        val_norm_str = " ".join(f"||H_{k}||={n:.1f}" for k, n in enumerate(val_avg_norms))
        alpha_summary = model.alpha_summary() if hasattr(model, "alpha_summary") else {"count": 0}
        alpha_str = ""
        if alpha_summary.get("count", 0) > 0:
            per_loop_alpha = alpha_summary.get("per_loop_alpha", [])
            if len(per_loop_alpha) > 1:
                per_loop_str = "[" + ",".join(f"{a:.3f}" for a in per_loop_alpha) + "]"
            else:
                per_loop_str = ""
            alpha_str = (
                f"alpha={alpha_summary['alpha']:.4f} "
                f"{('per_loop_alpha=' + per_loop_str + ' ') if per_loop_str else ''}"
                f"log_s_ref={alpha_summary['log_s_ref']:.4f} "
                f"s_ref_mode={alpha_summary.get('s_ref_mode', 'unknown')} "
            )
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"train_grad_norm={grad_norm:.4f} "
            f"taper_gate={taper_summary['gate']:.4f} "
            f"taper_calibrated={taper_summary['calibrated']}/{taper_summary['count']} "
            f"taper_scale={taper_summary['scale_coeff']:.4f} "
            f"{alpha_str}"
            f"train_norms=[{norm_str}] "
            f"val_norms=[{val_norm_str}]"
        )

        # Save checkpoint
        save_checkpoint(epoch, completed_epoch=True)

    print("Done.")


if __name__ == "__main__":
    main()
