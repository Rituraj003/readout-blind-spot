"""Benchmark actual sequence-level early exits for autoregressive LoopLMs.

Unlike `early_exit_curve.py`, this script does not merely simulate exits from a
full forward pass. After calibration, it runs later loops only for sequences
that have not exited yet, so the timing includes the real batching overhead of
dynamic depth.

The policy is intentionally simple: after each non-final loop, exit a sequence
when its mean validation confidence exceeds a calibrated threshold. The default
confidence is mean logit margin.

Example:
    uv run python src/eval/adaptive_halt_benchmark.py \
      --checkpoint perstep_raw=outputs/ar_looplm_150m_perstep_raw_s42/last.pt \
      --checkpoint final_norm=outputs/ar_looplm_150m_perstep_final_norm_s42/last.pt \
      --device cuda --batch-size 16
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_ROOT = _PROJECT_ROOT / ".cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


POLICIES = ("logit_margin", "max_prob", "confidence_entropy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Actual adaptive halt benchmark.")
    parser.add_argument("--checkpoint", action="append", required=True, help="label=path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--calib-batches", type=int, default=100)
    parser.add_argument("--bench-batches", type=int, default=100)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--num-thresholds", type=int, default=51)
    parser.add_argument("--target-rel-ppl", type=float, default=1.01)
    parser.add_argument("--policy", choices=POLICIES, default="logit_margin")
    parser.add_argument("--output-dir", default="outputs/adaptive_halt_benchmark")
    return parser.parse_args()


def parse_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(spec)
    return path.parent.name, path


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_model_and_val_data(ckpt_path: Path, device: torch.device):
    from data.dataset import RegexTokenizer, join_texts, load_hf_text_split
    from models.ar_looplm import AutoregressiveLoopLM
    from train_ar_looplm import TextChunkDataset, build_dataset, collate_fn

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    tok_state = ckpt.get("tokenizer")
    if tok_state and tok_state.get("kind") == "regex":
        tokenizer = RegexTokenizer.from_state(tok_state)
        data_config = config["data"]
        val_texts = load_hf_text_split(data_config, data_config["hf_val_split"])
        val_ids = tokenizer.encode(join_texts(val_texts, data_config.get("join_with", "\n\n")))
    else:
        tokenizer, _, val_ids = build_dataset(config)

    from models.ar_looplm import build_ar_looplm_from_config
    model = build_ar_looplm_from_config(
        config,
        vocab_size=len(tokenizer),
        dropout_override=0.0,
    ).to(device).eval()
    # Strict load: arch must match training config (alpha settings included).
    model.load_state_dict(ckpt["model_state"], strict=True)

    seq_len = config["data"]["seq_len"]
    return model, tokenizer, TextChunkDataset(val_ids, seq_len, seq_len), collate_fn, config


def decode_logits(model, h: torch.Tensor, loop_idx: int, num_loops: int) -> torch.Tensor:
    is_final = loop_idx == num_loops - 1
    if model.use_decode_norm and (is_final or not model.decode_norm_final_only):
        return model.lm_head(model.decode_norm(h))
    return model.lm_head(h)


def score_logits(logits: torch.Tensor, valid: torch.Tensor, policy: str, vocab_size: int) -> torch.Tensor:
    logits_f = logits.float()
    top2 = torch.topk(logits_f, k=2, dim=-1).values
    if policy == "logit_margin":
        token_score = top2[..., 0] - top2[..., 1]
    elif policy == "max_prob":
        token_score = torch.exp(top2[..., 0] - torch.logsumexp(logits_f, dim=-1))
    elif policy == "confidence_entropy":
        log_probs = F.log_softmax(logits_f, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        token_score = 1.0 - entropy / float(np.log(vocab_size))
    else:
        raise ValueError(policy)
    valid_f = valid.float()
    return (token_score * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1)


def ce_per_sequence(logits: torch.Tensor, labels: torch.Tensor, pad_id: int) -> torch.Tensor:
    ce = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_id,
        reduction="none",
    )
    return ce.view_as(labels).sum(dim=1)


def init_hidden(model, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=input_ids.device)
    h = model.token_embedding(input_ids) + model.position_embedding(positions)
    h = model.input_dropout(h)
    return h, model.causal_mask[:seq_len, :seq_len]


def run_loop_step(model, h: torch.Tensor, causal_mask: torch.Tensor, loop_idx: int) -> torch.Tensor:
    for layer in model.layers:
        h = layer(h, causal_mask=causal_mask)
    return h


def apply_recurrence_postprocess(model, h: torch.Tensor, loop_idx: int, num_loops: int) -> torch.Tensor:
    if model.inter_loop_norm:
        h = model.loop_norm(h)
    if model.use_spectral_damping and loop_idx < num_loops - 1:
        h = h * torch.sigmoid(model.raw_alpha)
    return h


def calibrate_threshold(
    model,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
    *,
    policy: str,
    max_batches: int,
    num_thresholds: int,
    target_rel_ppl: float,
) -> dict[str, float]:
    num_loops = model.num_loops
    ce_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    token_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            valid = labels.ne(tokenizer.pad_id)
            h, causal_mask = init_hidden(model, input_ids)

            loop_ce = []
            loop_scores = []
            for loop_idx in range(num_loops):
                h = run_loop_step(model, h, causal_mask, loop_idx)
                logits = decode_logits(model, h, loop_idx, num_loops)
                loop_ce.append(ce_per_sequence(logits, labels, tokenizer.pad_id))
                loop_scores.append(score_logits(logits, valid, policy, len(tokenizer)))
                h = apply_recurrence_postprocess(model, h, loop_idx, num_loops)

            ce_chunks.append(torch.stack(loop_ce, dim=1).cpu().numpy())
            score_chunks.append(torch.stack(loop_scores, dim=1).cpu().numpy())
            token_chunks.append(valid.sum(dim=1).cpu().numpy())

    ce = np.concatenate(ce_chunks, axis=0)
    scores = np.concatenate(score_chunks, axis=0)
    tokens = np.concatenate(token_chunks, axis=0)
    total_tokens = float(tokens.sum())
    k4_ppl = float(np.exp(ce[:, -1].sum() / total_tokens))

    finite = scores[:, : max(1, num_loops - 1)].reshape(-1)
    thresholds = np.unique(np.quantile(finite[np.isfinite(finite)], np.linspace(0, 1, num_thresholds)))
    thresholds = np.concatenate(([finite.min() - 1e-6], thresholds, [finite.max() + 1e-6]))

    best: dict[str, float] | None = None
    item_idx = np.arange(ce.shape[0])
    for threshold in thresholds:
        eligible = scores[:, : num_loops - 1] >= threshold
        has_exit = eligible.any(axis=1)
        exit_idx = np.where(has_exit, eligible.argmax(axis=1), num_loops - 1)
        ppl = float(np.exp(ce[item_idx, exit_idx].sum() / total_tokens))
        avg_loops = float(np.mean(exit_idx + 1))
        if ppl <= target_rel_ppl * k4_ppl:
            candidate = {"threshold": float(threshold), "calib_ppl": ppl, "calib_avg_loops": avg_loops}
            if best is None or candidate["calib_avg_loops"] < best["calib_avg_loops"]:
                best = candidate

    if best is None:
        best = {"threshold": float(finite.max() + 1e-6), "calib_ppl": k4_ppl, "calib_avg_loops": float(num_loops)}
    best["fixed_ppl"] = k4_ppl
    return best


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_fixed(
    model,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
    *,
    warmup_batches: int,
    bench_batches: int,
) -> dict[str, float]:
    vocab_size = len(tokenizer)

    def run_batches(max_batches: int) -> tuple[float, int]:
        ce_sum = 0.0
        tokens = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids)
                valid = labels.ne(tokenizer.pad_id)
                ce_sum += float(
                    F.cross_entropy(
                        outputs["final_logits"].float().reshape(-1, vocab_size),
                        labels.reshape(-1),
                        ignore_index=tokenizer.pad_id,
                        reduction="sum",
                    ).item()
                )
                tokens += int(valid.sum().item())
        return ce_sum, tokens

    run_batches(warmup_batches)
    sync_if_needed(device)
    start = time.perf_counter()
    ce_sum, tokens = run_batches(bench_batches)
    sync_if_needed(device)
    elapsed = time.perf_counter() - start

    return {
        "ppl": float(np.exp(ce_sum / tokens)),
        "avg_loops": float(model.num_loops),
        "tokens": float(tokens),
        "seconds": elapsed,
        "tokens_per_second": float(tokens / elapsed),
    }


def benchmark_dynamic(
    model,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float,
    policy: str,
    warmup_batches: int,
    bench_batches: int,
) -> dict[str, float]:
    num_loops = model.num_loops

    def run_batches(max_batches: int) -> tuple[float, int, float]:
        ce_sum = 0.0
        tokens = 0
        loop_sum = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                valid = labels.ne(tokenizer.pad_id)
                batch_tokens = valid.sum(dim=1)
                batch_size = input_ids.size(0)
                h, causal_mask = init_hidden(model, input_ids)
                active_idx = torch.arange(batch_size, device=device)
                selected_ce = torch.zeros(batch_size, device=device)
                exit_loops = torch.zeros(batch_size, device=device)

                for loop_idx in range(num_loops):
                    h = run_loop_step(model, h, causal_mask, loop_idx)
                    labels_active = labels[active_idx]
                    valid_active = valid[active_idx]
                    logits = decode_logits(model, h, loop_idx, num_loops)
                    seq_score = score_logits(logits, valid_active, policy, len(tokenizer))
                    seq_ce = ce_per_sequence(logits, labels_active, tokenizer.pad_id)

                    should_exit = seq_score >= threshold
                    if loop_idx == num_loops - 1:
                        should_exit = torch.ones_like(should_exit, dtype=torch.bool)

                    exiting_idx = active_idx[should_exit]
                    selected_ce[exiting_idx] = seq_ce[should_exit]
                    exit_loops[exiting_idx] = float(loop_idx + 1)

                    keep = ~should_exit
                    if keep.sum().item() == 0:
                        break

                    h = h[keep]
                    active_idx = active_idx[keep]
                    h = apply_recurrence_postprocess(model, h, loop_idx, num_loops)

                ce_sum += float(selected_ce.sum().item())
                tokens += int(batch_tokens.sum().item())
                loop_sum += float(exit_loops.sum().item())

        return ce_sum, tokens, loop_sum

    run_batches(warmup_batches)
    sync_if_needed(device)
    start = time.perf_counter()
    ce_sum, tokens, loop_sum = run_batches(bench_batches)
    sync_if_needed(device)
    elapsed = time.perf_counter() - start
    # Sequence count is tokens / seq_len because validation chunks are fixed-length.
    avg_loops = loop_sum / max(1, bench_batches * loader.batch_size)
    return {
        "ppl": float(np.exp(ce_sum / tokens)),
        "avg_loops": float(avg_loops),
        "tokens": float(tokens),
        "seconds": elapsed,
        "tokens_per_second": float(tokens / elapsed),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    rows: list[dict[str, Any]] = []
    for spec in args.checkpoint:
        label, ckpt_path = parse_spec(spec)
        print(f"\n=== {label} ===")
        model, tokenizer, val_ds, collate_fn, _ = load_model_and_val_data(ckpt_path, device)
        loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        print(f"  device={device} params={sum(p.numel() for p in model.parameters()):,}")

        calibration = calibrate_threshold(
            model,
            tokenizer,
            loader,
            device,
            policy=args.policy,
            max_batches=args.calib_batches,
            num_thresholds=args.num_thresholds,
            target_rel_ppl=args.target_rel_ppl,
        )
        print(
            f"  threshold={calibration['threshold']:.6g} "
            f"calib_ppl={calibration['calib_ppl']:.3f} "
            f"calib_loops={calibration['calib_avg_loops']:.2f} "
            f"fixed_ppl={calibration['fixed_ppl']:.3f}"
        )

        fixed = benchmark_fixed(
            model,
            tokenizer,
            loader,
            device,
            warmup_batches=args.warmup_batches,
            bench_batches=args.bench_batches,
        )
        dynamic = benchmark_dynamic(
            model,
            tokenizer,
            loader,
            device,
            threshold=calibration["threshold"],
            policy=args.policy,
            warmup_batches=args.warmup_batches,
            bench_batches=args.bench_batches,
        )
        speedup = dynamic["tokens_per_second"] / fixed["tokens_per_second"]
        print(
            f"  fixed: ppl={fixed['ppl']:.3f} tok/s={fixed['tokens_per_second']:.0f}; "
            f"dynamic: ppl={dynamic['ppl']:.3f} loops={dynamic['avg_loops']:.2f} "
            f"tok/s={dynamic['tokens_per_second']:.0f} speedup={speedup:.2f}x"
        )

        rows.append(
            {
                "checkpoint": label,
                "policy": args.policy,
                "threshold": calibration["threshold"],
                "calib_fixed_ppl": calibration["fixed_ppl"],
                "calib_dynamic_ppl": calibration["calib_ppl"],
                "calib_dynamic_avg_loops": calibration["calib_avg_loops"],
                "bench_fixed_ppl": fixed["ppl"],
                "bench_fixed_seconds": fixed["seconds"],
                "bench_fixed_tokens_per_second": fixed["tokens_per_second"],
                "bench_dynamic_ppl": dynamic["ppl"],
                "bench_dynamic_avg_loops": dynamic["avg_loops"],
                "bench_dynamic_seconds": dynamic["seconds"],
                "bench_dynamic_tokens_per_second": dynamic["tokens_per_second"],
                "bench_speedup": speedup,
            }
        )

    write_csv(output_dir / "adaptive_halt_benchmark.csv", rows)
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
