"""Estimate early-exit compute/quality curves for autoregressive LoopLMs.

The model still computes all loops during this offline evaluation. The script
uses intermediate logits to simulate policies that would stop after loop k.

Default granularity is sequence-level exit, which corresponds to actually
saving loop iterations in a standard shared-depth implementation.

Example:
    uv run python src/eval/early_exit_curve.py \
      --checkpoint perstep_raw=outputs/ar_looplm_150m_perstep_raw/last.pt \
      --checkpoint perstep_norm=outputs/ar_looplm_150m_perstep_norm/last.pt \
      --device mps \
      --max-batches 10
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_ROOT = _PROJECT_ROOT / ".cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


POLICIES = ("max_prob", "logit_margin", "confidence_entropy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Early-exit compute/quality curves.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        action="append",
        required=True,
        help="label=/path/to/checkpoint.pt",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--num-thresholds", type=int, default=31)
    parser.add_argument(
        "--granularity",
        choices=("sequence", "token"),
        default="sequence",
        help="Exit whole sequences, or simulate token-wise exits for analysis.",
    )
    parser.add_argument(
        "--policy",
        choices=POLICIES,
        action="append",
        default=None,
        help="Policy to sweep. Defaults to all policies.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/early_exit_curve")
    parser.add_argument("--dpi", type=int, default=300)
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
    """Load a LoopLM checkpoint and validation chunks with the saved tokenizer."""
    from data.dataset import RegexTokenizer, join_texts, load_hf_text_split
    from models.ar_looplm import AutoregressiveLoopLM
    from train_ar_looplm import TextChunkDataset, collate_fn

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    tok_state = ckpt.get("tokenizer")

    if tok_state and tok_state.get("kind") == "regex":
        tokenizer = RegexTokenizer.from_state(tok_state)
        data_config = config["data"]
        val_texts = load_hf_text_split(data_config, data_config["hf_val_split"])
        join_with = data_config.get("join_with", "\n\n")
        val_ids = tokenizer.encode(join_texts(val_texts, join_with))
    else:
        from train_ar_looplm import build_dataset

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
    val_ds = TextChunkDataset(val_ids, seq_len, seq_len)
    return model, tokenizer, val_ds, collate_fn, config


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    denom = mask.sum(dim=dim).clamp_min(1)
    return (x * mask).sum(dim=dim) / denom


def collect_loop_metrics(
    *,
    model,
    tokenizer,
    val_ds,
    collate_fn,
    device: torch.device,
    batch_size: int,
    max_batches: int,
    granularity: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return CE matrix [N, K], token counts [N], and score matrices [N, K]."""
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    vocab_size = len(tokenizer)
    num_loops = model.num_loops
    log_vocab = float(np.log(vocab_size))

    ce_chunks: list[np.ndarray] = []
    token_chunks: list[np.ndarray] = []
    score_chunks: dict[str, list[np.ndarray]] = {p: [] for p in POLICIES}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            valid = labels.ne(tokenizer.pad_id)
            outputs = model(input_ids)

            loop_ce = []
            loop_scores: dict[str, list[torch.Tensor]] = {p: [] for p in POLICIES}

            for logits in outputs["all_logits"]:
                logits_f = logits.float()
                flat_ce = F.cross_entropy(
                    logits_f.reshape(-1, vocab_size),
                    labels.reshape(-1),
                    ignore_index=tokenizer.pad_id,
                    reduction="none",
                ).view_as(labels)
                loop_ce.append(flat_ce)

                top2 = torch.topk(logits_f, k=2, dim=-1).values
                log_z = torch.logsumexp(logits_f, dim=-1)
                max_prob = torch.exp(top2[..., 0] - log_z)
                logit_margin = top2[..., 0] - top2[..., 1]

                log_probs = F.log_softmax(logits_f, dim=-1)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1)
                confidence_entropy = 1.0 - entropy / log_vocab

                loop_scores["max_prob"].append(max_prob)
                loop_scores["logit_margin"].append(logit_margin)
                loop_scores["confidence_entropy"].append(confidence_entropy)

            if granularity == "sequence":
                mask_f = valid.float()
                ce_mat = torch.stack(
                    [(ce * mask_f).sum(dim=1) for ce in loop_ce],
                    dim=1,
                )
                tokens = valid.sum(dim=1).float()
                score_mats = {
                    policy: torch.stack(
                        [_masked_mean(score, mask_f, dim=1) for score in scores],
                        dim=1,
                    )
                    for policy, scores in loop_scores.items()
                }
            else:
                valid_flat = valid.reshape(-1)
                ce_mat = torch.stack([ce.reshape(-1)[valid_flat] for ce in loop_ce], dim=1)
                tokens = torch.ones(ce_mat.size(0), device=ce_mat.device)
                score_mats = {
                    policy: torch.stack(
                        [score.reshape(-1)[valid_flat] for score in scores],
                        dim=1,
                    )
                    for policy, scores in loop_scores.items()
                }

            ce_chunks.append(ce_mat.cpu().numpy())
            token_chunks.append(tokens.cpu().numpy())
            for policy, score_mat in score_mats.items():
                score_chunks[policy].append(score_mat.cpu().numpy())

            print(f"  collected batch {batch_idx + 1}/{max_batches}")

    ce = np.concatenate(ce_chunks, axis=0)
    tokens = np.concatenate(token_chunks, axis=0)
    scores = {policy: np.concatenate(chunks, axis=0) for policy, chunks in score_chunks.items()}
    return ce, tokens, scores


def fixed_depth_rows(label: str, ce: np.ndarray, tokens: np.ndarray) -> list[dict[str, object]]:
    rows = []
    num_loops = ce.shape[1]
    total_tokens = float(tokens.sum())
    for k in range(num_loops):
        total_ce = float(ce[:, k].sum())
        ppl = float(np.exp(total_ce / total_tokens))
        exits = [0] * num_loops
        exits[k] = int(ce.shape[0])
        rows.append(
            {
                "checkpoint": label,
                "policy": f"fixed_K{k + 1}",
                "threshold": "",
                "avg_loops": float(k + 1),
                "avg_loop_fraction": float((k + 1) / num_loops),
                "ppl": ppl,
                "ce": total_ce / total_tokens,
                "num_items": int(ce.shape[0]),
                "num_tokens": int(total_tokens),
                **{f"exit_K{i + 1}": exits[i] for i in range(num_loops)},
            }
        )
    return rows


def policy_rows(
    label: str,
    policy: str,
    ce: np.ndarray,
    tokens: np.ndarray,
    scores: np.ndarray,
    num_thresholds: int,
) -> list[dict[str, object]]:
    num_items, num_loops = ce.shape
    total_tokens = float(tokens.sum())

    flat_scores = scores[:, : max(1, num_loops - 1)].reshape(-1)
    finite_scores = flat_scores[np.isfinite(flat_scores)]
    if finite_scores.size == 0:
        return []

    quantiles = np.linspace(0.0, 1.0, num_thresholds)
    thresholds = np.unique(np.quantile(finite_scores, quantiles))
    thresholds = np.concatenate(
        ([finite_scores.min() - 1e-6], thresholds, [finite_scores.max() + 1e-6])
    )

    item_idx = np.arange(num_items)
    rows = []
    for threshold in thresholds:
        eligible = scores[:, : num_loops - 1] >= threshold
        has_exit = eligible.any(axis=1)
        exit_idx = np.where(has_exit, eligible.argmax(axis=1), num_loops - 1)

        selected_ce = ce[item_idx, exit_idx]
        total_ce = float(selected_ce.sum())
        ppl = float(np.exp(total_ce / total_tokens))
        avg_loops = float(np.mean(exit_idx + 1))
        exit_counts = np.bincount(exit_idx, minlength=num_loops)

        rows.append(
            {
                "checkpoint": label,
                "policy": policy,
                "threshold": float(threshold),
                "avg_loops": avg_loops,
                "avg_loop_fraction": float(avg_loops / num_loops),
                "ppl": ppl,
                "ce": total_ce / total_tokens,
                "num_items": int(num_items),
                "num_tokens": int(total_tokens),
                **{f"exit_K{i + 1}": int(exit_counts[i]) for i in range(num_loops)},
            }
        )
    rows.sort(key=lambda r: (float(r["avg_loops"]), float(r["ppl"])))
    return rows


def write_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, object]], output_dir: Path, dpi: int) -> None:
    adaptive_rows = [r for r in rows if not str(r["policy"]).startswith("fixed_")]
    fixed_rows = [r for r in rows if str(r["policy"]).startswith("fixed_")]

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in adaptive_rows:
        grouped.setdefault((str(row["checkpoint"]), str(row["policy"])), []).append(row)

    for (checkpoint, policy), group in grouped.items():
        group = sorted(group, key=lambda r: float(r["avg_loops"]))
        xs = [float(r["avg_loops"]) for r in group]
        ys = [float(r["ppl"]) for r in group]
        ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.8, label=f"{checkpoint}: {policy}")

    for row in fixed_rows:
        ax.scatter(
            [float(row["avg_loops"])],
            [float(row["ppl"])],
            marker="x",
            s=70,
            color="black",
            alpha=0.55,
        )

    all_ppl = [float(r["ppl"]) for r in rows if np.isfinite(float(r["ppl"]))]
    if all_ppl and max(all_ppl) > 3 * min(all_ppl):
        ax.set_yscale("log")

    ax.set_xlabel("Average Loops Used")
    ax.set_ylabel("Perplexity")
    ax.set_title("Early-Exit Compute/Quality Curve")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "early_exit_curve.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "early_exit_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def print_summary(label: str, rows: list[dict[str, object]]) -> None:
    fixed = [r for r in rows if str(r["policy"]).startswith("fixed_")]
    adaptive = [r for r in rows if not str(r["policy"]).startswith("fixed_")]
    print(f"\n=== {label} summary ===")
    for row in fixed:
        print(f"  {row['policy']}: loops={float(row['avg_loops']):.1f} ppl={float(row['ppl']):.2f}")
    for policy in POLICIES:
        candidates = [r for r in adaptive if r["policy"] == policy]
        if not candidates:
            continue
        best = min(candidates, key=lambda r: float(r["ppl"]))
        low_compute = min(candidates, key=lambda r: (float(r["avg_loops"]), float(r["ppl"])))
        print(
            f"  {policy}: best ppl={float(best['ppl']):.2f} "
            f"at {float(best['avg_loops']):.2f} loops; "
            f"lowest-compute point ppl={float(low_compute['ppl']):.2f} "
            f"at {float(low_compute['avg_loops']):.2f} loops"
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    policies = tuple(args.policy) if args.policy else POLICIES
    all_rows: list[dict[str, object]] = []

    for spec in args.checkpoint:
        label, ckpt_path = parse_spec(spec)
        print(f"\n=== {label} ===")
        model, tokenizer, val_ds, cf, config = load_model_and_val_data(ckpt_path, device)
        print(
            f"  device={device} params={sum(p.numel() for p in model.parameters()):,} "
            f"loops={model.num_loops} granularity={args.granularity}"
        )
        print(
            "  config: "
            f"supervision={config.get('supervision', {}).get('mode')} "
            f"use_decode_norm={config.get('model', {}).get('use_decode_norm')} "
            f"decode_norm_final_only={config.get('model', {}).get('decode_norm_final_only', False)} "
            f"norm_penalty={config.get('supervision', {}).get('norm_penalty_weight', 0.0)}"
        )

        ce, tokens, scores = collect_loop_metrics(
            model=model,
            tokenizer=tokenizer,
            val_ds=val_ds,
            collate_fn=cf,
            device=device,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            granularity=args.granularity,
        )

        rows = fixed_depth_rows(label, ce, tokens)
        for policy in policies:
            rows.extend(
                policy_rows(
                    label=label,
                    policy=policy,
                    ce=ce,
                    tokens=tokens,
                    scores=scores[policy],
                    num_thresholds=args.num_thresholds,
                )
            )
        all_rows.extend(rows)
        print_summary(label, rows)

    write_csv(all_rows, output_dir / "early_exit_curve.csv")
    plot_rows(all_rows, output_dir, args.dpi)
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
