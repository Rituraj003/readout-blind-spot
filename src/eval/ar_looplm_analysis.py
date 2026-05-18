"""Analysis suite for autoregressive LoopLM experiments.

1. Per-loop perplexity: decode at each loop iteration, measure next-token PPL
2. Variable loop depth: evaluate at K != training K
3. Norm growth across loops
4. Per-loop text quality: generate actual text from each loop's logits

Usage:
    uv run python src/eval/ar_looplm_analysis.py \
      --checkpoint terminal=outputs/ar_looplm_terminal/last.pt \
      --checkpoint perstep=outputs/ar_looplm_perstep/last.pt \
      --device auto
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_ROOT = _PROJECT_ROOT / ".cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AR LoopLM checkpoints.")
    parser.add_argument("--checkpoint", type=str, action="append", required=True,
                        help="label=/path/to/checkpoint.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="outputs/ar_looplm_analysis")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def parse_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), Path(path.strip())
    return Path(spec).parent.name, Path(spec)


def load_model_and_data(ckpt_path: Path, device: torch.device):
    """Load checkpoint, reconstruct model and validation data."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train_ar_looplm import build_dataset, TextChunkDataset, collate_fn
    from models.ar_looplm import AutoregressiveLoopLM
    from data.dataset import tokenizer_from_state

    # Rebuild tokenizer from checkpoint
    tok_state = ckpt.get("tokenizer")
    if tok_state and tok_state.get("kind") == "regex":
        from data.dataset import RegexTokenizer
        tokenizer = RegexTokenizer.from_state(tok_state)
    else:
        # Fallback: rebuild from data
        tokenizer, _, _ = build_dataset(config)

    from models.ar_looplm import build_ar_looplm_from_config
    model = build_ar_looplm_from_config(
        config,
        vocab_size=len(tokenizer),
        dropout_override=0.0,
    ).to(device).eval()

    # Strict load: any missing/unexpected key is a real architecture mismatch
    # (e.g., alpha config diverged from training). Fail loudly instead of silently
    # using default values.
    model.load_state_dict(ckpt["model_state"], strict=True)

    # Build val data
    _, _, val_ids = build_dataset(config)
    seq_len = config["data"]["seq_len"]
    val_ds = TextChunkDataset(val_ids, seq_len, seq_len)

    return model, tokenizer, val_ds, config, collate_fn


def analyze_per_loop_ppl(model, tokenizer, val_ds, collate_fn, device, max_batches):
    """Measure perplexity at each loop iteration."""
    from torch.utils.data import DataLoader

    loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    V = len(tokenizer)
    K = model.num_loops

    # Per-loop CE accumulators
    loop_ce = [0.0] * K
    loop_norms = [0.0] * K
    total_tokens = 0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            if n_batches >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)

            bs = input_ids.size(0) * input_ids.size(1)
            total_tokens += bs

            for k, logits_k in enumerate(outputs["all_logits"]):
                ce = F.cross_entropy(
                    logits_k.reshape(-1, V), labels.reshape(-1),
                    ignore_index=tokenizer.pad_id, reduction="sum"
                ).item()
                loop_ce[k] += ce

            for k, norm in enumerate(outputs["all_norms"]):
                loop_norms[k] += norm

            n_batches += 1

    ppl = [np.exp(ce / total_tokens) for ce in loop_ce]
    avg_norms = [n / n_batches for n in loop_norms]
    return ppl, avg_norms


def analyze_variable_depth(model, tokenizer, val_ds, collate_fn, device, max_batches):
    """Evaluate at different loop depths (K != training K)."""
    from torch.utils.data import DataLoader

    loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    V = len(tokenizer)
    train_K = model.num_loops

    # Test depths: 1 through 3*train_K
    test_depths = list(range(1, train_K * 3 + 1))
    results = {}

    for test_K in test_depths:
        ce_sum = 0.0
        total = 0
        norms = []

        # Temporarily change loop count
        old_K = model.num_loops
        model.num_loops = test_K

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= max_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)
                ce = F.cross_entropy(
                    outputs["final_logits"].reshape(-1, V),
                    labels.reshape(-1),
                    ignore_index=tokenizer.pad_id, reduction="sum"
                ).item()

                bs = input_ids.size(0) * input_ids.size(1)
                ce_sum += ce
                total += bs
                norms.append(outputs["all_norms"][-1])  # final loop norm

        model.num_loops = old_K
        ppl = np.exp(ce_sum / total)
        avg_norm = np.mean(norms)
        results[test_K] = {"ppl": ppl, "norm": avg_norm}
        print(f"  K={test_K}: ppl={ppl:.2f} norm={avg_norm:.1f}")

    return results


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    all_results = {}

    for spec in args.checkpoint:
        label, ckpt_path = parse_spec(spec)
        print(f"\n=== {label} ===")

        model, tokenizer, val_ds, config, cf = load_model_and_data(ckpt_path, device)
        print(f"  params={sum(p.numel() for p in model.parameters()):,} "
              f"loops={model.num_loops} vocab={len(tokenizer)}")

        # 1. Per-loop PPL
        print("  Per-loop perplexity:")
        ppl, norms = analyze_per_loop_ppl(model, tokenizer, val_ds, cf, device, args.max_batches)
        for k, (p, n) in enumerate(zip(ppl, norms)):
            print(f"    loop {k}: ppl={p:.2f} norm={n:.1f}")

        # 2. Variable depth
        print("  Variable depth:")
        var_depth = analyze_variable_depth(model, tokenizer, val_ds, cf, device, args.max_batches)

        all_results[label] = {
            "per_loop_ppl": ppl,
            "per_loop_norms": norms,
            "variable_depth": var_depth,
        }

    # Plot
    labels = list(all_results.keys())
    pretty_labels = {"norm": "RMSNorm readout", "raw": "No readout norm",
                     "terminal_norm": "Terminal + RMSNorm", "terminal_raw": "Terminal + Raw"}
    colors = {"norm": "#1f77b4", "raw": "#2ca02c",
              "terminal_norm": "#1f77b4", "terminal_raw": "#2ca02c",
              "terminal": "#1f77b4", "perstep": "#ff7f0e", "per-step": "#ff7f0e"}

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold"})

    # Figure 1: Per-loop PPL + norm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for label in labels:
        r = all_results[label]
        K = len(r["per_loop_ppl"])
        c = colors.get(label, "gray")
        pl = pretty_labels.get(label, label)
        ax1.plot(range(K), r["per_loop_ppl"], marker="o", color=c, linewidth=2.2, label=pl)
        ax2.plot(range(K), r["per_loop_norms"], marker="o", color=c, linewidth=2.2, label=pl)

    ax1.set_xlabel("Loop Iteration")
    ax1.set_ylabel("Perplexity")
    ax1.set_title("Per-Loop Perplexity")
    ax1.legend()
    ax2.set_xlabel("Loop Iteration")
    ax2.set_ylabel("Hidden State Norm")
    ax2.set_yscale("log")
    ax2.set_title("Per-Loop Norm")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "per_loop_analysis.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(output_dir / "per_loop_analysis.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Variable depth — the key downstream result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for label in labels:
        r = all_results[label]
        depths = sorted(r["variable_depth"].keys())
        ppls = [r["variable_depth"][d]["ppl"] for d in depths]
        vnorms = [r["variable_depth"][d]["norm"] for d in depths]
        c = colors.get(label, "gray")
        pl = pretty_labels.get(label, label)

        train_K = len(r["per_loop_ppl"])
        ax1.plot(depths, ppls, marker="o", markersize=5, color=c, linewidth=2.2, label=pl)
        ax1.axvline(x=train_K, color="gray", linestyle="--", alpha=0.3)
        ax2.plot(depths, vnorms, marker="o", markersize=5, color=c, linewidth=2.2, label=pl)
        ax2.axvline(x=train_K, color="gray", linestyle="--", alpha=0.3)

    ax1.set_xlabel("Inference Loop Depth K")
    ax1.set_ylabel("Perplexity")
    ax1.set_title("PPL vs Inference Depth")
    ax1.legend()
    ax2.set_xlabel("Inference Loop Depth K")
    ax2.set_ylabel("Final Hidden State Norm")
    ax2.set_yscale("log")
    ax2.set_title("Norm vs Inference Depth")
    ax2.legend()
    # Add annotation for training K
    ax1.annotate("training K", xy=(train_K, ax1.get_ylim()[1]),
                 fontsize=9, color="gray", ha="center", va="top")
    fig.tight_layout()
    fig.savefig(output_dir / "variable_depth.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(output_dir / "variable_depth.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
