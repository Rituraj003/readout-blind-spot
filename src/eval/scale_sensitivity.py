"""Scale sensitivity test: the smoking gun for readout scale invariance.

For each checkpoint, multiply the final hidden state by α ∈ [0.1, 10]
before the readout, and measure CE loss. If the readout is scale-invariant,
loss should be nearly flat. If not, loss should change dramatically.

This directly tests the paper's core claim without any training.

Usage:
    uv run python src/eval/scale_sensitivity.py \
      --checkpoint norm=outputs/ar_looplm_full_terminal_norm/last.pt \
      --checkpoint raw=outputs/ar_looplm_full_terminal_raw/last.pt \
      --device auto
"""

from __future__ import annotations

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Scale sensitivity test.")
    parser.add_argument("--checkpoint", type=str, action="append", required=True,
                        help="label=/path/to/checkpoint.pt")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="outputs/scale_sensitivity")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def parse_spec(spec):
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), Path(path.strip())
    return Path(spec).parent.name, Path(spec)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train_ar_looplm import load_config, build_dataset, TextChunkDataset, collate_fn
    from models.ar_looplm import AutoregressiveLoopLM

    all_results = {}

    for spec in args.checkpoint:
        label, ckpt_path = parse_spec(spec)
        print(f"\n=== {label} ===")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        mc = config["model"]

        # Rebuild tokenizer
        tok_state = ckpt.get("tokenizer")
        if tok_state and tok_state.get("kind") == "regex":
            from data.dataset import RegexTokenizer
            tokenizer = RegexTokenizer.from_state(tok_state)
        else:
            tokenizer, _, _ = build_dataset(config)

        # Build model
        from models.ar_looplm import build_ar_looplm_from_config
        model = build_ar_looplm_from_config(
            config,
            vocab_size=len(tokenizer),
            dropout_override=0.0,
        ).to(device).eval()
        # Strict load: arch must match training config (alpha settings included).
        model.load_state_dict(ckpt["model_state"], strict=True)

        decode_norm_type = type(model.decode_norm).__name__
        print(f"  decode_norm: {decode_norm_type}")
        print(f"  use_decode_norm: {model.use_decode_norm}")

        # Build val data
        _, _, val_ids = build_dataset(config)
        seq_len = config["data"]["seq_len"]
        val_ds = TextChunkDataset(val_ids, seq_len, seq_len)
        loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

        V = len(tokenizer)

        # For each α, scale the hidden state before readout and measure loss
        alpha_losses = {}
        for alpha in args.alphas:
            ce_sum = 0.0
            total = 0

            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if i >= args.max_batches:
                        break
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    B, L = input_ids.shape

                    # Run the model's forward manually, intercepting before readout
                    positions = torch.arange(L, device=device)
                    h = model.token_embedding(input_ids) + model.position_embedding(positions)

                    causal_mask = model.causal_mask[:L, :L]

                    for k in range(model.num_loops):
                        for layer in model.layers:
                            h = layer(h, causal_mask=causal_mask)

                        if model.inter_loop_norm and k < model.num_loops - 1:
                            h = model.loop_norm(h)

                        if model.use_spectral_damping and k < model.num_loops - 1:
                            a = torch.sigmoid(model.raw_alpha)
                            h = h * a

                    # Scale the final hidden state by α
                    h_scaled = h * alpha

                    # Apply readout (with or without decode_norm)
                    if model.use_decode_norm:
                        logits = model.lm_head(model.decode_norm(h_scaled))
                    else:
                        logits = model.lm_head(h_scaled)

                    ce = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1),
                                         ignore_index=tokenizer.pad_id, reduction="sum")
                    ce_sum += ce.item()
                    total += B * L

            loss = ce_sum / total
            ppl = np.exp(loss)
            alpha_losses[alpha] = {"loss": loss, "ppl": ppl}
            print(f"  α={alpha:5.2f}: CE={loss:.4f} PPL={ppl:.2f}")

        all_results[label] = alpha_losses

    # Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold"})

    # Pretty labels for paper
    pretty_labels = {"norm": "RMSNorm readout", "raw": "No readout norm",
                     "terminal_norm": "Terminal + RMSNorm", "terminal_raw": "Terminal + Raw",
                     "perstep_norm": "Per-step + RMSNorm", "perstep_raw": "Per-step + Raw"}

    colors = {"norm": "#1f77b4", "raw": "#2ca02c",
              "terminal_norm": "#1f77b4", "terminal_raw": "#2ca02c",
              "perstep_norm": "#ff7f0e", "perstep_raw": "#d62728"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for label, alpha_losses in all_results.items():
        alphas = sorted(alpha_losses.keys())
        losses = [alpha_losses[a]["loss"] for a in alphas]
        ppls = [alpha_losses[a]["ppl"] for a in alphas]
        c = colors.get(label, "gray")
        pl = pretty_labels.get(label, label)

        ax1.plot(alphas, losses, marker="o", markersize=5, linewidth=2.2,
                 color=c, label=pl)
        ax2.plot(alphas, ppls, marker="o", markersize=5, linewidth=2.2,
                 color=c, label=pl)

    ax1.set_xscale("log")
    ax1.set_xlabel(r"Scale factor $\alpha$")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss vs Hidden-State Scale")
    ax1.legend()
    ax1.axvline(x=1.0, color="gray", linestyle="--", alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Scale factor $\alpha$")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Perplexity vs Hidden-State Scale")
    ax2.legend()
    ax2.axvline(x=1.0, color="gray", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "scale_sensitivity.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(output_dir / "scale_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
