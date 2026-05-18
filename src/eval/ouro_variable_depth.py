"""Variable-depth perplexity evaluation for Ouro 1.4B checkpoints.

Reuses the load_model path from ouro_downstream_eval (handles α-readout,
no-norm, baseline). For each target K in 1..max_K, override the model's
total_ut_steps and measure cross-entropy perplexity on a fixed validation
chunk of WikiText-103.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from eval.ouro_downstream_eval import (  # noqa: E402
    load_model,
    parse_args as _ds_parse_args,  # noqa: F401  (only for shared helpers)
    resolve_device,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", action="append", required=True, help="label=path")
    p.add_argument("--model", default="ByteDance/Ouro-1.4B")
    p.add_argument("--max-K", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--num-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--output-dir", default="outputs/ouro_variable_depth")
    p.add_argument(
        "--readout-norm-mode",
        choices=["checkpoint", "no_norm", "baseline"],
        default="checkpoint",
    )
    p.add_argument("--dataset", choices=["fineweb", "wikitext"], default="fineweb")
    p.add_argument("--fineweb-subset", default="sample-10BT")
    p.add_argument("--fineweb-skip", type=int, default=9_500_000)
    return p.parse_args()


def get_validation_batches(
    tokenizer, seq_len: int, batch_size: int, num_batches: int,
    dataset: str = "fineweb",
    fineweb_subset: str = "sample-10BT",
    fineweb_skip: int = 9_500_000,
):
    """Yield fixed-length token chunks for variable-depth PPL.

    For Ouro-1.4B (trained on FineWeb), we draw from a tail slice of the same
    sample-10BT subset, skipping the first ~9.5M docs to avoid overlap with the
    1.3B-token training window. WikiText is supported as a fallback for sanity
    checks but inflates absolute PPL because Ouro is trained on web text.
    """
    from datasets import load_dataset

    if dataset == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
        ids = tokenizer.encode(text, add_special_tokens=False)
    elif dataset == "fineweb":
        # Stream the FineWeb sample-10BT subset and skip ahead to a held-out tail.
        ds = load_dataset(
            "HuggingFaceFW/fineweb", name=fineweb_subset, split="train", streaming=True
        )
        ds = ds.skip(fineweb_skip)
        ids: list[int] = []
        target = batch_size * num_batches * seq_len + 16
        for ex in ds:
            ids.extend(tokenizer.encode(ex["text"], add_special_tokens=False))
            if len(ids) >= target:
                break
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    chunks = []
    for i in range(0, len(ids) - seq_len, seq_len):
        chunks.append(ids[i : i + seq_len])
        if len(chunks) >= batch_size * num_batches:
            break
    out = []
    for b in range(num_batches):
        batch_ids = chunks[b * batch_size : (b + 1) * batch_size]
        if len(batch_ids) < batch_size:
            break
        out.append(torch.tensor(batch_ids, dtype=torch.long))
    return out


def find_ut_attr(model):
    """Return (object, attr_name) where total_ut_steps lives, so we can set it."""
    candidates = [
        (model, "total_ut_steps"),
        (getattr(model, "config", None), "total_ut_steps"),
        (getattr(model, "model", None), "total_ut_steps"),
        (getattr(getattr(model, "model", None), "config", None), "total_ut_steps"),
    ]
    found = []
    for obj, attr in candidates:
        if obj is not None and hasattr(obj, attr):
            found.append((obj, attr, getattr(obj, attr)))
    return found


def measure_ppl_at_K(model, batches, K: int, device, dtype, reset_norm):
    # Patch every total_ut_steps attribute we can find
    found = find_ut_attr(model)
    saved = [(obj, attr, getattr(obj, attr)) for (obj, attr, _) in found]
    try:
        for obj, attr, _ in found:
            setattr(obj, attr, K)
        ce_sum = 0.0
        token_count = 0
        for batch in batches:
            batch = batch.to(device)
            if reset_norm is not None:
                reset_norm()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                out = model(input_ids=batch, labels=batch, use_cache=False)
            loss = out.loss
            n = (batch.numel() - batch.shape[0])  # next-token positions
            ce_sum += float(loss) * n
            token_count += n
        avg_ce = ce_sum / max(1, token_count)
        return avg_ce, math.exp(avg_ce) if avg_ce < 50 else float("inf")
    finally:
        for obj, attr, val in saved:
            setattr(obj, attr, val)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in args.checkpoint:
        label, path = entry.split("=", 1)
        print(f"\n=== {label} : {path} ===", flush=True)
        # Reuse load_model from ouro_downstream_eval; it returns reset_norm too.
        ds_args = argparse.Namespace(
            checkpoint=path,
            model=args.model,
            readout_norm_mode=args.readout_norm_mode,
        )
        model, tokenizer, reset_norm, meta = load_model(ds_args, device, dtype)
        found = find_ut_attr(model)
        print(f"  total_ut_steps locations: {[(getattr(o,'__class__',type(o)).__name__, a, v) for (o,a,v) in found]}", flush=True)
        batches = get_validation_batches(
            tokenizer,
            args.seq_len,
            args.batch_size,
            args.num_batches,
            dataset=args.dataset,
            fineweb_subset=args.fineweb_subset,
            fineweb_skip=args.fineweb_skip,
        )
        print(f"  batches={len(batches)} seq_len={args.seq_len}", flush=True)

        for K in range(1, args.max_K + 1):
            ce, ppl = measure_ppl_at_K(model, batches, K, device, dtype, reset_norm)
            rows.append({
                "label": label,
                "checkpoint": path,
                "step": meta.get("step", -1),
                "tokens_seen_m": meta.get("tokens_seen_m"),
                "readout_norm_mode": meta.get("readout_norm_mode", "?"),
                "K": K,
                "ce": ce,
                "ppl": ppl,
            })
            print(f"  K={K:2d}  ce={ce:.4f}  ppl={ppl:.3f}", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = out_dir / "variable_depth.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json_path = out_dir / "variable_depth.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {csv_path} and {json_path}", flush=True)


if __name__ == "__main__":
    main()
