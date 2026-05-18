"""Per-token norm statistics for the 2x2 ablation.

Reports mean / median / p99 / max / std of ||H_K||_2 (per-token) across
validation tokens. Addresses the concern that the global RMS in Table 3
hides per-token outlier behavior.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.ar_looplm import build_ar_looplm_from_config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", action="append", required=True, help="label=path")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-batches", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="outputs/token_norm_stats")
    return p.parse_args()


@torch.no_grad()
def measure_token_norms(model, batches, device):
    """Run K=K_max loops on each batch; return per-token ||H_K||_2 across tokens."""
    layers = model.layers
    causal_mask = model.causal_mask
    pos_emb = model.position_embedding
    tok_emb = model.token_embedding
    K = model.num_loops

    norms_all = []  # flat array over tokens
    for batch_ids in batches:
        batch_ids = batch_ids.to(device)
        B, L = batch_ids.shape
        positions = torch.arange(L, device=device)
        h = tok_emb(batch_ids) + pos_emb(positions)
        cmask = causal_mask[:L, :L]
        for k in range(K):
            for layer in layers:
                h = layer(h, causal_mask=cmask)
            if k < K - 1:
                if model.inter_loop_norm:
                    h = model.loop_norm(h)
                if model.use_spectral_damping:
                    alpha = torch.sigmoid(model.raw_alpha)
                    h = h * alpha
        # h is final-loop H_K; compute per-token L2 norm
        per_token = h.float().norm(dim=-1).reshape(-1).cpu().numpy()  # [B*L]
        norms_all.append(per_token)
    arr = np.concatenate(norms_all, axis=0)
    return {
        "n_tokens": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "min": float(arr.min()),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from data.dataset import RegexTokenizer, join_texts, load_hf_text_split

    results = {}
    for entry in args.checkpoint:
        label, path = entry.split("=", 1)
        print(f"\n=== {label} : {path} ===", flush=True)
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        tok_state = ckpt.get("tokenizer")
        tokenizer = RegexTokenizer.from_state(tok_state)
        data_cfg = cfg["data"]
        val_texts = load_hf_text_split(data_cfg, split="validation")
        join_with = data_cfg.get("text_join", "\n\n")
        val_ids = tokenizer.encode(join_texts(val_texts, join_with))
        chunks = []
        for i in range(0, len(val_ids) - args.seq_len, args.seq_len):
            chunks.append(val_ids[i:i + args.seq_len])
            if len(chunks) >= args.batch_size * args.num_batches:
                break
        batches = []
        for b in range(args.num_batches):
            slc = chunks[b * args.batch_size:(b + 1) * args.batch_size]
            if len(slc) < args.batch_size:
                break
            batches.append(torch.tensor(slc, dtype=torch.long))
        print(f"  batches={len(batches)} seq_len={args.seq_len}", flush=True)
        model = build_ar_looplm_from_config(cfg, vocab_size=len(tokenizer), dropout_override=0.0)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model = model.to(device).eval()
        stats = measure_token_norms(model, batches, device)
        print(f"  n_tokens={stats['n_tokens']} mean={stats['mean']:.2f} "
              f"median={stats['median']:.2f} p99={stats['p99']:.2f} "
              f"max={stats['max']:.2f} std={stats['std']:.2f}", flush=True)
        results[label] = {"checkpoint": path, **stats}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = out_dir / "token_norm_stats.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
