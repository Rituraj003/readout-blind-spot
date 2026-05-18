"""Verify Lemma 2 (large-scale residual expansion) on trained checkpoints.

For F(H) = H + B(Norm(H)) and H = su, Lemma 2 predicts:
    s_{k+1} - s_k = a(u_k) + O(s_k^{-1})              # radial increment ~ const
    ||u_{k+1} - u_k|| = ||b_perp(u_k)|| / s_k + O(s_k^{-2})   # angular step ~ 1/s

We measure both quantities directly across K=30 loops on validation tokens for
each trained checkpoint.

Output: per-loop arrays of (s_k, radial_increment_k, angular_step_k) saved to
JSON, then a 3-panel verification figure produced locally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from models.ar_looplm import build_ar_looplm_from_config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", action="append", required=True, help="label=path")
    p.add_argument("--target-K", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-batches", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="outputs/lemma2_verification")
    return p.parse_args()


@torch.no_grad()
def measure_lemma2(model, batches, target_K: int, device: torch.device):
    """Capture per-loop s_k, radial increment a_k, angular step ||du||.

    All computed on GPU per-token, then averaged to scalars before being moved
    to CPU. Memory is O(B*L*d) at any time, not O(K*B*L*d).
    """
    layers = model.layers
    causal_mask = model.causal_mask
    pos_emb = model.position_embedding
    tok_emb = model.token_embedding
    eps = 1e-9

    # Per-loop accumulators (k = 0..K-1)
    sum_s = [0.0] * target_K              # mean ||h_k|| (rms)
    sum_a = [0.0] * (target_K - 1)        # mean <u_k, h_{k+1} - h_k>
    sum_du = [0.0] * (target_K - 1)       # mean ||u_{k+1} - u_k||
    sum_b_perp = [0.0] * (target_K - 1)   # mean ||u_{k+1} - u_k|| * s_k (predicted to be ~const)
    cnt = [0] * target_K

    for batch_ids in batches:
        batch_ids = batch_ids.to(device)
        B, L = batch_ids.shape
        positions = torch.arange(L, device=device)
        h = tok_emb(batch_ids) + pos_emb(positions)
        cmask = causal_mask[:L, :L]

        # We need to remember h_k as the OUTPUT of one loop iteration (after layers,
        # before any inter-loop norm). u_k = h_k / s_k computed on raw h before
        # inter-loop normalization, so the comparison is in the same coordinate
        # system across conditions.
        prev_h = None
        for k in range(target_K):
            for layer in layers:
                h = layer(h, causal_mask=cmask)
            # Per-token RMS norm
            s_k = h.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=eps)  # [B, L, 1]
            u_k = h.float() / s_k  # unit-RMS direction
            # accumulate
            sum_s[k] += float(s_k.mean().item())
            cnt[k] += 1
            if prev_h is not None:
                # We have prev_h = h_k (last loop's output) and h = h_{k+1} (this loop's output)
                # WAIT: at this point in iteration k, h IS h_{k+1} relative to the previous iteration.
                # Let's re-read the indexing carefully:
                #   - At iteration k, we apply layers to (possibly normalized) prev state and produce h.
                #   - At first iteration (k=0), prev_h is None and we are producing h_0.
                #   - At second iteration (k=1), h is h_1 and prev_h is h_0.
                # So the increment computed here is for the transition (k-1) -> k, stored at index k-1.
                idx = k - 1
                # Radial increment: a_{k-1} = <u_{k-1}, h_k - h_{k-1}>
                #   prev_s and prev_u are from prev_h
                prev_s = prev_h.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=eps)
                prev_u = prev_h.float() / prev_s
                delta_h = h.float() - prev_h.float()
                a = (prev_u * delta_h).sum(dim=-1)  # [B, L] dot product per token
                sum_a[idx] += float(a.mean().item())
                # Angular step: ||u_k - u_{k-1}||
                du = (u_k - prev_u).pow(2).sum(dim=-1).clamp(min=0).sqrt()  # [B, L]
                sum_du[idx] += float(du.mean().item())
                # Predicted constant: ||u_k - u_{k-1}|| * s_{k-1} ≈ ||b_perp(u_{k-1})||
                # (using prev_s, since Lemma 2 says angular step at transition k-1 -> k scales as 1/s_{k-1})
                bperp_est = (du * prev_s.squeeze(-1)).mean()
                sum_b_perp[idx] += float(bperp_est.item())

            # Snapshot pre-norm h for next iteration's radial/angular computation
            prev_h = h.detach().clone()

            # Apply inter-loop norm for next iteration (matches training-time path)
            if model.inter_loop_norm:
                h = model.loop_norm(h)
            if model.use_spectral_damping and k < target_K - 1:
                alpha = torch.sigmoid(model.raw_alpha)
                h = h * alpha

    avg_s = [sum_s[k] / max(cnt[k], 1) for k in range(target_K)]
    avg_a = [sum_a[k] / max(cnt[0], 1) for k in range(target_K - 1)]
    avg_du = [sum_du[k] / max(cnt[0], 1) for k in range(target_K - 1)]
    avg_bperp = [sum_b_perp[k] / max(cnt[0], 1) for k in range(target_K - 1)]
    return avg_s, avg_a, avg_du, avg_bperp


def get_validation_batches(val_ids, seq_len, batch_size, num_batches):
    chunks = []
    for i in range(0, len(val_ids) - seq_len, seq_len):
        chunks.append(val_ids[i : i + seq_len])
        if len(chunks) >= batch_size * num_batches:
            break
    out = []
    for b in range(num_batches):
        batch_ids = chunks[b * batch_size : (b + 1) * batch_size]
        if len(batch_ids) < batch_size:
            break
        out.append(torch.tensor(batch_ids, dtype=torch.long))
    return out


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
        if tok_state is None:
            raise RuntimeError(f"{path}: no tokenizer in checkpoint")
        tokenizer = RegexTokenizer.from_state(tok_state)
        data_cfg = cfg["data"]
        val_texts = load_hf_text_split(data_cfg, split="validation")
        join_with = data_cfg.get("text_join", "\n\n")
        val_ids = tokenizer.encode(join_texts(val_texts, join_with))
        batches = get_validation_batches(val_ids, args.seq_len, args.batch_size, args.num_batches)
        print(f"  batches={len(batches)} seq_len={args.seq_len}", flush=True)

        model = build_ar_looplm_from_config(cfg, vocab_size=len(tokenizer), dropout_override=0.0)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model = model.to(device).eval()

        s, a, du, bperp = measure_lemma2(model, batches, args.target_K, device)
        print(f"  K  s_k         a_k (radial inc)    ||du||              ||du||*s_k (b_perp est)", flush=True)
        for k in range(args.target_K):
            sline = f"  {k:2d}  {s[k]:9.2f}"
            if k < args.target_K - 1:
                sline += f"  {a[k]:+9.4f}            {du[k]:.5f}             {bperp[k]:.4f}"
            print(sline, flush=True)
        results[label] = {
            "checkpoint": path,
            "s": s,
            "radial_increment": a,
            "angular_step": du,
            "b_perp_estimate": bperp,
            "config_supervision": cfg.get("supervision", {}).get("mode", "?"),
            "config_use_decode_norm": cfg.get("model", {}).get("use_decode_norm", "?"),
            "config_use_spectral_damping": cfg.get("model", {}).get("use_spectral_damping", "?"),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_json = out_dir / "lemma2_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_json}", flush=True)


if __name__ == "__main__":
    main()
