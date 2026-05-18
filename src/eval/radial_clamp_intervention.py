"""Radial clamp intervention for autoregressive LoopLM checkpoints.

The interventions remove recurrent radial scale growth at inference time while
preserving as much of the learned update as possible.  The most direct mode,
``remove_radial``, subtracts only the component of each residual loop update
parallel to the previous loop state.  The simpler ``freeze_loop1`` mode leaves
loop 1 unchanged and then rescales every later loop state to the per-token RMS
scale measured at loop 1 before decoding and before feeding the state to the
next recurrent step.

This tests whether the recurrent scale coordinate is active but weakly used by
the supervised interface:

* normalized readouts should be nearly insensitive to the clamp, except through
  future-direction changes;
* raw readouts should be more affected because the readout uses hidden scale.

Example:
    python src/eval/radial_clamp_intervention.py \
      --checkpoint norm=outputs/ar_looplm_150m_perstep_norm/last.pt \
      --checkpoint raw=outputs/ar_looplm_150m_perstep_raw/last.pt \
      --device cuda --batch-size 8 --max-batches 20
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Radial clamp intervention.")
    parser.add_argument("--checkpoint", action="append", required=True, help="label=/path/to/checkpoint.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument(
        "--mode",
        action="append",
        default=["none", "remove_radial", "freeze_loop1"],
        choices=["none", "remove_radial", "freeze_loop1"],
        help="Clamp mode to evaluate. Repeat to add modes.",
    )
    parser.add_argument("--output-dir", default="outputs/radial_clamp_intervention")
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
    from train_ar_looplm import TextChunkDataset, build_dataset, collate_fn
    from models.ar_looplm import build_ar_looplm_from_config

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

    model = build_ar_looplm_from_config(
        config,
        vocab_size=len(tokenizer),
        dropout_override=0.0,
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state"], strict=True)

    seq_len = config["data"]["seq_len"]
    val_ds = TextChunkDataset(val_ids, seq_len, seq_len)
    return model, tokenizer, val_ds, collate_fn, config


def decode_logits(model, h: torch.Tensor, loop_idx: int, num_loops: int) -> torch.Tensor:
    is_final = loop_idx == num_loops - 1
    if model.use_decode_norm and (is_final or not model.decode_norm_final_only):
        if getattr(model, "per_loop_alpha", False):
            return model.lm_head(model.decode_norm[loop_idx](h))
        return model.lm_head(model.decode_norm(h))
    return model.lm_head(h)


def token_rms(h: torch.Tensor) -> torch.Tensor:
    return h.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)


def rescale_to_token_rms(h: torch.Tensor, target_rms: torch.Tensor) -> torch.Tensor:
    scale = target_rms.to(device=h.device, dtype=h.float().dtype) / token_rms(h)
    return h * scale.to(dtype=h.dtype)


def remove_radial_update(h_new: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
    """Remove the component of h_new - h_prev parallel to h_prev.

    With u = h_prev / ||h_prev||_rms, the RMS-coordinate radial increment is
    a = <u, h_new - h_prev> / d = mean_c u_c delta_c.  Subtracting a*u removes
    the first-order radial update from the transition while preserving the
    perpendicular residual component.
    """
    prev_rms = token_rms(h_prev)
    u = h_prev.float() / prev_rms
    delta = h_new.float() - h_prev.float()
    a = (u * delta).mean(dim=-1, keepdim=True)
    corrected = h_new.float() - a * u
    return corrected.to(dtype=h_new.dtype)


def apply_recurrence_postprocess(model, h: torch.Tensor, loop_idx: int, num_loops: int) -> torch.Tensor:
    if model.inter_loop_norm:
        h = model.loop_norm(h)
    if model.use_spectral_damping and loop_idx < num_loops - 1:
        h = h * torch.sigmoid(model.raw_alpha)
    return h


@torch.no_grad()
def forward_with_clamp(
    model,
    input_ids: torch.Tensor,
    *,
    mode: str,
) -> tuple[list[torch.Tensor], list[float], list[float]]:
    """Return per-loop logits, Euclidean norms, and RMS norms under a clamp mode."""
    num_loops = model.num_loops
    _, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=input_ids.device)
    h = model.token_embedding(input_ids) + model.position_embedding(positions)
    h = model.input_dropout(h)
    causal_mask = model.causal_mask[:seq_len, :seq_len]

    logits: list[torch.Tensor] = []
    norms_l2: list[float] = []
    norms_rms: list[float] = []
    loop1_rms: torch.Tensor | None = None
    prev_loop_h: torch.Tensor | None = None

    for loop_idx in range(num_loops):
        for layer in model.layers:
            h = layer(h, causal_mask=causal_mask)

        if mode == "freeze_loop1":
            if loop1_rms is None:
                loop1_rms = token_rms(h).detach()
            else:
                h = rescale_to_token_rms(h, loop1_rms)
        elif mode == "remove_radial":
            if prev_loop_h is not None:
                h = remove_radial_update(h, prev_loop_h)
        elif mode != "none":
            raise ValueError(f"unknown clamp mode: {mode}")

        logits.append(decode_logits(model, h, loop_idx, num_loops))
        norms_l2.append(float(h.float().norm(dim=-1).mean().item()))
        norms_rms.append(float(token_rms(h).mean().item()))
        prev_loop_h = h.detach()

        h = apply_recurrence_postprocess(model, h, loop_idx, num_loops)

    return logits, norms_l2, norms_rms


def evaluate_mode(
    model,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
    *,
    mode: str,
    max_batches: int,
) -> tuple[list[dict[str, Any]], int]:
    num_loops = model.num_loops
    ce_sums = np.zeros(num_loops, dtype=np.float64)
    norm_l2_sums = np.zeros(num_loops, dtype=np.float64)
    norm_rms_sums = np.zeros(num_loops, dtype=np.float64)
    total_tokens = 0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        valid = labels.ne(tokenizer.pad_id)
        logits, norms_l2, norms_rms = forward_with_clamp(model, input_ids, mode=mode)
        total_tokens += int(valid.sum().item())

        for loop_idx, logits_k in enumerate(logits):
            ce = F.cross_entropy(
                logits_k.float().reshape(-1, logits_k.size(-1)),
                labels.reshape(-1),
                ignore_index=tokenizer.pad_id,
                reduction="sum",
            )
            ce_sums[loop_idx] += float(ce.item())
            norm_l2_sums[loop_idx] += norms_l2[loop_idx]
            norm_rms_sums[loop_idx] += norms_rms[loop_idx]
        n_batches += 1

    rows: list[dict[str, Any]] = []
    for loop_idx in range(num_loops):
        ce = ce_sums[loop_idx] / max(1, total_tokens)
        rows.append(
            {
                "mode": mode,
                "loop": loop_idx + 1,
                "ce": ce,
                "ppl": float(np.exp(ce)),
                "norm_l2": norm_l2_sums[loop_idx] / max(1, n_batches),
                "norm_rms": norm_rms_sums[loop_idx] / max(1, n_batches),
                "num_tokens": total_tokens,
                "num_batches": n_batches,
            }
        )
    return rows, total_tokens


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    modes = list(dict.fromkeys(args.mode))

    per_loop_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for spec in args.checkpoint:
        label, ckpt_path = parse_spec(spec)
        print(f"\n=== {label} ===", flush=True)
        model, tokenizer, val_ds, collate_fn, config = load_model_and_val_data(ckpt_path, device)
        loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        params = sum(p.numel() for p in model.parameters())
        mc = config.get("model", {})
        supervision = config.get("supervision", {})
        print(
            f"  device={device} params={params:,} loops={model.num_loops} "
            f"decode_norm={model.use_decode_norm} final_only={model.decode_norm_final_only} "
            f"modes={','.join(modes)}",
            flush=True,
        )

        mode_rows: dict[str, list[dict[str, Any]]] = {}
        for mode in modes:
            rows, _ = evaluate_mode(
                model,
                tokenizer,
                loader,
                device,
                mode=mode,
                max_batches=args.max_batches,
            )
            mode_rows[mode] = rows
            for row in rows:
                per_loop_rows.append(
                    {
                        "checkpoint": label,
                        "params": params,
                        "supervision": supervision.get("mode"),
                        "use_decode_norm": mc.get("use_decode_norm", True),
                        "decode_norm_final_only": mc.get("decode_norm_final_only", False),
                        **row,
                    }
                )

        baseline_final = mode_rows["none"][-1] if "none" in mode_rows else None
        for mode, rows in mode_rows.items():
            final = rows[-1]
            first = rows[0]
            delta_ce = None if baseline_final is None else final["ce"] - baseline_final["ce"]
            rel_ppl = None if baseline_final is None else final["ppl"] / baseline_final["ppl"]
            norm_ratio = None if baseline_final is None else final["norm_l2"] / max(1e-12, baseline_final["norm_l2"])
            summary_rows.append(
                {
                    "checkpoint": label,
                    "mode": mode,
                    "params": params,
                    "supervision": supervision.get("mode"),
                    "use_decode_norm": mc.get("use_decode_norm", True),
                    "decode_norm_final_only": mc.get("decode_norm_final_only", False),
                    "ppl_K1": first["ppl"],
                    f"ppl_K{model.num_loops}": final["ppl"],
                    "ce_K1": first["ce"],
                    f"ce_K{model.num_loops}": final["ce"],
                    "norm_l2_K1": first["norm_l2"],
                    f"norm_l2_K{model.num_loops}": final["norm_l2"],
                    "norm_rms_K1": first["norm_rms"],
                    f"norm_rms_K{model.num_loops}": final["norm_rms"],
                    "delta_ce_vs_none": delta_ce,
                    "rel_ppl_vs_none": rel_ppl,
                    "norm_l2_ratio_vs_none": norm_ratio,
                    "num_tokens": final["num_tokens"],
                    "num_batches": final["num_batches"],
                }
            )
            print(
                f"  {mode:12s} K1 ppl={first['ppl']:.3f} norm={first['norm_l2']:.1f} | "
                f"K{model.num_loops} ppl={final['ppl']:.3f} norm={final['norm_l2']:.1f} "
                f"delta_ce={delta_ce if delta_ce is not None else float('nan'):+.4f}",
                flush=True,
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(output_dir / "radial_clamp_per_loop.csv", per_loop_rows)
    write_csv(output_dir / "radial_clamp_summary.csv", summary_rows)
    print(f"\nSaved outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
