"""DDP training entrypoint for Ouro 1.4B from scratch.

This intentionally lives beside, rather than replacing,
``train_ouro_from_scratch.py`` so existing single-GPU jobs can keep resuming
unchanged. Launch with torchrun, for example:

    torchrun --standalone --nproc_per_node=2 src/train_ouro_from_scratch_ddp.py \
        --no-readout-norm \
        --output-dir /work/.../outputs/ouro_1.4b_no_readout_norm_ddp \
        --total-steps 50000 \
        --per-device-batch-size 8 \
        --grad-accum 3

Step semantics match the single-GPU script: ``total_steps`` counts microsteps,
and an optimizer update happens every ``grad_accum`` microsteps.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import signal
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

_SHOULD_CHECKPOINT_AND_EXIT = False


def _handle_preempt_signal(signum, _frame) -> None:
    global _SHOULD_CHECKPOINT_AND_EXIT
    _SHOULD_CHECKPOINT_AND_EXIT = True
    print(f"Received signal {signum}; will checkpoint at the next safe point.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ouro arch from scratch with DDP.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--no-readout-norm", action="store_true")
    parser.add_argument("--alpha-readout", action="store_true",
                        help="Replace final-step RMSNorm with learnable alpha-readout")
    parser.add_argument("--alpha-init", type=float, default=0.5)
    parser.add_argument("--alpha-fixed", type=float, default=None,
                        help="If set, use fixed alpha instead of learning")
    parser.add_argument("--alpha-ema-decay", type=float, default=0.99)
    parser.add_argument("--per-loop-alpha", action="store_true",
                        help="Use a different alpha-readout at each UT step "
                             "(replaces self.norm entirely; alpha learned per loop)")
    parser.add_argument("--total-steps", type=int, default=50000)
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument(
        "--lr-schedule",
        choices=["cosine", "continuation_cosine"],
        default="cosine",
        help=(
            "cosine preserves the original schedule over total_steps. "
            "continuation_cosine starts from the checkpoint optimizer LR "
            "and decays to min_lr over the remaining steps."
        ),
    )
    parser.add_argument(
        "--continuation-lr",
        type=float,
        default=None,
        help="Optional starting LR for continuation_cosine. Defaults to checkpoint optimizer LR.",
    )
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=500, help="0 disables validation.")
    parser.add_argument("--save-every", type=int, default=500, help="0 disables step-based checkpointing.")
    parser.add_argument("--save-every-minutes", type=float, default=60.0, help="0 disables time-based checkpointing.")
    parser.add_argument("--final-save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exit-on-signal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--fast-exit",
        action="store_true",
        help="After the final DDP barrier, exit without Python/C++ teardown. Useful for smoke tests on unstable stacks.",
    )
    parser.add_argument("--ddp-timeout-minutes", type=float, default=60.0)
    parser.add_argument("--ddp-static-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reentrant-checkpointing", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fineweb-subset", type=str, default="sample-10BT")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument(
        "--exact-stream-resume",
        action="store_true",
        help=(
            "Skip the approximate number of per-rank tokens in the streaming "
            "dataset on resume. Disabled by default because it is slow and "
            "not usually worth it for large-scale pretraining."
        ),
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Optional checkpoint to initialize model weights from. Does not load optimizer/step.",
    )
    parser.add_argument(
        "--init-from-hf",
        type=str,
        default=None,
        help=(
            "HuggingFace model name (e.g. 'ByteDance/Ouro-1.4B') to initialize "
            "model weights from. Loaded BEFORE any readout-norm wrapping so "
            "state-dict keys align with the pretrained checkpoint. "
            "Mutually exclusive with --init-from."
        ),
    )
    parser.add_argument(
        "--save-at-tokens",
        type=str,
        default="",
        help=(
            "Comma-separated token milestones in millions at which to save "
            "named eval snapshots (e.g. '100,200,300,400,500'). Saved as "
            "eval_step{step}_{tokens_M}M.pt next to last.pt. Does not affect "
            "last.pt (which still tracks --save-every / --save-every-minutes)."
        ),
    )
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile on the model before DDP wrapping.")
    return parser.parse_args()


def ensure_token_ids(config, tokenizer) -> None:
    """Normalize remote-code config attrs across Transformers versions."""
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(config, "pad_token_id", None) is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(config, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(config, "bos_token_id", 0)
        config.pad_token_id = pad_token_id

    if getattr(config, "eos_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        config.eos_token_id = tokenizer.eos_token_id
    if getattr(config, "bos_token_id", None) is None and getattr(tokenizer, "bos_token_id", None) is not None:
        config.bos_token_id = tokenizer.bos_token_id


def patch_transformers_rope_registry() -> None:
    """Restore the standard RoPE alias expected by Ouro remote code."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
        del seq_len, layer_type
        base = getattr(config, "rope_theta", 10000.0)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = compute_default_rope_parameters


def normalize_default_rope(config) -> None:
    """Map default RoPE to linear factor=1 for newer Transformers init code."""
    rope_scaling = getattr(config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict):
        return

    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_type != "default":
        return

    rope_parameters = getattr(config, "rope_parameters", {})
    merged_rope = {}
    if isinstance(rope_parameters, dict):
        merged_rope.update(rope_parameters)
    merged_rope.update(rope_scaling)
    merged_rope.pop("type", None)
    merged_rope["rope_type"] = "linear"
    merged_rope["factor"] = 1.0
    merged_rope["rope_theta"] = merged_rope.get("rope_theta", getattr(config, "rope_theta", 10000.0))

    config.rope_scaling = dict(merged_rope)
    if hasattr(config, "rope_parameters"):
        config.rope_parameters = dict(merged_rope)


class ConditionalNorm(nn.Module):
    """Wrap RMSNorm so the final UT/readout step is left norm-sensitive."""

    def __init__(self, norm: nn.Module, total_steps: int):
        super().__init__()
        self.norm = norm
        self.total_steps = total_steps
        self._call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._call_count += 1
        if self._call_count % self.total_steps == 0:
            return x
        return self.norm(x)

    def reset_counter(self) -> None:
        self._call_count = 0


class AlphaReadout(nn.Module):
    """Power-modulated readout: gamma(s) = s_ref^(1-alpha) * s^alpha.

    alpha=0: constant scalar s_ref (RMSNorm-without-per-channel-gain)
    alpha=1: gamma=s -> raw readout (gamma * h/s = h)

    s_ref is a learnable scalar parameter. (Earlier EMA-tracked version drifted
    with activation scale and caused alpha to collapse to 0 under terminal CE.)
    """

    def __init__(self, d_model: int, ema_decay: float = 0.99,
                 init_alpha: float = 0.5, fixed_alpha=None,
                 init_log_s_ref: float = 0.0, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.fixed_alpha = fixed_alpha
        if fixed_alpha is not None:
            assert 0.0 <= fixed_alpha <= 1.0
            self.register_buffer("alpha_value", torch.tensor(float(fixed_alpha)))
            self.register_buffer("alpha_logit", torch.tensor(0.0))
        else:
            init_a = float(min(max(init_alpha, 1e-4), 1.0 - 1e-4))
            init_logit = float(torch.logit(torch.tensor(init_a)).item())
            self.alpha_logit = nn.Parameter(torch.tensor(init_logit))
        # Learnable scalar (init log_s_ref=0 -> s_ref=1).
        self.log_s_ref = nn.Parameter(torch.tensor(float(init_log_s_ref)))

    def get_alpha(self):
        if self.fixed_alpha is not None:
            return self.alpha_value
        return torch.sigmoid(self.alpha_logit)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        s = h.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=self.eps)
        log_s = s.log()
        u = h / s
        alpha = self.get_alpha()
        log_gamma = (1.0 - alpha) * self.log_s_ref + alpha * log_s
        gamma = log_gamma.exp()
        return gamma * u


class ConditionalAlphaReadout(nn.Module):
    """For non-final UT steps: apply RMSNorm. For final UT step: apply alpha-readout."""

    def __init__(self, original_norm: nn.Module, alpha_readout: AlphaReadout,
                 total_steps: int):
        super().__init__()
        self.norm = original_norm
        self.alpha_readout = alpha_readout
        self.total_steps = total_steps
        self._call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._call_count += 1
        if self._call_count % self.total_steps == 0:
            return self.alpha_readout(x)
        return self.norm(x)

    def reset_counter(self) -> None:
        self._call_count = 0


class PerLoopAlphaReadout(nn.Module):
    """Apply a different AlphaReadout at each UT step (replaces self.norm entirely).

    With K total_steps, holds K independent AlphaReadout modules. Each UT step
    routes to its own readout based on call count. This subsumes inter-loop
    RMSNorm: setting alpha=0 fixed at intermediate steps recovers RMSNorm-like
    inter-loop normalization, while final step alpha is free to learn.
    """

    def __init__(self, d_model: int, total_steps: int,
                 init_alpha: float = 0.5, fixed_alpha=None,
                 init_log_s_ref: float = 0.0):
        super().__init__()
        self.total_steps = total_steps
        self.alpha_readouts = nn.ModuleList([
            AlphaReadout(d_model, init_alpha=init_alpha, fixed_alpha=fixed_alpha,
                         init_log_s_ref=init_log_s_ref)
            for _ in range(total_steps)
        ])
        self._call_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = self._call_count % self.total_steps
        out = self.alpha_readouts[idx](x)
        self._call_count += 1
        return out

    def reset_counter(self) -> None:
        self._call_count = 0

    def per_loop_alpha(self):
        return [float(m.get_alpha().item()) for m in self.alpha_readouts]

    def per_loop_log_s_ref(self):
        return [float(m.log_s_ref.item()) for m in self.alpha_readouts]


def ddp_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def is_main(rank: int) -> bool:
    return rank == 0


def log_main(rank: int, msg: str) -> None:
    if is_main(rank):
        print(msg, flush=True)


def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))


def get_continuation_lr(
    step: int,
    start_step: int,
    total_steps: int,
    start_lr: float,
    min_lr: float,
) -> float:
    progress = (step - start_step) / max(1, total_steps - start_step)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (start_lr - min_lr) * (1 + np.cos(np.pi * progress))


def tokenize_stream(dataset, tokenizer, seq_len: int, chunk_rank: int = 0, chunk_world_size: int = 1):
    buffer: list[int] = []
    chunk_idx = 0
    for example in dataset:
        text = example.get("text", "")
        if not text or len(text.strip()) < 50:
            continue
        tokens = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
        buffer.extend(tokens)
        while len(buffer) >= seq_len + 1:
            chunk = buffer[: seq_len + 1]
            buffer = buffer[seq_len:]
            if chunk_idx % chunk_world_size == chunk_rank:
                yield torch.tensor(chunk, dtype=torch.long)
            chunk_idx += 1


def make_train_iter(args: argparse.Namespace, tokenizer, rank: int, world_size: int):
    from datasets import load_dataset

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name=args.fineweb_subset,
        split="train",
        streaming=True,
    )
    if args.shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    chunk_rank, chunk_world_size = 0, 1
    if world_size > 1:
        try:
            dataset = dataset.shard(num_shards=world_size, index=rank)
        except Exception as exc:
            log_main(
                rank,
                f"  Warning: dataset.shard failed ({exc}); falling back to chunk-strided sharding.",
            )
            chunk_rank, chunk_world_size = rank, world_size
    return tokenize_stream(dataset, tokenizer, args.seq_len, chunk_rank, chunk_world_size)


def load_validation(args: argparse.Namespace, tokenizer, rank: int):
    from datasets import load_dataset

    if not is_main(rank):
        return None
    val_dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    val_texts = [text for text in val_dataset["text"] if len(text.strip()) > 50]
    val_text = "\n\n".join(val_texts[:500])
    val_tokens = tokenizer(val_text, return_tensors="pt", truncation=False)["input_ids"][0]
    val_chunks = [
        val_tokens[i : i + args.seq_len + 1]
        for i in range(0, len(val_tokens) - args.seq_len - 1, args.seq_len)
    ]
    return torch.stack(val_chunks)


def evaluate(model, val_chunks, device: torch.device, max_batches: int, reset_norm=None) -> tuple[float, float]:
    model.eval()
    ce_sum, total = 0.0, 0
    with torch.no_grad():
        for i in range(0, min(len(val_chunks), max_batches * 4), 4):
            if reset_norm is not None:
                reset_norm()
            batch = val_chunks[i : i + 4].to(device, non_blocking=True)
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]
            outputs = model(input_ids=input_ids, labels=labels)
            ce_sum += outputs.loss.item() * labels.numel()
            total += labels.numel()
    model.train()
    loss = ce_sum / max(1, total)
    return loss, float(np.exp(loss))


def skip_stream_tokens(train_iter, tokens_to_skip: int, seq_len: int, rank: int) -> None:
    chunks_to_skip = max(0, tokens_to_skip // seq_len)
    if chunks_to_skip == 0:
        return
    log_main(rank, f"  Exact stream resume: skipping ~{tokens_to_skip / 1e6:.1f}M local tokens")
    for i in range(chunks_to_skip):
        try:
            next(train_iter)
        except StopIteration:
            break
        if is_main(rank) and (i + 1) % 10_000 == 0:
            print(f"    skipped {(i + 1) * seq_len / 1e6:.1f}M local tokens", flush=True)


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "betas": (0.9, 0.95),
    }
    if args.fused_adamw:
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(model.parameters(), **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return torch.optim.AdamW(model.parameters(), **kwargs)


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    log: dict,
    args: argparse.Namespace,
    world_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_dir / f".last.pt.tmp.rank0.pid{os.getpid()}"
    final_path = output_dir / "last.pt"
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "tokens_seen": tokens_seen,
            "log": log,
            "config": {
                "no_readout_norm": args.no_readout_norm,
                "alpha_readout": args.alpha_readout,
                "alpha_init": args.alpha_init,
                "alpha_fixed": args.alpha_fixed,
                "alpha_ema_decay": args.alpha_ema_decay,
                "lr": args.lr,
                "min_lr": args.min_lr,
                "lr_schedule": args.lr_schedule,
                "continuation_lr": args.continuation_lr,
                "total_steps": args.total_steps,
                "seed": args.seed,
                "world_size": world_size,
                "per_device_batch_size": args.per_device_batch_size,
                "grad_accum": args.grad_accum,
                "seq_len": args.seq_len,
                "final_save": args.final_save,
                "ddp_timeout_minutes": args.ddp_timeout_minutes,
                "ddp_static_graph": args.ddp_static_graph,
                "gradient_checkpointing": args.gradient_checkpointing,
                "reentrant_checkpointing": args.reentrant_checkpointing,
            },
        },
        tmp_path,
    )
    os.replace(tmp_path, final_path)


def save_milestone_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    log: dict,
    args: argparse.Namespace,
    world_size: int,
    milestone_M: int,
) -> None:
    """Save a named milestone snapshot (e.g. eval_step1234_100M.pt).

    Same payload as save_checkpoint() but written to a unique filename so it
    does not overwrite last.pt. Used by --save-at-tokens.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    final_name = f"eval_step{step}_{milestone_M}M.pt"
    tmp_path = output_dir / f".{final_name}.tmp.rank0.pid{os.getpid()}"
    final_path = output_dir / final_name
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "tokens_seen": tokens_seen,
            "log": log,
            "config": {
                "no_readout_norm": args.no_readout_norm,
                "alpha_readout": args.alpha_readout,
                "alpha_init": args.alpha_init,
                "alpha_fixed": args.alpha_fixed,
                "alpha_ema_decay": args.alpha_ema_decay,
                "lr": args.lr,
                "min_lr": args.min_lr,
                "lr_schedule": args.lr_schedule,
                "continuation_lr": args.continuation_lr,
                "total_steps": args.total_steps,
                "seed": args.seed,
                "world_size": world_size,
                "per_device_batch_size": args.per_device_batch_size,
                "grad_accum": args.grad_accum,
                "seq_len": args.seq_len,
                "init_from_hf": args.init_from_hf,
                "milestone_M": milestone_M,
            },
        },
        tmp_path,
    )
    os.replace(tmp_path, final_path)


def write_checkpoint_marker(output_dir: Path, step: int) -> None:
    marker_path = output_dir / f".checkpoint_step_{step}.done"
    tmp_path = output_dir / f".checkpoint_step_{step}.done.tmp.rank0.pid{os.getpid()}"
    tmp_path.write_text(str(time.time()))
    os.replace(tmp_path, marker_path)


def wait_for_checkpoint_marker(output_dir: Path, step: int, timeout_seconds: float = 1800.0) -> None:
    marker_path = output_dir / f".checkpoint_step_{step}.done"
    deadline = time.time() + timeout_seconds
    while not marker_path.exists():
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for checkpoint marker {marker_path}")
        time.sleep(2.0)


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGTERM, _handle_preempt_signal)
    signal.signal(signal.SIGUSR1, _handle_preempt_signal)
    rank, local_rank, world_size = ddp_info()
    distributed = world_size > 1

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=args.ddp_timeout_minutes))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    if is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    log_main(rank, "Loading Ouro architecture (random init)...")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    patch_transformers_rope_registry()

    model_name = "ByteDance/Ouro-1.4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    ensure_token_ids(config, tokenizer)
    normalize_default_rope(config)
    config.use_cache = False

    raw_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    raw_model = raw_model.to(dtype=torch.bfloat16, device=device)
    raw_model.train()

    if args.init_from is not None and args.init_from_hf is not None:
        raise ValueError("Use either --init-from or --init-from-hf, not both.")

    if args.init_from_hf is not None:
        log_main(rank, f"Initializing model weights from HuggingFace: {args.init_from_hf}")
        log_main(rank, "  (rank 0 loads, then broadcasts to all ranks)")
        # Multi-node note: only rank 0 reads from the HF cache (which may live
        # on shared NFS). Other ranks keep their random init and receive the
        # weights via NCCL broadcast, which is fast and avoids 4 concurrent
        # NFS reads of the same 3GB safetensors that would desync the ranks
        # and trip NCCL heartbeat timeout at the next collective.
        if rank == 0:
            hf_pretrained = AutoModelForCausalLM.from_pretrained(
                args.init_from_hf,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            hf_state = hf_pretrained.state_dict()
            missing, unexpected = raw_model.load_state_dict(hf_state, strict=False)
            if missing:
                log_main(rank, f"  WARNING: {len(missing)} missing keys when loading HF weights, e.g. {missing[:3]}")
            if unexpected:
                log_main(rank, f"  WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
            if not missing and not unexpected:
                log_main(rank, "  HF state_dict loaded cleanly on rank 0 (0 missing, 0 unexpected)")
            del hf_pretrained, hf_state
            torch.cuda.empty_cache()
        if distributed:
            dist.barrier()
            for p in raw_model.parameters():
                dist.broadcast(p.data, src=0)
            for b in raw_model.buffers():
                dist.broadcast(b.data, src=0)
            log_main(rank, "  Broadcast HF weights from rank 0 to all ranks")
        raw_model = raw_model.to(device)
        raw_model.train()

    total_ut_steps = getattr(config, "total_ut_steps", 4)
    params = sum(p.numel() for p in raw_model.parameters())
    log_main(rank, f"  params={params:,}")
    log_main(rank, f"  total_ut_steps={total_ut_steps}")
    log_main(rank, f"  hidden_size={config.hidden_size}")
    log_main(rank, f"  num_layers={config.num_hidden_layers}")
    log_main(rank, f"  no_readout_norm={args.no_readout_norm}")
    log_main(rank, f"  alpha_readout={args.alpha_readout} fixed={args.alpha_fixed} init={args.alpha_init}")
    log_main(rank, f"  world_size={world_size}, local_rank={local_rank}")

    reset_fn = None
    if args.alpha_readout:
        if args.per_loop_alpha:
            raw_model.model.norm = PerLoopAlphaReadout(
                d_model=config.hidden_size,
                total_steps=total_ut_steps,
                init_alpha=args.alpha_init,
                fixed_alpha=args.alpha_fixed,
            ).to(device)
            reset_fn = lambda: raw_model.model.norm.reset_counter()
            log_main(rank, f"  Per-loop alpha-readout enabled "
                           f"({total_ut_steps} independent readouts, init={args.alpha_init})")
        else:
            original_norm = raw_model.model.norm
            alpha_module = AlphaReadout(
                d_model=config.hidden_size,
                ema_decay=args.alpha_ema_decay,
                init_alpha=args.alpha_init,
                fixed_alpha=args.alpha_fixed,
            ).to(device)
            raw_model.model.norm = ConditionalAlphaReadout(
                original_norm, alpha_module, total_ut_steps
            ).to(device)
            reset_fn = lambda: raw_model.model.norm.reset_counter()
            if args.alpha_fixed is not None:
                log_main(rank, f"  Alpha-readout enabled (fixed alpha={args.alpha_fixed})")
            else:
                log_main(rank, f"  Alpha-readout enabled (learned, init={args.alpha_init})")
    elif args.no_readout_norm:
        original_norm = raw_model.model.norm
        raw_model.model.norm = ConditionalNorm(original_norm, total_ut_steps).to(device)
        reset_fn = lambda: raw_model.model.norm.reset_counter()
        log_main(rank, "  Readout norm REMOVED on final UT step")
    else:
        log_main(rank, "  Readout norm KEPT")

    if args.gradient_checkpointing:
        if args.reentrant_checkpointing:
            raw_model.gradient_checkpointing_enable()
            log_main(rank, "  Gradient checkpointing enabled (reentrant)")
        else:
            try:
                raw_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                log_main(rank, "  Gradient checkpointing enabled (non-reentrant)")
            except TypeError:
                raw_model.gradient_checkpointing_enable()
                log_main(rank, "  Gradient checkpointing enabled (default reentrant)")

    if args.init_from:
        init_path = Path(args.init_from)
        log_main(rank, f"Initializing model weights from {init_path}")
        log_main(rank, "  WARNING: --init-from is weight-only. Optimizer state, "
                       "step counter, tokens_seen, LR schedule position, and "
                       "stream position are all reset. For a true continuation "
                       "(resume mid-run), copy the source last.pt into "
                       f"{output_dir}/last.pt before launching; the script will "
                       "then auto-resume with full state.")
        ckpt = torch.load(init_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state"])

    if args.compile:
        log_main(rank, "Compiling model with torch.compile...")
        raw_model = torch.compile(raw_model)

    model = raw_model
    if distributed:
        ddp_kwargs = {
            "device_ids": [local_rank],
            "output_device": local_rank,
            "gradient_as_bucket_view": True,
            "find_unused_parameters": False,
        }
        ddp_supports_static_graph = "static_graph" in inspect.signature(DDP).parameters
        if ddp_supports_static_graph:
            ddp_kwargs["static_graph"] = args.ddp_static_graph
        model = DDP(raw_model, **ddp_kwargs)
        if args.ddp_static_graph and not ddp_supports_static_graph and hasattr(model, "_set_static_graph"):
            model._set_static_graph()

    log_main(rank, f"Loading FineWeb ({args.fineweb_subset}) streaming...")
    train_iter = make_train_iter(args, tokenizer, rank, world_size)

    log_main(rank, "Loading WikiText-103 validation on rank 0...")
    val_chunks = load_validation(args, tokenizer, rank)
    if val_chunks is not None:
        log_main(rank, f"  val chunks: {len(val_chunks)}")

    optimizer = make_optimizer(raw_model, args)

    start_step = 0
    tokens_seen = 0
    checkpoint_lr = args.lr
    log = {
        "steps": [],
        "train_loss": [],
        "val_loss": [],
        "val_ppl": [],
        "tokens_seen": [],
        "wall_time": [],
        "interval_tok_s": [],
    }

    ckpt_path = output_dir / "last.pt"
    if ckpt_path.exists():
        log_main(rank, f"Resuming from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        legacy_keys = [k for k in ckpt["model_state"] if k.endswith(".ema_count")]
        for k in legacy_keys:
            ckpt["model_state"].pop(k)
        if legacy_keys:
            log_main(rank, f"  Dropped {len(legacy_keys)} legacy EMA buffer key(s) from checkpoint")
        raw_model.load_state_dict(ckpt["model_state"])
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except ValueError as e:
            log_main(rank, f"  Optimizer state mismatch ({e}); resetting Adam moments and continuing.")
        start_step = int(ckpt["step"])
        tokens_seen = int(ckpt["tokens_seen"])
        checkpoint_lr = float(optimizer.param_groups[0].get("lr", args.lr))
        log = ckpt.get("log", log)
        if args.exact_stream_resume:
            skip_stream_tokens(train_iter, tokens_seen // max(1, world_size), args.seq_len, rank)
        else:
            log_main(rank, "  Fast resume: model/optimizer restored; streaming data starts at a fresh shard.")

    if distributed:
        dist.barrier()

    if start_step == 0 and args.eval_every != 0 and is_main(rank):
        val_loss, val_ppl = evaluate(raw_model, val_chunks, device, args.eval_batches, reset_norm=reset_fn)
        log_main(rank, f"  Initial val_ppl={val_ppl:.2f}")
    if distributed:
        dist.barrier()

    global_batch = args.per_device_batch_size * world_size * args.grad_accum
    tokens_per_microstep = args.per_device_batch_size * world_size * (args.seq_len - 1)
    continuation_start_lr = args.continuation_lr
    if continuation_start_lr is None:
        continuation_start_lr = checkpoint_lr if start_step > 0 else args.lr
    log_main(rank, f"\nTraining: {args.total_steps} microsteps, global effective batch={global_batch}")
    log_main(rank, f"  tokens/microstep={tokens_per_microstep:,}")
    if args.lr_schedule == "continuation_cosine":
        log_main(
            rank,
            f"  LR schedule=continuation_cosine, start_step={start_step}, "
            f"start_lr={continuation_start_lr:.3e} -> {args.min_lr:.3e}",
        )
    else:
        log_main(rank, f"  LR schedule=cosine, LR={args.lr} -> {args.min_lr}, warmup={args.warmup_steps}")
    log_main(rank, "-" * 80)

    # Parse --save-at-tokens milestones (in millions). Track which have already
    # fired so a single milestone never saves twice across the run, including
    # across resumes (any milestone <= current tokens_seen is treated as done).
    milestone_targets_M: list[int] = []
    if args.save_at_tokens:
        try:
            milestone_targets_M = sorted(
                {int(x.strip()) for x in args.save_at_tokens.split(",") if x.strip()}
            )
        except ValueError as e:
            raise ValueError(f"--save-at-tokens must be comma-separated integers (millions): {e}")
    milestones_done: set[int] = {m for m in milestone_targets_M if tokens_seen >= m * 1_000_000}
    if milestone_targets_M:
        log_main(
            rank,
            f"Token milestones (M): {milestone_targets_M}; "
            f"already passed at resume: {sorted(milestones_done)}",
        )

    running_loss = 0.0
    t0 = time.time()
    last_log_time = t0
    last_log_tokens = tokens_seen
    last_save_time = t0
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step + 1, args.total_steps + 1):
        if args.lr_schedule == "continuation_cosine":
            lr = get_continuation_lr(step, start_step, args.total_steps, continuation_start_lr, args.min_lr)
        else:
            lr = get_lr(step, args.warmup_steps, args.total_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if reset_fn is not None:
            reset_fn()

        batch_chunks = []
        for _ in range(args.per_device_batch_size):
            try:
                chunk = next(train_iter)
            except StopIteration:
                train_iter = make_train_iter(args, tokenizer, rank, world_size)
                chunk = next(train_iter)
            batch_chunks.append(chunk)

        batch = torch.stack(batch_chunks).to(device, non_blocking=True)
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        tokens_seen += input_ids.numel() * world_size

        sync_grad = step % args.grad_accum == 0
        use_no_sync = distributed and not args.ddp_static_graph
        sync_context = nullcontext() if sync_grad or not use_no_sync else model.no_sync()
        with sync_context:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            running_loss += loss.item() * args.grad_accum

        if sync_grad:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            now = time.time()
            interval_tokens = tokens_seen - last_log_tokens
            interval_tps = interval_tokens / max(1e-6, now - last_log_time)
            elapsed = now - t0
            avg_loss = running_loss / args.log_every
            running_loss = 0.0
            last_log_time = now
            last_log_tokens = tokens_seen
            log_main(
                rank,
                f"step={step:6d} | loss={avg_loss:.4f} | lr={lr:.2e} | "
                f"tokens={tokens_seen / 1e6:.1f}M | {interval_tps / 1000:.1f}k tok/s | "
                f"{elapsed / 60:.1f}min",
            )

        if args.eval_every > 0 and step % args.eval_every == 0:
            if distributed:
                dist.barrier()
            if is_main(rank):
                val_loss, val_ppl = evaluate(raw_model, val_chunks, device, args.eval_batches, reset_norm=reset_fn)
                elapsed = time.time() - t0
                log["steps"].append(step)
                log["val_loss"].append(val_loss)
                log["val_ppl"].append(val_ppl)
                log["tokens_seen"].append(tokens_seen)
                log["wall_time"].append(elapsed)
                log_main(rank, f"  >>> EVAL step={step}: val_ppl={val_ppl:.2f} | tokens={tokens_seen / 1e6:.1f}M")
            if distributed:
                dist.barrier()

        signal_save_due = _SHOULD_CHECKPOINT_AND_EXIT
        if distributed:
            signal_tensor = torch.tensor([1 if signal_save_due else 0], device=device, dtype=torch.int32)
            dist.all_reduce(signal_tensor, op=dist.ReduceOp.MAX)
            signal_save_due = bool(signal_tensor.item())

        now = time.time()
        step_save_due = args.save_every > 0 and step % args.save_every == 0
        time_save_due = args.save_every_minutes > 0 and (now - last_save_time) > args.save_every_minutes * 60
        # Only save at optimizer-boundary (sync_grad) so we don't drop pending
        # microsteps. step/tokens_seen advance per microstep but the optimizer
        # only updates every args.grad_accum microsteps; saving mid-window means
        # resume silently loses up to grad_accum-1 microsteps of training.
        save_due = (step_save_due or time_save_due or signal_save_due) and sync_grad
        if distributed:
            save_tensor = torch.tensor([1 if save_due else 0], device=device, dtype=torch.int32)
            dist.all_reduce(save_tensor, op=dist.ReduceOp.MAX)
            save_due = bool(save_tensor.item())

        if is_main(rank) and save_due:
            save_checkpoint(output_dir, raw_model, optimizer, step, tokens_seen, log, args, world_size)
            write_checkpoint_marker(output_dir, step)
            log_main(rank, f"  Saved checkpoint at step {step}")
        elif save_due:
            wait_for_checkpoint_marker(output_dir, step)
        if save_due:
            last_save_time = time.time()

        # Token-milestone snapshots (separate file, does not touch last.pt).
        # Only fire on optimizer-boundary steps so we don't save mid-grad-accum.
        if sync_grad and milestone_targets_M:
            crossed_now = [
                m for m in milestone_targets_M
                if m not in milestones_done and tokens_seen >= m * 1_000_000
            ]
            if crossed_now:
                # Take the smallest crossed milestone in this iteration to keep
                # the filename deterministic; mark all crossed targets done.
                m = crossed_now[0]
                if distributed:
                    dist.barrier()
                if is_main(rank):
                    save_milestone_checkpoint(
                        output_dir, raw_model, optimizer, step, tokens_seen,
                        log, args, world_size, m,
                    )
                    log_main(
                        rank,
                        f"  Saved milestone snapshot eval_step{step}_{m}M.pt "
                        f"(tokens_seen={tokens_seen / 1e6:.1f}M)",
                    )
                if distributed:
                    dist.barrier()
                for mm in crossed_now:
                    milestones_done.add(mm)

        if signal_save_due and args.exit_on_signal and save_due:
            # Only exit after a successful boundary save. If signal arrived
            # mid-grad-accum, defer exit until the next sync_grad iteration.
            log_main(rank, f"Exiting after signal-triggered checkpoint at step {step}.")
            if distributed:
                if args.fast_exit:
                    os._exit(0)
                dist.destroy_process_group()
            # Exit code 0 (not 128+SIGTERM=143) so Slurm afterok chains continue
            # after a successful checkpoint. The signal was handled cleanly.
            sys.exit(0)

    if is_main(rank):
        if args.final_save:
            save_checkpoint(output_dir, raw_model, optimizer, step, tokens_seen, log, args, world_size)
        if args.eval_every != 0:
            val_loss, val_ppl = evaluate(raw_model, val_chunks, device, args.eval_batches, reset_norm=reset_fn)
            log_main(rank, f"\nDone. {tokens_seen / 1e9:.2f}B tokens")
            log_main(rank, f"Final val_ppl={val_ppl:.2f}")
        with (output_dir / "train_log.json").open("w") as f:
            json.dump(log, f, indent=2)

    if distributed:
        dist.barrier()
        if args.fast_exit:
            os._exit(0)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
