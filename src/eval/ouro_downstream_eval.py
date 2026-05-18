"""Downstream multiple-choice evaluation for scratch-trained Ouro checkpoints.

The checkpoint format is produced by ``train_ouro_from_scratch_ddp.py``. This
script loads the raw model state, applies the same readout-norm patch recorded
in the checkpoint config, and scores answer choices by continuation
log-likelihood.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from train_ouro_from_scratch_ddp import (  # noqa: E402
    AlphaReadout,
    ConditionalAlphaReadout,
    ConditionalNorm,
    PerLoopAlphaReadout,
    ensure_token_ids,
    normalize_default_rope,
    patch_transformers_rope_registry,
)


@dataclass(frozen=True)
class MCExample:
    task: str
    idx: int
    prompt: str
    choices: list[str]
    label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Ouro checkpoint on MC downstream tasks.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to train_ouro_from_scratch_ddp checkpoint. Required unless "
            "--use-hf-only is set (in which case the published HF weights are "
            "evaluated directly without any of our fine-tuning)."
        ),
    )
    parser.add_argument("--model", default="ByteDance/Ouro-1.4B")
    parser.add_argument(
        "--use-hf-only",
        action="store_true",
        help=(
            "Evaluate the published HuggingFace --model directly (no checkpoint "
            "load, no readout-norm wrapping). Used to establish the raw baseline "
            "before our fine-tuning runs."
        ),
    )
    parser.add_argument(
        "--tasks",
        default="piqa,arc_easy,arc_challenge,hellaswag",
        help=(
            "Comma-separated tasks: piqa, arc_easy, arc_challenge, hellaswag, "
            "openbookqa, boolq, winogrande, commonsenseqa, sciq."
        ),
    )
    parser.add_argument("--max-samples-per-task", type=int, default=250, help="0 means full validation set.")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--output-dir", default="outputs/ouro_downstream_eval")
    parser.add_argument("--results-name", default=None)
    parser.add_argument(
        "--readout-norm-mode",
        choices=["checkpoint", "no_norm", "baseline"],
        default="checkpoint",
        help="Default uses checkpoint config. Override only for diagnostics.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "bf16" or (dtype_arg == "auto" and device.type == "cuda"):
        return torch.bfloat16
    if dtype_arg == "fp16":
        return torch.float16
    return torch.float32


def load_torch_checkpoint(path: Path) -> dict[str, Any]:
    kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)
    except RuntimeError:
        return torch.load(path, **kwargs)


def should_remove_readout_norm(args: argparse.Namespace, ckpt: dict[str, Any]) -> bool:
    if args.readout_norm_mode == "no_norm":
        return True
    if args.readout_norm_mode == "baseline":
        return False
    return bool(ckpt.get("config", {}).get("no_readout_norm", False))


def load_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    use_hf_only = bool(getattr(args, "use_hf_only", False))
    if not use_hf_only and args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --use-hf-only is set")

    patch_transformers_rope_registry()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    ensure_token_ids(config, tokenizer)
    normalize_default_rope(config)
    config.use_cache = False

    if use_hf_only:
        print(f"Loading published HF weights {args.model} (no checkpoint, no wrap)...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=config, trust_remote_code=True, torch_dtype=dtype,
        )
        model = model.to(dtype=dtype).to(device)
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        metadata = {
            "checkpoint": None,
            "step": -1,
            "tokens_seen": -1,
            "tokens_seen_m": None,
            "checkpoint_config": {},
            "model": args.model,
            "readout_norm_mode": "hf_pretrained",
            "dtype": str(dtype).replace("torch.", ""),
            "device": str(device),
        }
        print(
            f"Loaded HF model {args.model}, "
            f"params={sum(p.numel() for p in model.parameters()):,}",
            flush=True,
        )
        return model, tokenizer, None, metadata

    checkpoint_path = Path(args.checkpoint)
    print(f"Loading checkpoint metadata/state from {checkpoint_path}", flush=True)
    ckpt = load_torch_checkpoint(checkpoint_path)

    print(f"Building {args.model} on CPU...", flush=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = model.to(dtype=dtype)

    ckpt_config = dict(ckpt.get("config", {}))
    is_alpha_readout = bool(ckpt_config.get("alpha_readout", False))
    is_per_loop_alpha = bool(ckpt_config.get("per_loop_alpha", False))
    total_ut_steps = int(getattr(config, "total_ut_steps", 4))
    reset_norm = None
    no_readout_norm = False

    if is_alpha_readout:
        d_model = int(getattr(config, "hidden_size", model.model.norm.weight.shape[0]))
        alpha_init = float(ckpt_config.get("alpha_init", 0.5))
        alpha_fixed = ckpt_config.get("alpha_fixed", None)
        init_log_s_ref = float(ckpt_config.get("alpha_init_log_s_ref", 0.0))
        if is_per_loop_alpha:
            new_norm = PerLoopAlphaReadout(
                d_model=d_model,
                total_steps=total_ut_steps,
                init_alpha=alpha_init,
                fixed_alpha=alpha_fixed,
                init_log_s_ref=init_log_s_ref,
            )
        else:
            alpha_module = AlphaReadout(
                d_model=d_model,
                init_alpha=alpha_init,
                fixed_alpha=alpha_fixed,
                init_log_s_ref=init_log_s_ref,
            )
            new_norm = ConditionalAlphaReadout(
                original_norm=model.model.norm,
                alpha_readout=alpha_module,
                total_steps=total_ut_steps,
            )
        new_norm = new_norm.to(dtype=dtype)
        model.model.norm = new_norm
        reset_norm = lambda: model.model.norm.reset_counter()
        print(
            f"Readout norm mode: alpha_readout per_loop={is_per_loop_alpha} "
            f"init_alpha={alpha_init} fixed={alpha_fixed}",
            flush=True,
        )
    else:
        no_readout_norm = should_remove_readout_norm(args, ckpt)
        if no_readout_norm:
            model.model.norm = ConditionalNorm(model.model.norm, total_ut_steps)
            reset_norm = lambda: model.model.norm.reset_counter()
            print("Readout norm mode: no_norm", flush=True)
        else:
            print("Readout norm mode: baseline", flush=True)

    legacy_keys = [k for k in ckpt["model_state"] if k.endswith(".ema_count")]
    for k in legacy_keys:
        ckpt["model_state"].pop(k)
    if legacy_keys:
        print(f"Dropped {len(legacy_keys)} legacy EMA buffer key(s) from checkpoint", flush=True)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Unexpected checkpoint load result: missing={missing}, unexpected={unexpected}")

    step = int(ckpt.get("step", -1))
    tokens_seen = int(ckpt.get("tokens_seen", -1))
    del ckpt

    model = model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    metadata = {
        "checkpoint": str(checkpoint_path),
        "step": step,
        "tokens_seen": tokens_seen,
        "tokens_seen_m": tokens_seen / 1e6 if tokens_seen >= 0 else None,
        "checkpoint_config": ckpt_config,
        "model": args.model,
        "readout_norm_mode": (
            f"alpha_readout{'_per_loop' if is_per_loop_alpha else ''}"
            if is_alpha_readout
            else ("no_norm" if no_readout_norm else "baseline")
        ),
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
    }
    print(
        f"Loaded checkpoint step={step}, tokens={tokens_seen / 1e6:.1f}M, "
        f"params={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    return model, tokenizer, reset_norm, metadata


def task_names(tasks_arg: str) -> list[str]:
    names = [name.strip().lower() for name in tasks_arg.split(",") if name.strip()]
    unknown = sorted(set(names) - set(TASK_LOADERS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Available: {sorted(TASK_LOADERS)}")
    return names


def load_piqa(max_samples: int, sample_offset: int) -> list[MCExample]:
    from datasets import load_dataset

    # The canonical `piqa` HF repo is a legacy dataset script, which recent
    # `datasets` versions reject. This mirror exposes the validation data as
    # Parquet with the same goal/sol1/sol2/label schema.
    dataset = load_dataset("gimmaru/piqa", split="validation")
    examples: list[MCExample] = []
    for row_idx, row in enumerate(dataset):
        label = int(row["label"])
        examples.append(
            MCExample(
                task="piqa",
                idx=row_idx,
                prompt=f"Question: {row['goal']}\nAnswer:",
                choices=[f" {row['sol1']}", f" {row['sol2']}"],
                label=label,
            )
        )
    return slice_examples(examples, max_samples, sample_offset)


def load_arc(config_name: str, task: str, max_samples: int, sample_offset: int) -> list[MCExample]:
    from datasets import load_dataset

    dataset = load_dataset("ai2_arc", config_name, split="validation")
    examples: list[MCExample] = []
    for row_idx, row in enumerate(dataset):
        choice_labels = [str(label) for label in row["choices"]["label"]]
        choice_texts = [str(text) for text in row["choices"]["text"]]
        answer = str(row["answerKey"])
        if answer not in choice_labels:
            continue
        examples.append(
            MCExample(
                task=task,
                idx=row_idx,
                prompt=f"Question: {row['question']}\nAnswer:",
                choices=[f" {text}" for text in choice_texts],
                label=choice_labels.index(answer),
            )
        )
    return slice_examples(examples, max_samples, sample_offset)


def load_hellaswag(max_samples: int, sample_offset: int) -> list[MCExample]:
    from datasets import load_dataset

    dataset = load_dataset("Rowan/hellaswag", split="validation")
    examples: list[MCExample] = []
    for row_idx, row in enumerate(dataset):
        label_raw = row.get("label", "")
        if label_raw in ("", None):
            continue
        ctx_b = str(row.get("ctx_b", ""))
        if ctx_b:
            ctx_b = ctx_b[:1].upper() + ctx_b[1:]
        prompt = f"{row.get('ctx_a', '')} {ctx_b}".strip()
        examples.append(
            MCExample(
                task="hellaswag",
                idx=row_idx,
                prompt=prompt,
                choices=[f" {ending}" for ending in row["endings"]],
                label=int(label_raw),
            )
        )
    return slice_examples(examples, max_samples, sample_offset)


def load_openbookqa(max_samples: int, sample_offset: int) -> list[MCExample]:
    from datasets import load_dataset

    dataset = load_dataset("openbookqa", "main", split="validation")
    examples: list[MCExample] = []
    for row_idx, row in enumerate(dataset):
        labels = [str(label) for label in row["choices"]["label"]]
        texts = [str(text) for text in row["choices"]["text"]]
        answer = str(row["answerKey"])
        if answer not in labels:
            continue
        examples.append(
            MCExample(
                task="openbookqa",
                idx=row_idx,
                prompt=f"Question: {row['question_stem']}\nAnswer:",
                choices=[f" {text}" for text in texts],
                label=labels.index(answer),
            )
        )
    return slice_examples(examples, max_samples, sample_offset)


def load_boolq(max_samples: int, sample_offset: int) -> list[MCExample]:
    from datasets import load_dataset

    dataset = load_dataset("super_glue", "boolq", split="validation")
    examples: list[MCExample] = []
    for row_idx, row in enumerate(dataset):
        examples.append(
            MCExample(
                task="boolq",
                idx=row_idx,
                prompt=f"Passage: {row['passage']}\nQuestion: {row['question']}?\nAnswer:",
                choices=[" no", " yes"],
                label=int(row["label"]),
            )
        )
    return slice_examples(examples, max_samples, sample_offset)


def load_winogrande(max_samples: int, sample_offset: int) -> list[MCExample]:
    from datasets import load_dataset

    dataset = load_dataset("winogrande", "winogrande_xl", split="validation")
    examples: list[MCExample] = []
    for row_idx, row in enumerate(dataset):
        sentence = str(row["sentence"])
        choices = [
            " " + sentence.replace("_", str(row["option1"])),
            " " + sentence.replace("_", str(row["option2"])),
        ]
        examples.append(
            MCExample(
                task="winogrande",
                idx=row_idx,
                prompt="Complete the sentence:",
                choices=choices,
                label=int(row["answer"]) - 1,
            )
        )
    return slice_examples(examples, max_samples, sample_offset)


def load_commonsenseqa(max_samples: int, sample_offset: int) -> list[MCExample]:
    from datasets import load_dataset

    dataset = load_dataset("commonsense_qa", split="validation")
    examples: list[MCExample] = []
    for row_idx, row in enumerate(dataset):
        labels = [str(label) for label in row["choices"]["label"]]
        texts = [str(text) for text in row["choices"]["text"]]
        answer = str(row["answerKey"])
        if answer not in labels:
            continue
        examples.append(
            MCExample(
                task="commonsenseqa",
                idx=row_idx,
                prompt=f"Question: {row['question']}\nAnswer:",
                choices=[f" {text}" for text in texts],
                label=labels.index(answer),
            )
        )
    return slice_examples(examples, max_samples, sample_offset)


def load_sciq(max_samples: int, sample_offset: int) -> list[MCExample]:
    from datasets import load_dataset

    dataset = load_dataset("sciq", split="validation")
    examples: list[MCExample] = []
    for row_idx, row in enumerate(dataset):
        choices = [
            str(row["correct_answer"]),
            str(row["distractor1"]),
            str(row["distractor2"]),
            str(row["distractor3"]),
        ]
        # Deterministic per-example rotation prevents always placing the correct
        # answer first while keeping exact reproducibility.
        shift = row_idx % len(choices)
        rotated = choices[shift:] + choices[:shift]
        label = rotated.index(str(row["correct_answer"]))
        examples.append(
            MCExample(
                task="sciq",
                idx=row_idx,
                prompt=f"Question: {row['question']}\nAnswer:",
                choices=[f" {choice}" for choice in rotated],
                label=label,
            )
        )
    return slice_examples(examples, max_samples, sample_offset)


def slice_examples(examples: list[MCExample], max_samples: int, sample_offset: int) -> list[MCExample]:
    if sample_offset:
        examples = examples[sample_offset:]
    if max_samples > 0:
        examples = examples[:max_samples]
    return examples


TASK_LOADERS: dict[str, Callable[[int, int], list[MCExample]]] = {
    "piqa": load_piqa,
    "arc_easy": lambda max_samples, sample_offset: load_arc("ARC-Easy", "arc_easy", max_samples, sample_offset),
    "arc_challenge": lambda max_samples, sample_offset: load_arc(
        "ARC-Challenge", "arc_challenge", max_samples, sample_offset
    ),
    "hellaswag": load_hellaswag,
    "openbookqa": load_openbookqa,
    "boolq": load_boolq,
    "winogrande": load_winogrande,
    "commonsenseqa": load_commonsenseqa,
    "sciq": load_sciq,
}


def prepare_choice_tensors(
    tokenizer,
    prompt: str,
    choices: list[str],
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    rows: list[tuple[list[int], list[int], list[bool]]] = []
    for choice in choices:
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        continuation_ids = tokenizer(choice, add_special_tokens=False)["input_ids"]
        if not continuation_ids:
            continuation_ids = [pad_id]

        if len(continuation_ids) >= max_length:
            continuation_ids = continuation_ids[: max_length - 1]
        max_prompt_tokens = max(1, max_length - len(continuation_ids))
        if len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[-max_prompt_tokens:]

        full_ids = prompt_ids + continuation_ids
        if len(full_ids) < 2:
            bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else pad_id
            full_ids = [bos_id] + full_ids

        continuation_start = len(prompt_ids)
        input_ids = full_ids[:-1]
        labels = full_ids[1:]
        mask_start = max(0, continuation_start - 1)
        choice_mask = [position >= mask_start for position in range(len(labels))]
        rows.append((input_ids, labels, choice_mask))

    batch_size = len(rows)
    seq_len = max(len(input_ids) for input_ids, _, _ in rows)
    input_tensor = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    label_tensor = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    mask_tensor = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)

    for row_idx, (input_ids, labels, choice_mask) in enumerate(rows):
        length = len(input_ids)
        input_tensor[row_idx, :length] = torch.tensor(input_ids, dtype=torch.long)
        label_tensor[row_idx, :length] = torch.tensor(labels, dtype=torch.long)
        mask_tensor[row_idx, :length] = torch.tensor(choice_mask, dtype=torch.bool)
        attention_mask[row_idx, :length] = 1

    return (
        input_tensor.to(device),
        attention_mask.to(device),
        label_tensor.to(device),
        mask_tensor.to(device),
    )


@torch.inference_mode()
def score_choices(
    model,
    tokenizer,
    example: MCExample,
    device: torch.device,
    max_length: int,
    reset_norm,
) -> tuple[list[float], list[float]]:
    input_ids, attention_mask, labels, choice_mask = prepare_choice_tensors(
        tokenizer,
        example.prompt,
        example.choices,
        max_length,
        device,
    )
    if reset_norm is not None:
        reset_norm()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    safe_labels = labels.masked_fill(labels < 0, 0)
    token_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~choice_mask, 0.0)
    sums = token_log_probs.sum(dim=-1)
    counts = choice_mask.sum(dim=-1).clamp_min(1)
    normed = sums / counts
    return sums.detach().cpu().tolist(), normed.detach().cpu().tolist()


def evaluate_task(
    model,
    tokenizer,
    task: str,
    examples: list[MCExample],
    device: torch.device,
    max_length: int,
    reset_norm,
    progress_every: int,
    writer: csv.DictWriter,
) -> dict[str, Any]:
    raw_correct = 0
    norm_correct = 0
    t0 = time.time()

    for i, example in enumerate(examples, start=1):
        raw_scores, norm_scores = score_choices(model, tokenizer, example, device, max_length, reset_norm)
        raw_pred = int(max(range(len(raw_scores)), key=raw_scores.__getitem__))
        norm_pred = int(max(range(len(norm_scores)), key=norm_scores.__getitem__))
        raw_is_correct = raw_pred == example.label
        norm_is_correct = norm_pred == example.label
        raw_correct += int(raw_is_correct)
        norm_correct += int(norm_is_correct)

        writer.writerow(
            {
                "task": task,
                "idx": example.idx,
                "label": example.label,
                "raw_pred": raw_pred,
                "norm_pred": norm_pred,
                "raw_correct": int(raw_is_correct),
                "norm_correct": int(norm_is_correct),
                "raw_scores": json.dumps(raw_scores),
                "norm_scores": json.dumps(norm_scores),
            }
        )

        if progress_every > 0 and i % progress_every == 0:
            print(
                f"{task}: {i}/{len(examples)} "
                f"acc={raw_correct / i:.4f} acc_norm={norm_correct / i:.4f}",
                flush=True,
            )

    total = len(examples)
    return {
        "task": task,
        "n": total,
        "acc": raw_correct / max(1, total),
        "acc_norm": norm_correct / max(1, total),
        "raw_correct": raw_correct,
        "norm_correct": norm_correct,
        "elapsed_sec": time.time() - t0,
    }


def write_summary(output_dir: Path, results_name: str, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    summary = {"metadata": metadata, "tasks": rows}
    json_path = output_dir / f"{results_name}.json"
    csv_path = output_dir / f"{results_name}.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    fieldnames = ["task", "n", "acc", "acc_norm", "raw_correct", "norm_correct", "elapsed_sec"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)


def default_results_name(metadata: dict[str, Any]) -> str:
    step = metadata.get("step", "unknown")
    tokens_m = metadata.get("tokens_seen_m")
    if isinstance(tokens_m, float) and math.isfinite(tokens_m):
        return f"step{step}_tokens{tokens_m:.0f}M"
    return f"step{step}"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"device={device} dtype={dtype}", flush=True)

    model, tokenizer, reset_norm, metadata = load_model(args, device, dtype)
    metadata.update(
        {
            "tasks": task_names(args.tasks),
            "max_samples_per_task": args.max_samples_per_task,
            "sample_offset": args.sample_offset,
            "max_length": args.max_length,
        }
    )
    results_name = args.results_name or default_results_name(metadata)

    example_path = output_dir / f"{results_name}_examples.csv"
    summary_rows: list[dict[str, Any]] = []
    with example_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "idx",
                "label",
                "raw_pred",
                "norm_pred",
                "raw_correct",
                "norm_correct",
                "raw_scores",
                "norm_scores",
            ],
        )
        writer.writeheader()
        for task in metadata["tasks"]:
            examples = TASK_LOADERS[task](args.max_samples_per_task, args.sample_offset)
            print(f"\n=== {task}: {len(examples)} examples ===", flush=True)
            summary_rows.append(
                evaluate_task(
                    model,
                    tokenizer,
                    task,
                    examples,
                    device,
                    args.max_length,
                    reset_norm,
                    args.progress_every,
                    writer,
                )
            )
            print(
                f"{task}: acc={summary_rows[-1]['acc']:.4f} "
                f"acc_norm={summary_rows[-1]['acc_norm']:.4f}",
                flush=True,
            )

    write_summary(output_dir, results_name, metadata, summary_rows)
    print("\nSummary:", flush=True)
    for row in summary_rows:
        print(
            f"{row['task']}: n={row['n']} acc={row['acc']:.4f} "
            f"acc_norm={row['acc_norm']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
