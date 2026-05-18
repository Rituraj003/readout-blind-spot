# Dense Supervision Is Not Enough: The Readout Blind Spot in Looped Language Models

Code release for the paper
**"Dense Supervision Is Not Enough: The Readout Blind Spot in Looped Language Models"**
by Rituraj Sharma and Tu Vu (Virginia Tech).

> Looped language models are commonly trained by applying cross-entropy at every loop, on the assumption that direct supervision of each intermediate state stabilizes the recurrent computation. We show that this assumption does not hold when all supervised readouts are scale-invariant. In 44M and 129M looped transformers without inter-loop normalization, per-loop cross-entropy through RMSNorm readouts still drives final hidden-state norms into the thousands or tens of thousands. The reason is a visibility–activity mismatch: scale-invariant readouts hide hidden-state scale from the immediate cross-entropy loss, while pre-norm residual recurrence continues to carry and update that same scale.

## What's here

```
code/
├── configs/                       # YAML configs for the experiments in the paper
│   ├── 44M/                       # 44M-parameter looped LM (d=512, 8 layers)
│   ├── 129m/                      # 129M-parameter looped LM (d=768, 12 layers)
│   └── 1.4b/                      # 1.4B Ouro-scale sanity-check experiment
├── src/
│   ├── models/ar_looplm.py        # Looped LM model definition (the AR-LoopLM family)
│   ├── data/{dataset,collator}.py # WikiText-103 loader, regex tokenizer, collator
│   ├── losses/objectives.py       # Per-loop CE, terminal CE, norm penalty
│   ├── common.py                  # Shared training utilities
│   ├── train_ar_looplm.py         # Training entry point for the 44M/129M ablations
│   └── eval/                      # Diagnostic and evaluation scripts (see below)
└── scripts/                       # Convenience reproduction scripts
```

## Quickstart

```bash
# 1) Environment
python -m venv .venv && source .venv/bin/activate
pip install -e .                  # uses pyproject.toml

# 2) Train one condition (e.g., 129M per-loop + RMSNorm, seed 42)
python src/train_ar_looplm.py \
  --config configs/129m/ar_looplm_150m_perstep_norm.yaml \
  --set train.seed=42 \
  --set train.output_dir=outputs/perstep_norm_129m_s42

# 3) Verify the Lemma 2 expansion on the trained checkpoint
python src/eval/lemma2_verification.py \
  --checkpoint perstep_norm_129m=outputs/perstep_norm_129m_s42/last.pt \
  --output-dir outputs/lemma2_verification
```

## Reproducing the paper's main tables

### Table 1 (main 2×2 + extensions): final-loop PPL and ‖H_K‖

Train each of the seven conditions × 3 seeds × 2 model scales (44M, 129M). The seven conditions per scale are:

| Condition | Config (44M) | Config (129M) | Notes |
|---|---|---|---|
| Terminal + RMSNorm | `configs/44m/ar_looplm_full_terminal_norm.yaml` | `configs/129m/ar_looplm_150m_terminal_norm.yaml` | — |
| Terminal + raw | `configs/44m/ar_looplm_full_terminal_raw.yaml` | `configs/129m/ar_looplm_150m_terminal_raw.yaml` | — |
| Per-loop + RMSNorm | `configs/44m/ar_looplm_full_perstep_norm.yaml` | `configs/129m/ar_looplm_150m_perstep_norm.yaml` | — |
| Per-loop + raw | `configs/44m/ar_looplm_full_perstep_raw.yaml` | `configs/129m/ar_looplm_150m_perstep_raw.yaml` | — |
| Per-loop + final-only norm | `configs/44m/ar_looplm_full_perstep_final_norm.yaml` | `configs/129m/ar_looplm_150m_perstep_final_norm.yaml` | — |
| Terminal + norm penalty | `configs/44m/ar_looplm_full_terminal_norm_penalty.yaml` | `configs/129m/ar_looplm_150m_terminal_norm_penalty.yaml` | λ=0.01 |
| Per-loop + norm penalty | (per-loop RMSNorm config + `--set supervision.norm_penalty_weight=0.01`) | (same) | λ=0.01 |

See `scripts/reproduce_table1.sh` for the seed loop.

### Table 3 (radial gradient diagnostic) and Table 12 (residual-update decomposition)

```bash
python src/eval/lemma2_verification.py \
  --checkpoint norm_44m_s42=outputs/perstep_norm_44m_s42/last.pt \
  --checkpoint raw_44m_s42=outputs/perstep_raw_44m_s42/last.pt \
  --checkpoint ... \
  --output-dir outputs/lemma2_verification
```

### Table 4 (radial clamp): scale growth is cheap to remove

```bash
python src/eval/radial_clamp_intervention.py \
  --checkpoint outputs/perstep_norm_129m_s42/last.pt \
  --output-dir outputs/radial_clamp
```

### Table 5 (variable-depth K=1..4) and Figure 4 (Pareto frontier)

```bash
python src/eval/early_exit_curve.py \
  --checkpoint raw_s42=outputs/perstep_raw_129m_s42/last.pt \
  --checkpoint norm_s42=outputs/perstep_norm_129m_s42/last.pt \
  ... \
  --output-dir outputs/pareto
python src/eval/plot_pareto_compute_quality.py --input outputs/pareto
```

### Table 8 (calibrated halting)

```bash
python src/eval/adaptive_halt_benchmark.py \
  --checkpoint ... \
  --target-rel-ppl 1.01 \
  --policy logit_margin \
  --output-dir outputs/halting
```

### Table 10 (per-token norm distribution)

```bash
python src/eval/token_norm_stats.py \
  --checkpoint outputs/perstep_norm_44m_s42/last.pt \
  --output-dir outputs/token_norms
```

### Appendix B: 1.4B Ouro sanity check

```bash
# Train (long; use a multi-GPU setup; see configs/1.4b/)
python src/train_ar_looplm.py --config configs/1.4b/ar_looplm_1.4b_terminal_norm.yaml ...
python src/train_ar_looplm.py --config configs/1.4b/ar_looplm_1.4b_terminal_raw.yaml ...

# Mechanism diagnostic at 1.4B
python src/eval/ouro_scale_sensitivity.py --checkpoint .../last.pt
python src/eval/ouro_variable_depth.py    --checkpoint .../last.pt
python src/eval/ouro_downstream_eval.py   --checkpoint .../last.pt
```

## Data

- **44M / 129M ablations**: WikiText-103 (Merity et al., 2017), accessed via `datasets`. A custom regex tokenizer with vocab 20k is fit from the training split; details in `src/data/dataset.py`.
- **1.4B sanity check**: FineWeb sample-10BT (Penedo et al., 2024). The 1.4B run uses the Ouro tokenizer from `ByteDance/Ouro-1.4B`.

## Citation

```bibtex
@misc{sharma2026readout,
  title  = {Dense Supervision Is Not Enough: The Readout Blind Spot in Looped Language Models},
  author = {Sharma, Rituraj and Vu, Tu},
  year   = {2026},
  eprint = {TBD},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## License

MIT. See `LICENSE`.

## Acknowledgments

We thank Chris Thomas for insightful conversations and feedback on the manuscript. The authors acknowledge Advanced Research Computing at Virginia Tech (<https://arc.vt.edu/>) for providing computational resources and technical support that have contributed to the results reported within this paper.
