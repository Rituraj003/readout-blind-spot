#!/usr/bin/env bash
# Reproduce the four mechanism diagnostics (Tables 3, 4, 12 + Figure 3) from trained checkpoints.
# Assumes scripts/reproduce_table1.sh has produced checkpoints under outputs/.

set -euo pipefail

# 1. Radial-gradient diagnostic + Lemma 2 verification figure (Table 3, Figure 3, Table 12)
python src/eval/lemma2_verification.py \
  --checkpoint norm_44m_s42=outputs/perstep_norm_44m_s42/last.pt \
  --checkpoint raw_44m_s42=outputs/perstep_raw_44m_s42/last.pt \
  --checkpoint final_only_norm_44m_s42=outputs/perstep_final_norm_44m_s42/last.pt \
  --checkpoint norm_129m_s42=outputs/perstep_norm_129m_s42/last.pt \
  --checkpoint raw_129m_s42=outputs/perstep_raw_129m_s42/last.pt \
  --checkpoint final_only_norm_129m_s42=outputs/perstep_final_norm_129m_s42/last.pt \
  --output-dir outputs/lemma2_verification

# 2. Per-token norm distribution (Table 10)
python src/eval/token_norm_stats.py \
  --checkpoint outputs/perstep_norm_44m_s42/last.pt \
  --checkpoint outputs/perstep_norm_129m_s42/last.pt \
  --output-dir outputs/token_norms

# 3. Radial clamp intervention (Table 4)
for cond in perstep_norm perstep_raw perstep_final_norm; do
  for scale in 44m 129m; do
    for seed in 42 101 202; do
      python src/eval/radial_clamp_intervention.py \
        --checkpoint outputs/${cond}_${scale}_s${seed}/last.pt \
        --output-dir outputs/radial_clamp
    done
  done
done

# 4. Variable-depth (Table 5) and Pareto figure (Figure 4)
python src/eval/early_exit_curve.py \
  --checkpoint norm_s42=outputs/perstep_norm_129m_s42/last.pt \
  --checkpoint raw_s42=outputs/perstep_raw_129m_s42/last.pt \
  --checkpoint final_only_norm_s42=outputs/perstep_final_norm_129m_s42/last.pt \
  --checkpoint norm_penalty_s42=outputs/perstep_norm_penalty_129m_s42/last.pt \
  --output-dir outputs/pareto_129m

python src/eval/plot_pareto_compute_quality.py \
  --input outputs/pareto_129m \
  --output outputs/figures/pareto_compute_quality.pdf

# 5. Calibrated dynamic halting (Table 8)
python src/eval/adaptive_halt_benchmark.py \
  --checkpoint norm_s42=outputs/perstep_norm_129m_s42/last.pt \
  --checkpoint raw_s42=outputs/perstep_raw_129m_s42/last.pt \
  --checkpoint final_only_norm_s42=outputs/perstep_final_norm_129m_s42/last.pt \
  --checkpoint norm_penalty_s42=outputs/perstep_norm_penalty_129m_s42/last.pt \
  --target-rel-ppl 1.01 \
  --policy logit_margin \
  --output-dir outputs/halting_129m

echo "All diagnostics complete."
