#!/usr/bin/env bash
# Reproduce Table 1 of the paper: main 2x2 ablation + final-only norm + norm penalty,
# 3 seeds (42, 101, 202) at 44M and 129M.
#
# Each run is ~4 epochs on WikiText-103, K=4 loops.
# Wall-clock: ~7h per 129M run on a single L40S; ~3h per 44M run.

set -euo pipefail

SEEDS=(42 101 202)
SCALES=(44m 129m)

# Map scale label -> config prefix
declare -A PREFIX=(
  [44m]="configs/44m/ar_looplm_full"
  [129m]="configs/129m/ar_looplm_150m"
)

# The seven conditions
CONDITIONS=(
  "terminal_norm"
  "terminal_raw"
  "perstep_norm"
  "perstep_raw"
  "perstep_final_norm"
  "terminal_norm_penalty"
)

for scale in "${SCALES[@]}"; do
  for cond in "${CONDITIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      cfg="${PREFIX[$scale]}_${cond}.yaml"
      out="outputs/${cond}_${scale}_s${seed}"
      echo "==== ${scale} ${cond} seed=${seed} ===="
      python src/train_ar_looplm.py \
        --config "${cfg}" \
        --set "train.seed=${seed}" \
        --set "train.output_dir=${out}"
    done
  done
  # Per-loop + norm penalty reuses the per-loop-norm config with the penalty flag set
  for seed in "${SEEDS[@]}"; do
    cfg="${PREFIX[$scale]}_perstep_norm.yaml"
    out="outputs/perstep_norm_penalty_${scale}_s${seed}"
    echo "==== ${scale} perstep_norm_penalty seed=${seed} ===="
    python src/train_ar_looplm.py \
      --config "${cfg}" \
      --set "supervision.norm_penalty_weight=0.01" \
      --set "train.seed=${seed}" \
      --set "train.output_dir=${out}"
  done
done

echo "All Table 1 runs complete."
