#!/usr/bin/env bash
set -euo pipefail

# One-factor ablation of the corrected running-momentum controller on Physics-IQ.
# Each arm runs its own paired baseline and motion setting with the same seed/clips.
# The default is the seven-clip pilot; FULL=1 selects the whole benchmark.
# Example: FULL=1 DIAGNOSTICS=1 bash scripts/experiments/run_running_momentum_safety_ablation.sh

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
sweep_id="${SWEEP_ID:-$(date -u +%Y%m%d_%H%M%S)}"
output_root="${OUTPUT_ROOT:-/data/experiments/phaselock_running_momentum_safety_ablation/$sweep_id}"
guidance_strength="${GUIDANCE_STRENGTH:-0.10}"

sample_args=()
if [[ "${FULL:-0}" != 1 ]]; then
  read -r -a sample_ids <<< "${SAMPLE_IDS:-0001 0002 0004 0005 0007 0008 0010}"
  sample_args=(--sample-id "${sample_ids[@]}")
fi

diagnostic_args=(--no-diagnostics)
if [[ "${DIAGNOSTICS:-0}" == 1 ]]; then
  diagnostic_args=(--diagnostics)
fi

run_arm() {
  local run_id="$1"
  shift
  printf '\nRunning %s\n' "$run_id"
  "$python_bin" scripts/run_physics_iq.py \
    --config configs/experiments/physics_iq_running_momentum.yaml \
    --guidance baseline motion --source latent \
    "${sample_args[@]}" "${diagnostic_args[@]}" --no-save-prior -- \
    "output__root=$output_root" "output__run_id=$run_id" \
    "phaselock__guidance_strength=$guidance_strength" \
    phaselock__prior_mode=running_momentum \
    phaselock__running_momentum_mode=residual \
    phaselock__guide_start=0 phaselock__guide_end=25 \
    phaselock__beta1=0.9 phaselock__beta2=0.999 \
    phaselock__velocity_decay=0.01 \
    phaselock__variance_floor_fraction=0.1 \
    phaselock__max_update_ratio=0.1 \
    "$@"
}

# Reference: variance floor 0.1, update cap 0.1, velocity decay 0.01.
run_arm reference

# Variance floor, holding the update cap and all other settings at reference.
for value in 0.0 0.01 0.3; do
  run_arm "floor_${value//./p}" "phaselock__variance_floor_fraction=$value"
done

# Applied update cap, holding the variance floor at reference.
for value in 0.025 0.05 0.2; do
  run_arm "cap_${value//./p}" "phaselock__max_update_ratio=$value"
done

# Recheck first-moment decay because its bias correction changed.
for value in 0.0 0.05; do
  run_arm "decay_${value//./p}" "phaselock__velocity_decay=$value"
done

"$python_bin" scripts/summarise_guidance.py "$output_root" \
  --no-backfill --out summary.csv
printf '\nResults: %s/summary.csv\n' "$output_root"
