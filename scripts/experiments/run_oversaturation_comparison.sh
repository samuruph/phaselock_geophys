#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository root so this launcher works even when called from another cwd.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Compare oversaturation on the exact sample intersection shared by real Physics-IQ,
# baseline, PhaseLock, and the running-momentum Ours videos. The Python evaluator derives
# real-video paths from the Physics-IQ metadata and joins every generated directory by its
# leading four-digit sample ID.

REAL_ROOT="${REAL_ROOT:-/data/datasets/physics-IQ-benchmark-verified}"
BASELINE_ROOT="${BASELINE_ROOT:-/data/experiments/phaselock_running_momentum/cogvideox_5b_i2v/physics_iq/physics_iq_test_running_momentum/videos/baseline}"
PHASELOCK_ROOT="${PHASELOCK_ROOT:-/data/experiments/phaselock/physics_iq}"
OURS_ROOT="${OURS_ROOT:-/data/experiments/phaselock_running_momentum/running_momentum_source_latent/cogvideox_5b_i2v/physics_iq/physics_iq_test_running_momentum/videos/motion_on_latent}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/oversaturation/physics_iq_running_momentum}"

python scripts/measure_oversaturation.py \
  --real-root "$REAL_ROOT" \
  --baseline-root "$BASELINE_ROOT" \
  --phaselock-root "$PHASELOCK_ROOT" \
  --ours-root "$OURS_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
