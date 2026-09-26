#!/usr/bin/env bash
set -euo pipefail
# Compare latent, x0_hat, and blended running-momentum sources on the same 8 clips.
# To run the full set, remove the --sample-id list from each command.

python scripts/run_physics_iq.py \
  --config configs/experiments/physics_iq_running_momentum.yaml \
  --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- \
  output__root=/data/experiments/phaselock_running_momentum_ablations/momentum_sources output__run_id=running_momentum_source_latent diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49

python scripts/run_physics_iq.py \
  --config configs/experiments/physics_iq_running_momentum.yaml \
  --guidance motion --source x0_hat --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- \
  output__root=/data/experiments/phaselock_running_momentum_ablations/momentum_sources output__run_id=running_momentum_source_x0_hat diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49

python scripts/run_physics_iq.py \
  --config configs/experiments/physics_iq_running_momentum.yaml \
  --guidance motion --source blend --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- \
  output__root=/data/experiments/phaselock_running_momentum_ablations/momentum_sources output__run_id=running_momentum_source_blend diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
