#!/usr/bin/env bash
set -euo pipefail
# One-parameter-at-a-time running-momentum ablations. All runs use latent, the same
# 8 Physics-IQ clips, and the configured defaults for every parameter not overridden.
# To run the full set, remove the --sample-id list from each command.

# Guidance strength: configured value is 0.05.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_strength_0025 phaselock__guidance_strength=0.025 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_strength_010 phaselock__guidance_strength=0.10 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_strength_020 phaselock__guidance_strength=0.20 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49

# Window start/end: configured window is [0, 25), with guide_end exclusive.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_guide_start_5 phaselock__guide_start=5 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_guide_end_15 phaselock__guide_end=15 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_guide_end_35 phaselock__guide_end=35 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49

# Adam-style moment coefficients: vary one beta while holding the other at its default.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_beta1_080 phaselock__beta1=0.80 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_beta1_095 phaselock__beta1=0.95 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_beta2_099 phaselock__beta2=0.99 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_beta2_09999 phaselock__beta2=0.9999 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49

# Extra first-moment decay: configured value is 0.01.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_velocity_decay_0 phaselock__velocity_decay=0.0 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_velocity_decay_005 phaselock__velocity_decay=0.05 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49

# Momentum formula: configured mode is residual.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_mode_snr phaselock__running_momentum_mode=snr diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
