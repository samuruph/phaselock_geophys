#!/usr/bin/env bash
set -euo pipefail
# One-parameter-at-a-time running-momentum ablations. Every command is standalone and
# uses latent motion, the same seven Physics-IQ clips and seed 42. The rm_fixed_ run IDs
# are fresh: old rm_ videos were produced before the momentum correction was fixed, and
# run_physics_iq.py skips an existing video unless --overwrite is supplied.
# To run the full benchmark, remove the --sample-id list from each command and use new
# run IDs so the seven-clip pilot outputs are not mixed with full-benchmark outputs.

# Corrected default, including a paired unguided baseline. This is the common reference
# for every single-parameter command below: strength 0.05, floor 0.1, cap 0.1,
# velocity decay 0.01, beta1 0.9, beta2 0.999, and guidance window [0, 25).
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance baseline motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_default diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49

# Guidance strength: configured value is 0.05.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_strength_0025 phaselock__guidance_strength=0.025 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_strength_010 phaselock__guidance_strength=0.10 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_strength_020 phaselock__guidance_strength=0.20 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49

# Window start/end: configured window is [0, 25), with guide_end exclusive.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_guide_start_5 phaselock__guide_start=5 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_guide_end_15 phaselock__guide_end=15 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_guide_end_35 phaselock__guide_end=35 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49

# Adam-style moment coefficients: vary one beta while holding the other at its default.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_beta1_080 phaselock__beta1=0.80 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_beta1_095 phaselock__beta1=0.95 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_beta2_099 phaselock__beta2=0.99 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_beta2_09999 phaselock__beta2=0.9999 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49

# Extra first-moment decay: configured value is 0.01.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_velocity_decay_0 phaselock__velocity_decay=0.0 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_velocity_decay_005 phaselock__velocity_decay=0.05 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49

# Momentum formula: configured mode is residual.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_mode_snr phaselock__running_momentum_mode=snr diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49

# Adaptive denominator floor: 0.1 is the corrected default. The new floor/cap sweeps
# all use strength 0.10, with rm_fixed_strength_010 above as their reference. A stronger
# intervention makes the cap more likely to activate.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_floor_0 phaselock__guidance_strength=0.10 phaselock__variance_floor_fraction=0.0 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_floor_001 phaselock__guidance_strength=0.10 phaselock__variance_floor_fraction=0.01 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_floor_03 phaselock__guidance_strength=0.10 phaselock__variance_floor_fraction=0.3 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49

# Maximum applied update RMS relative to latent RMS: 0.1 is the corrected default.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_cap_0025 phaselock__guidance_strength=0.10 phaselock__max_update_ratio=0.025 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_cap_005 phaselock__guidance_strength=0.10 phaselock__max_update_ratio=0.05 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- output__run_id=rm_fixed_cap_02 phaselock__guidance_strength=0.10 phaselock__max_update_ratio=0.2 diagnostics__record_steps=0,1,2,3,4,5,10,20,30,40,49
