#!/usr/bin/env bash
set -euo pipefail
# Standalone CogVideoX I2V DDIM commands, same seven clips, source=latent, seed=42.
# Copy any one line. K=1 adds one P&P prediction at each step in [0, 25).
# For a one-sample smoke test, replace the sample list with --sample-id 0001.
# For the full benchmark, remove --sample-id and choose fresh run IDs.

# Matched DDIM baseline and running-momentum controls: no P&P predictions.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- refinement__enabled=true refinement__steps_per_timestep=0 phaselock__guidance_strength=0 output__run_id=pnp_v1_ddim_baseline_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- refinement__enabled=true refinement__steps_per_timestep=0 output__run_id=pnp_v1_ddim_momentum_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49

# P&P only, then P&P plus momentum without/with the confidence gate.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- refinement__enabled=true refinement__steps_per_timestep=1 refinement__confidence_gate=false phaselock__guidance_strength=0 output__run_id=pnp_v1_pnp_only_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- refinement__enabled=true refinement__steps_per_timestep=1 refinement__confidence_gate=false output__run_id=pnp_v1_pnp_momentum_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- refinement__enabled=true refinement__steps_per_timestep=1 refinement__confidence_gate=true output__run_id=pnp_v1_pnp_confidence_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49

# 75-step DDIM baseline matches the default P&P run's 75 model predictions.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- refinement__enabled=true refinement__steps_per_timestep=0 phaselock__guidance_strength=0 generation__num_steps=75 output__run_id=pnp_v1_compute_control_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49

# K=2 tests whether another same-timestep refinement pass helps. In [0,25), this
# uses 100 predictions in total; the 100-step DDIM arm matches that compute budget.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- refinement__enabled=true refinement__steps_per_timestep=0 phaselock__guidance_strength=0 generation__num_steps=100 output__run_id=pnp_v1_compute_control_100_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- refinement__enabled=true refinement__steps_per_timestep=2 refinement__confidence_gate=false phaselock__guidance_strength=0 output__run_id=pnp_v1_pnp_only_k2_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- refinement__enabled=true refinement__steps_per_timestep=2 refinement__confidence_gate=false output__run_id=pnp_v1_pnp_momentum_k2_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- refinement__enabled=true refinement__steps_per_timestep=2 refinement__confidence_gate=true output__run_id=pnp_v1_pnp_confidence_k2_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49

# Repeat the confidence-gated arm with independent generation seeds.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- refinement__enabled=true refinement__steps_per_timestep=1 refinement__confidence_gate=true generation__seed=43 output__run_id=pnp_v1_pnp_confidence_g43 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- refinement__enabled=true refinement__steps_per_timestep=1 refinement__confidence_gate=true generation__seed=44 output__run_id=pnp_v1_pnp_confidence_g44 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49

# Hold generation seed 42 fixed and vary only P&P's auxiliary re-noising seed.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- refinement__enabled=true refinement__steps_per_timestep=1 refinement__confidence_gate=true refinement__seed=1 output__run_id=pnp_v1_pnp_confidence_g42_a1 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- refinement__enabled=true refinement__steps_per_timestep=1 refinement__confidence_gate=true refinement__seed=2 output__run_id=pnp_v1_pnp_confidence_g42_a2 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49

# Secondary source sensitivity: repeat K=1 confidence-gated P&P plus momentum.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source x0_hat --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- refinement__enabled=true refinement__steps_per_timestep=1 refinement__confidence_gate=true output__run_id=pnp_v1_pnp_confidence_x0hat_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source blend --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- refinement__enabled=true refinement__steps_per_timestep=1 refinement__confidence_gate=true output__run_id=pnp_v1_pnp_confidence_blend_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
