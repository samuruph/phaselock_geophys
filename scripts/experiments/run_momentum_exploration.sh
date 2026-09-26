#!/usr/bin/env bash
set -euo pipefail
# Standalone commands for the same seven Physics-IQ samples, source=latent, seed=42.
# Copy any one command into a terminal. Each has its own run ID.
# For a one-sample smoke test, replace the sample list with --sample-id 0001.
# For the full benchmark, remove --sample-id and choose fresh run IDs.

# Momentum only: exploration is enabled for matched capture, but noise_ratio=0.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true exploration__noise_ratio=0 output__run_id=explore_v1_momentum_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49

# Structured exploration only: the momentum correction has zero strength.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true phaselock__guidance_strength=0 output__run_id=explore_v1_noise_only_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49

# Momentum plus motion x disagreement structured exploration.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true output__run_id=explore_v1_combined_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49

# Same proposed noise RMS, with no spatial/temporal noise smoothing.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true exploration__structured=false output__run_id=explore_v1_unstructured_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49

# Mask ablations. The default combined mask is the arm above.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true exploration__mask_mode=uniform output__run_id=explore_v1_uniform_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true exploration__mask_mode=motion output__run_id=explore_v1_motion_mask_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true exploration__mask_mode=disagreement output__run_id=explore_v1_disagreement_mask_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49

# Isolate the mask effect without momentum. Together these four commands test all mask
# modes under noise only; compare each with the noise-only combined-mask arm above.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true exploration__mask_mode=uniform phaselock__guidance_strength=0 output__run_id=explore_v1_noise_only_uniform_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true exploration__mask_mode=motion phaselock__guidance_strength=0 output__run_id=explore_v1_noise_only_motion_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true exploration__mask_mode=disagreement phaselock__guidance_strength=0 output__run_id=explore_v1_noise_only_disagreement_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49

# Test whether the default noise strength or the small mask floor drives the result.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true exploration__noise_ratio=0.0033 output__run_id=explore_v1_low_strength_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true exploration__noise_ratio=0.03 output__run_id=explore_v1_high_strength_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true exploration__noise_ratio=0.1 output__run_id=explore_v1_cap_stress_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true exploration__floor=0 output__run_id=explore_v1_no_floor_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49

# Auxiliary-seed diversity: copy the combined command and change both seed and run ID.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true exploration__seed=1 output__run_id=explore_v1_combined_g42_a1 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true exploration__seed=2 output__run_id=explore_v1_combined_g42_a2 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49

# Repeat the main combined arm with independent generation seeds.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true generation__seed=43 output__run_id=explore_v1_combined_g43 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true generation__seed=44 output__run_id=explore_v1_combined_g44 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49

# Source sensitivity is a secondary check: keep all exploration settings fixed and
# change only the running-momentum measurement source.
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source x0_hat --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true output__run_id=explore_v1_combined_x0hat_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source blend --sample-id 0001 0002 0004 0005 0007 0008 0010 --diagnostics -- exploration__enabled=true output__run_id=explore_v1_combined_blend_g42 diagnostics__record_steps=0,1,2,3,4,5,10,20,24,30,49
