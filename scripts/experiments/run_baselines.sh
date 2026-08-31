#!/usr/bin/env bash
# Unguided Physics-IQ baselines. Run one of these; they are independent.
#
# The `--` before the override is required: --guidance takes multiple values, so without
# it argparse reads output__root=... as another guidance name and exits.
#
# Add --limit 2 to any of them for a smoke test first.

# CogVideoX-5B-I2V          50 steps, cfg 6.0, 49 frames @ 8 fps
python scripts/run_physics_iq.py \
  --config configs/experiments/physics_iq.yaml \
  --guidance baseline -- \
  output__root=/data/experiments/physics-iq/cogvideox-i2v

# Wan2.1-I2V-14B-480P       50 steps, cfg 5.0, 81 frames @ 16 fps
python scripts/run_physics_iq.py \
  --config configs/experiments/physics_iq_wan21_i2v.yaml \
  --guidance baseline -- \
  output__root=/data/experiments/physics-iq/wan21-i2v

# Wan2.2-I2V-A14B           40 steps, cfg 3.5, 81 frames @ 16 fps
python scripts/run_physics_iq.py \
  --config configs/experiments/physics_iq_wan22_i2v.yaml \
  --guidance baseline -- \
  output__root=/data/experiments/physics-iq/wan22-i2v

# Wan2.2-TI2V-5B            50 steps, cfg 5.0, 121 frames @ 24 fps, 704x1280
# The lightest of the Wan options: one 5B transformer, not two 14B experts.
python scripts/run_physics_iq.py \
  --config configs/experiments/physics_iq_wan22_ti2v_5b.yaml \
  --guidance baseline -- \
  output__root=/data/experiments/physics-iq/wan22-ti2v-5b
