#!/usr/bin/env bash
# Every experiment, end to end, in dependency order.
#
#   scripts/experiments/run_all.sh              # the full set, n=100
#   N=5 scripts/experiments/run_all.sh          # a fast check
#   RESOLUTION=letterbox scripts/experiments/run_all.sh
#   STAGES="A C" scripts/experiments/run_all.sh # only some tracks
#
# See docs/RUNNING.md for what each stage is and how long it takes.
#
# Every stage is guarded rather than `set -e`: one broken stage must not cancel the others,
# because the point of a long unattended run is to have *something* for every track by the
# time you look.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$PWD"

N="${N:-100}"
SWEEP_N="${SWEEP_N:-30}"           # a (K, sigma) grid averages over clips; 100 buys nothing
VISUALS="${VISUALS:-4}"            # each visual pair costs an extra full inversion
RESOLUTION="${RESOLUTION:-native}" # native | letterbox
STAGES="${STAGES:-A B C D E}"
ROOT="${ROOT:-/data/experiments/phaselock_geophys}"

# One label for the whole invocation, computed ONCE here and passed to every stage. If a
# stage computed its own, the gate and the inversion would land in different folders and
# the report would not find its own baseline. Override to re-enter an existing run:
#   RUN_ID=20260810_1030_n100_native scripts/experiments/run_all.sh
#
# The format lives in phaselock.config.default_run_id, not here: one definition, and a
# test that pins it.
RUN_ID="${RUN_ID:-$(python -c "from phaselock.config import default_run_id
print(default_run_id($N, '$RESOLUTION'))")}"
# root as well as run id: ROOT otherwise names only the log directory, and a
# non-default ROOT would silently write results somewhere else.
ID="output__root=$ROOT output__run_id=$RUN_ID"

# Log to the run's own folder. Piping through `tee` by hand is easy to forget, and the
# one run you forget it on is the one you need the log for.
mkdir -p "$ROOT/$RUN_ID"
LOG="$ROOT/$RUN_ID/run.log"
exec > >(tee -a "$LOG") 2>&1
echo "logging to $LOG"

# Native 512x512 is the default because it measured *better*, not just faster: on a
# matched 12-pair comparison it beat the letterboxed geometry on four of five statistics,
# and lifted speed and angle consistency from below chance to above it. Wan's 480x832
# letterboxes a 512x512 source to 42% black, which dilutes the pooled feature that every
# statistic is computed from. Note reconstruction PSNR points the other way (41.3 dB vs
# 42.6) -- detection accuracy is the metric that matters here, and it disagrees.
if [ "$RESOLUTION" = native ]; then GEO="data__height=512 data__width=512"; else GEO=""; fi

WAN=configs/experiments/inversion_likephys_wan.yaml
COG=configs/experiments/inversion_likephys_cog_i2v.yaml
GEN=configs/experiments/generation_likephys.yaml
SWEEP=configs/experiments/step_sweep_likephys.yaml
IP2=configs/experiments/inversion_intphys2.yaml

step () {
  echo ""
  echo "############################################################"
  echo "## $1   started $(date +%H:%M:%S)"
  echo "############################################################"
  shift
  if "$@"; then echo "## OK $(date +%H:%M:%S)"
  else echo "## FAILED $(date +%H:%M:%S) -- continuing"; fi
}

has () { [[ " $STAGES " == *" $1 "* ]]; }

echo "n=$N  sweep=$SWEEP_N  visuals=$VISUALS  resolution=$RESOLUTION  stages=$STAGES"
echo "output: $ROOT/$RUN_ID"

# ---- A. Wan2.1-1.3B on LikePhys. The primary track: much the best inverter, and the
#         cheapest, so it runs first and is safely on disk early.
if has A; then
  step "A1 Wan gate" \
    python scripts/run_external.py --config $WAN --temporal-pool latent data__limit=$N $GEO $ID
  step "A2 Wan inversion" \
    python scripts/run_inversion.py --config $WAN data__limit=$N $GEO $ID \
      probe__save_trajectories=true --save-visuals $VISUALS
  step "A3 Wan report" \
    python scripts/report.py $ROOT/$RUN_ID/wan21_t2v_1_3b/likephys/inversion --figures
  step "A4 Wan overlays" \
    python scripts/annotate_videos.py $ROOT/$RUN_ID/wan21_t2v_1_3b/likephys/inversion
fi

# ---- B. CogVideoX-5B-I2V on LikePhys. Secondary: ~1.8x slower per clip, and it inverts
#         to a 17 dB gap against Wan's 7 dB. block_stride 6 keeps the trajectories small.
if has B; then
  step "B1 CogVideoX gate" \
    python scripts/run_external.py --config $COG --temporal-pool latent data__limit=$N $GEO $ID
  step "B2 CogVideoX inversion" \
    python scripts/run_inversion.py --config $COG data__limit=$N $GEO $ID \
      probe__save_trajectories=true probe__block_stride=6 --save-visuals $VISUALS
  step "B3 CogVideoX report" \
    python scripts/report.py $ROOT/$RUN_ID/cogvideox_5b_i2v/likephys/inversion --figures
  step "B4 CogVideoX overlays" \
    python scripts/annotate_videos.py $ROOT/$RUN_ID/cogvideox_5b_i2v/likephys/inversion
fi

# ---- C. Generation. Needs image-to-video, and CogVideoX-5B-I2V is the only one that
#         fits a 46 GB card -- Wan's I2V is 14B.
if has C; then
  # Every `key=value` override must sit in ONE run at the end. argparse fills the
  # positional `overrides` list from the first group it meets and then rejects the
  # second, so `--save-videos $GEO --verifier-source ... data__limit=...` dies with
  # "unrecognized arguments" -- which is how this stage produced nothing at n=100.
  step "C1 generation" \
    python scripts/run_generation.py --config $GEN --save-videos \
      --verifier-source hidden_states --verifier-statistic perr \
      data__limit=$N generation__num_candidates=1 probe__save_trajectories=true $GEO $ID
  step "C2 generation report" \
    python scripts/report.py $ROOT/$RUN_ID/cogvideox_5b_i2v/likephys/generation --figures
  step "C3 generation overlays" \
    python scripts/annotate_videos.py $ROOT/$RUN_ID/cogvideox_5b_i2v/likephys/generation
fi

# ---- D. Step sweep with PhaseLock's blur control.
if has D; then
  step "D1 step sweep" \
    python scripts/run_step_sweep.py --config $SWEEP data__limit=$SWEEP_N $GEO $ID
  step "D2 step sweep report" \
    python scripts/report.py $ROOT/$RUN_ID/cogvideox_5b_i2v/likephys/step_sweep --figures
fi

# ---- E. IntPhys2 on Wan rather than the config's CogVideoX default: this is an inversion
#         run and Wan is much the better inverter. window=0.5 keeps the centre 5.3 s, since
#         636 -> 81 frames over the whole clip can step straight over a brief violation.
if has E; then
  step "E1 IntPhys2 gate" \
    python scripts/run_external.py --config $IP2 --temporal-pool latent \
      backend__name=wan21_t2v_1_3b backend__offload=false data__limit=$N $GEO $ID
  step "E2 IntPhys2 inversion" \
    python scripts/run_inversion.py --config $IP2 \
      backend__name=wan21_t2v_1_3b backend__offload=false data__limit=$N $GEO $ID \
      probe__save_trajectories=true --save-visuals $VISUALS
  step "E3 IntPhys2 report" \
    python scripts/report.py $ROOT/$RUN_ID/wan21_t2v_1_3b/intphys2/inversion --figures
  step "E4 IntPhys2 overlays" \
    python scripts/annotate_videos.py $ROOT/$RUN_ID/wan21_t2v_1_3b/intphys2/inversion
fi

echo ""
echo "############################################################"
echo "## ALL STAGES ATTEMPTED -- finished $(date +%H:%M:%S)"
echo "############################################################"
find "$ROOT/$RUN_ID" \( -name "*.png" -o -name "*.csv" \) \
  | sed "s|$ROOT/$RUN_ID/||" | sort
