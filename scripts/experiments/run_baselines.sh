#!/usr/bin/env bash
# Unguided Physics-IQ baselines: plain sampling, one model per invocation of the driver.
#
#   scripts/experiments/run_baselines.sh                    # all three, all 198 scenarios
#   MODELS=cogvideox scripts/experiments/run_baselines.sh   # just one
#   N=2 scripts/experiments/run_baselines.sh                # smoke test before committing
#   MODELS="wan21 wan22" N=2 scripts/experiments/run_baselines.sh
#   CATEGORIES="Solid Mechanics" scripts/experiments/run_baselines.sh
#
# `--guidance baseline` is the whole point: no few-step pass, no callback, no prior. See
# generate() in scripts/run_physics_iq.py -- the baseline arm skips the 2-step pass outright
# rather than running it at strength 0, so this is the stock pipeline and nothing else.
#
# Each model runs at ITS OWN operating point, from its own config, not at a shared one. A
# baseline is only a baseline at the settings the checkpoint was released with:
#
#   cogvideox  50 steps, cfg 6.0   49 frames @  8 fps = 6.125 s, truncated to the 5 s window
#   wan21      50 steps, cfg 5.0   81 frames @ 16 fps = 5.06 s, covers the window exactly
#   wan22      40 steps, cfg 3.5   81 frames @ 16 fps   (2.2's numbers are from the
#                                                        reference script, NOT diffusers'
#                                                        defaults of 50 / 5.0)
#
# ---- two things that will bite you ----------------------------------------------------
#
# 1. The `--` before the overrides is load-bearing. `--guidance` is nargs="+", so without
#    it argparse reads `output__root=...` as another guidance name and dies with
#    "invalid choice". Verified; do not remove it.
#
# 2. RUN_ID is stable per model, so re-running RESUMES: the driver skips clips whose video
#    is already on disk. But run_physics_iq.py writes its CSV with "w" and only holds the
#    clips generated THIS invocation, so a resumed run overwrites the CSV with just the new
#    rows. The videos survive; the scores do not. Back the CSV up before resuming:
#      cp .../physics_iq_baseline.csv{,.bak}

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$PWD"

MODELS="${MODELS:-cogvideox wan21 wan22}"
N="${N:-0}"                                   # 0 means all 198 take-1 scenarios
CATEGORIES="${CATEGORIES:-}"                  # blank = all five
ROOT="${ROOT:-/data/experiments/physics-iq}"

config_for () {
  case "$1" in
    cogvideox) echo "configs/experiments/physics_iq.yaml" ;;
    wan21)     echo "configs/experiments/physics_iq_wan21_i2v.yaml" ;;
    wan22)     echo "configs/experiments/physics_iq_wan22_i2v.yaml" ;;
    *) echo "" ;;
  esac
}

LIMIT=""; [ "$N" != 0 ] && LIMIT="--limit $N"
CATS=""; [ -n "$CATEGORIES" ] && CATS="--categories"
SUFFIX="n${N}"; [ "$N" = 0 ] && SUFFIX="full"

mkdir -p "$ROOT"
echo "models: $MODELS"
echo "clips : $([ "$N" = 0 ] && echo 'all 198' || echo "$N")   categories: ${CATEGORIES:-all five}"
echo "output: $ROOT/<model>_baseline_$SUFFIX"

for model in $MODELS; do
  CONFIG="$(config_for "$model")"
  if [ -z "$CONFIG" ]; then
    echo "## SKIP $model -- no config; expected one of: cogvideox wan21 wan22"
    continue
  fi
  RUN_ID="${model}_baseline_${SUFFIX}"
  mkdir -p "$ROOT/$RUN_ID"
  LOG="$ROOT/$RUN_ID/run.log"

  echo ""
  echo "############################################################"
  echo "## $model baseline   started $(date +%H:%M:%S)   -> $LOG"
  echo "############################################################"
  # Guarded, not `set -e`: a 15-hour unattended run must not lose the models that work
  # because an earlier one ran out of disk or weights.
  if python scripts/run_physics_iq.py \
       --config "$CONFIG" \
       --guidance baseline \
       $CATS ${CATEGORIES:+"$CATEGORIES"} $LIMIT \
       -- \
       output__root="$ROOT" output__run_id="$RUN_ID" 2>&1 | tee -a "$LOG"
  then echo "## OK $model $(date +%H:%M:%S)"
  else echo "## FAILED $model $(date +%H:%M:%S) -- continuing"
  fi
done

echo ""
echo "############################################################"
echo "## ALL BASELINES ATTEMPTED -- finished $(date +%H:%M:%S)"
echo "############################################################"
cat <<'NOTES'
Videos:   <root>/<model>_baseline_<suffix>/<backend>/physics_iq/physics_iq/videos/baseline/
          Named as the official evaluator expects, so the directory can be handed to the
          Physics-IQ repo unchanged.

Compare:  python scripts/summarise_guidance.py /data/experiments/physics-iq
          Each model is a separate run, so they appear as separate tables -- these are
          different models, not settings of one, and there is no paired delta between them.

Before a long Wan run: wan21 and wan22 are registered but validated=False -- wired, config
loads, nothing has generated a frame yet. Smoke-test first:
  MODELS=wan21 N=2 scripts/experiments/run_baselines.sh
Weights are ~70 GB (wan21) and ~118 GB (wan22) and are NOT downloaded yet.
NOTES
