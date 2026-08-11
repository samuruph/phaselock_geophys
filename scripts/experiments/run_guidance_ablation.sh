#!/usr/bin/env bash
# Which GeoPhys signal, held fixed during sampling, helps Physics-IQ most?
#
#   scripts/experiments/run_guidance_ablation.sh                  # the staged default
#   N=4  GUIDANCE="baseline motion" scripts/experiments/run_guidance_ablation.sh   # smoke
#   N=24 scripts/experiments/run_guidance_ablation.sh             # prune the grid
#   SOURCES="latent x0_hat velocity" scripts/experiments/run_guidance_ablation.sh
#   N=0  scripts/experiments/run_guidance_ablation.sh             # all 198 take-1
#
# Every setting runs PhaseLock's equation (2) unchanged. Only the few-step prior differs:
#
#   motion   z[t+1] - z[t]                      PhaseLock as published
#   accel    z[t+2] - 2z[t+1] + z[t]
#   jerk     the third difference
#   perr     the component orthogonal to the affine span of the previous frames
#
# The prediction being tested: on the VAE latent -- PhaseLock's own space -- the detection
# study measured phi_speed, the statistic of `motion`, at 46.6%, BELOW chance, while perr,
# accel and jerk read 72.6/67.3/67.2. So `motion` is the weakest of the four, and a run
# guided on `perr` should beat it. See docs/RESULTS.md.
#
# One driver invocation per setting rather than one for all of them: they are hours apart
# in wall clock, and a failure in the fourth must not throw away the first three. Videos
# and rows are keyed by setting, and the driver skips clips already on disk, so re-running
# resumes rather than repeats.
#
# ---- on STRENGTH, before comparing settings -------------------------------------------
# lambda = 0.05 is the paper's, tuned for first differences. Higher-order priors are NOT
# reliably the same size, and the direction depends on how grainy the 2-step pass is.
# Measured on a synthetic latent, prior RMS relative to `motion`:
#
#            smooth    slightly grainy    grainy
#   accel     0.27x         0.71x          1.49x
#   jerk      0.06x         1.21x          2.70x
#   perr      0.00x         0.46x          0.85x
#
# A smooth trajectory has near-zero higher derivatives and lies in its own affine span; a
# grainy one has differencing amplify the grain. So the same lambda is a *weaker*
# intervention for `jerk` on a smooth prior and a *stronger* one on a grainy prior, and it
# cannot be corrected analytically -- it depends on the 2-step output.
#
# Every guided run therefore prints its measured prior RMS. Read it on the first small
# run. If the settings are within a small factor, one lambda is fair; if they are orders
# apart, sweep STRENGTH per setting before believing any ranking.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$PWD"

N="${N:-24}"                                  # 0 means all 198 take-1 scenarios
GUIDANCE="${GUIDANCE:-baseline motion accel jerk perr}"
SOURCES="${SOURCES:-latent}"           # latent | x0_hat | velocity
STRENGTH="${STRENGTH:-}"                      # blank keeps the paper's 0.05
ROOT="${ROOT:-/data/experiments/phaselock_prior_ablation}"
CONFIG="${CONFIG:-configs/experiments/physics_iq.yaml}"

RUN_ID="${RUN_ID:-$(python -c "from phaselock.config import default_run_id
print(default_run_id($N or None, '', 'guidance'))")}"

mkdir -p "$ROOT/$RUN_ID"
LOG="$ROOT/$RUN_ID/run.log"
exec > >(tee -a "$LOG") 2>&1
echo "logging to $LOG"

LIMIT=""; [ "$N" != 0 ] && LIMIT="--limit $N"
# output__root as well as the run id: without it the driver writes under
# OutputConfig's default and ROOT only ever names the log directory, which is
# how an empty folder gets created next to the real results.
OVERRIDE="output__root=$ROOT output__run_id=$RUN_ID"
[ -n "$STRENGTH" ] && OVERRIDE="$OVERRIDE phaselock__guidance_strength=$STRENGTH"

echo "few-step prior types: $GUIDANCE"
echo "sources: $SOURCES"
echo "clips: ${N:-all}   strength: ${STRENGTH:-0.05 (paper)}"
echo "output: $ROOT/$RUN_ID"

step () {
  echo ""
  echo "############################################################"
  echo "## $1   started $(date +%H:%M:%S)"
  echo "############################################################"
  shift
  if "$@"; then echo "## OK $(date +%H:%M:%S)"
  else echo "## FAILED $(date +%H:%M:%S) -- continuing"; fi
}

# The baseline first: every other number is a paired delta against it, so
# nothing downstream is interpretable until it exists.
# baseline is unguided, so it has no source and must not be run once per source.
for setting in $GUIDANCE; do
  if [ "$setting" = baseline ]; then
    step "baseline (unguided)" \
      python scripts/run_physics_iq.py --config "$CONFIG" --guidance baseline $LIMIT $OVERRIDE
    continue
  fi
  for source in $SOURCES; do
    step "$setting on $source" \
      python scripts/run_physics_iq.py --config "$CONFIG" --guidance "$setting" \
        --source "$source" $LIMIT $OVERRIDE
  done
done

echo ""
echo "############################################################"
echo "## ALL SETTINGS ATTEMPTED -- finished $(date +%H:%M:%S)"
echo "############################################################"
echo "Videos: $ROOT/$RUN_ID/**/videos/<setting>/, named as the official Physics-IQ evaluator"
echo "expects. Hand a directory to that repo and compare its score with the printed one."
