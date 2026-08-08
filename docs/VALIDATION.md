# GPU bring-up and validation plan

> **Outcome (executed).** The ladder was run. Every rung passed, after four real bugs that
> only a GPU run could surface. Recorded here rather than rewritten away, because the
> failures are the useful part.
>
> | # | bug | how it presented | why it was hard |
> |---|---|---|---|
> | 1 | `Tensor.to(dtype, device=…)` — not a valid overload | raised in the VAE encode | looked like a model bug |
> | 2 | `vae.device` reads `cpu` under CPU offload while accelerate runs it on GPU | device mismatch | the obvious fix (reorder the args) is still wrong |
> | 3 | system CUDA shadowing torch's `libcublasLt.so.13` | **process abort** inside the VAE encode | a plain `torch.matmul` succeeds, so the obvious smoke test misses it; and the abort spent minutes writing a core dump, so a hard crash looked like a hang |
> | 4 | **inversion never advanced** — renoising to the level just evaluated at is the identity | reconstruction PSNR *identical* at k=20/50/100 | the oracle test passed throughout, for the wrong reason |
>
> Bug 4 is the one that mattered. `(x0, eps)` are derived from `z` at level `s`, so
> recombining them at `s` reconstructs `z` exactly — the loop returned the clean latent
> untouched and every recorded trajectory was one point repeated. It was caught not by a
> test but by the *shape of the numbers*: a reconstruction that does not change with step
> count is not being integrated.
>
> The gate outcomes are in [RESULTS.md](RESULTS.md).

---

This is the plan as written before execution: what to fix first, the order to bring things
up, what each step must produce to count as passing, and what to do when it does not.

---

## Phase 0 — before the GPU is free (CPU work)

### 0.1 Wire up the ensembles *(done)*

`ensemble_over_statistics`, `majority_ensemble` and `majority_vote_accuracy` were
implemented and tested, but **no script reported them** — a real gap, because OR
(`argmax_b |z_b|`) is what carries GeoPhys's headline numbers: 98.3% on LikePhys against
77.6–80.8% for the best single signal. Reporting only single signals would have understated
the method by ~18 points and made the comparison against the paper meaningless.

`run_detection.py` now reports OR and Majority per `(source, block, step)` alongside the
single-signal table, and both land in `signals.csv` with `statistic ∈ {or, majority}`.

Writing the test for it exposed a second bug: `scale_normalize` divided by the standard
deviation, so a signal that ordered *every* pair by an identical margin -- the best
possible detector -- had zero spread and was normalised to zeros, discarding the sign the
ensembles vote on. It now falls back to the mean magnitude.

### 0.2 Re-read the two riskiest files end to end *(open)*

`pipelines/inversion.py` and `pipelines/generation.py` contain the only logic that has
never executed against a real model *and* cannot be fully checked on CPU. The oracle
backend validates the loop structure; it cannot validate the diffusers contract.

### 0.3 Confirm the GPU is actually free

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

Any long-running PhaseLock job must be gone (the legacy `test_physics_iq` script was
replaced by `datasets/physics_iq.py`, but an older copy may still be running). Do not start alongside it — the
box has 61 GB RAM with ~26 GB free, and CogVideoX-5B with offload will contend for both
GPU and host memory.

---

## Phase 1 — bring-up ladder

Each rung is cheap, gates the next, and isolates one class of failure. Do not skip ahead;
a failure three rungs up is far harder to localise.

### Rung 1 — Wan backend smoke (~10 min)

```bash
python scripts/inference.py --backend wan21_t2v_1_3b \
    --prompt "a red ball bouncing on a table" --output /tmp/wan.mp4 \
    --full-steps 4 --no-phaselock
```

**Passes if:** a 4-step mp4 is written with 81 frames.

**Exercises:** Wan weight loading, VAE tiling, `encode_prompt`, the pipeline call path.

**If it fails:** most likely OOM or host-RAM pressure. Retry with
`--height 256 --width 256 --num-frames 9`. This rung is the Wan2.1 deliverable, not a
prerequisite for the science — if Wan is stubborn, park it and go to Rung 2, which uses
CogVideoX.

### Rung 2 — PhaseLock on Wan (~15 min)

```bash
python scripts/inference.py --backend wan21_t2v_1_3b \
    --prompt "a red ball bouncing on a table" --output /tmp/wan_pl.mp4 \
    --few-steps 2 --full-steps 8 --save-few /tmp/wan_few.mp4
```

**Passes if:** it completes without a shape error and both files are written.

**Exercises:** the whole point of the Wan port — `BCTHW` canonicalisation inside
`LatentDeltaGuidance`, per-channel normalisation, deterministic `.mode()` encode.

**If it fails with a shape error**, that is the layout bug the port exists to fix and it
means canonicalisation is wrong somewhere. Check `to_canonical`/`from_canonical` against
the actual `latents.shape` the callback receives.

### Rung 3 — CogVideoX inversion, one clip (~10 min)

```bash
python scripts/run_detection.py --config configs/experiments/pilot_likephys.yaml \
    data__limit=1 inversion__reconstruction_check=true
```

**Passes if:** `reconstruction.json` appears with a plausible PSNR, and `statistics.csv`
has rows for all four sources.

**Exercises:** the single largest block of untested code — VAE encode, the inversion loop,
block hooks, pooling, `denoiser_state`, `renoise`, statistics, CSV writing.

**Predicted failure modes**, in order of likelihood:

1. **CPU-offload vs direct transformer call.** `invert()` calls
   `backend.transformer_forward()` directly rather than through `pipe.__call__`, so
   accelerate's offload hooks are exercised in an order diffusers never intended. Symptom:
   device-mismatch error, or absurd slowness. **Fix: `backend__offload=false`** —
   CogVideoX-5B in bf16 is ~10 GB and fits a 46 GB card easily. Do this first if anything
   looks wrong.
2. **`encode_prompt` outside `__call__`** — same class, same fix.
3. **Token-count mismatch in `pool_tokens`.** Raises `token count N does not match grid`.
   Means the grid computed from `token_grid` disagrees with what the block emits — check
   whether CogVideoX's `patch_size` is being applied twice.

**Record the per-step wall clock here.** Everything downstream is
`clips × num_steps × seconds-per-step`, and that number is currently unknown.

### Rung 4 — pilot detection, 2 pairs (~20 min)

```bash
python scripts/run_detection.py --config configs/experiments/pilot_likephys.yaml
python scripts/report.py /data/experiments/phaselock_geophys/pilot_likephys
```

**Passes if:** `signals.csv` is populated and `report.py` renders. Accuracies are
meaningless at n=2 — this checks plumbing only.

**Also check:** `trajectories/*.npz` load, and `drift_estimator` is `exact` for `latent`
and `empirical` for `hidden_states`.

### Rung 5 — DINOv2 correctness gate (~30 min)

```bash
python scripts/run_external.py --config configs/experiments/detection_likephys.yaml \
    data__limit=60
```

**This is the gate everything else depends on.**

**Passes if:** the best single-layer signal lands in **70–88%** — the published range is
77.6–80.8%, and the tolerance accounts for this being a different LikePhys release (800
pairs vs the paper's 650) and a smaller sample.

**If it lands near chance (45–55%),** stop. Something upstream is wrong and no internal
number will mean anything. Debug in this order:

1. **Temporal resampling destroying the signal.** Try `data__window=null` and raise the
   frame budget. LikePhys 60 → 49 frames should be nearly lossless, so this is unlikely
   here but is the first thing to rule out.
2. **Letterbox padding diluting the pooled feature.** A 512×512 source in a 480×720 frame
   is ~33% black bars, and those static pixels contribute nothing to displacement while
   still entering the spatial mean. It affects both members of a pair equally so pairwise
   accuracy *should* survive, but it lowers SNR. Worth testing a square-target variant.
3. **Wrong readout layer.** The sweep reports all layers; check whether *any* layer works
   before concluding the statistics are wrong.
4. **The statistics themselves.** Least likely — they are pinned against analytic
   trajectories — but check `metrics__residual_fit` and `metrics__ar_order`.

**If it lands in range:** proceed, and record which layer and statistic won. That is the
yardstick every internal number gets compared against.

### Rung 6 — generation smoke (~20 min)

```bash
python scripts/run_generation.py --config configs/experiments/generation_likephys.yaml \
    data__limit=1 generation__num_steps=8 --save-videos
```

**Passes if:** an mp4 is written and `candidates.csv` has sane motion-mask scores
(IoUs in [0,1], MSE small).

**Exercises:** `generate_with_probes` — the transformer forward hook, the
`callback_on_step_end` contract, and `combine_cfg`, none of which have ever run.

**Predicted failure modes:**

1. **Hook signature.** `register_forward_hook(..., with_kwargs=True)` needs torch ≥ 2.0
   (we have 2.13) and assumes diffusers calls the transformer with keyword arguments. It
   does — but if `hidden_states` arrives positionally the hook falls back to `args[0]`.
2. **CFG reassembly.** If `combine_cfg` raises "expected a batch of 1 or 2", the pipeline
   is batching differently than assumed. Sanity check: with `guidance_scale=1.0` there
   should be exactly one call per step and no combination at all.
3. **Silent wrongness, the dangerous one.** If the chunk order were reversed we would
   record the *unconditional* prediction and nothing would raise. Verify by generating the
   same clip at `guidance_scale=1.0` and at `6.0` and confirming the recorded `velocity`
   trajectories differ.

---

## Phase 2 — the experiments

Only after Rung 5 passes.

### 2.1 Sweeps first, on a small subset

The sweeps decide the settings for the expensive runs, so they come before them.

| sweep | command | decides |
|---|---|---|
| inversion steps | `inversion__num_steps=20,50,100` (three runs, `data__limit=24`) | **the load-bearing one.** The Invisible Hand reports probe accuracy collapsing 0.82 → 0.57 from 100 to 20 steps *while reconstruction still looks fine*. If our accuracy is flat across this sweep, either inversion is not working or the geometry is not reading the trajectory |
| AR order | `metrics__ar_order=2,3,4` | `H` is never stated in either paper |
| residual fit | `metrics__residual_fit=span,ridge,scalar` | the documented ambiguity |
| prompt | `inversion__prompt=""` vs a scenario prompt | whether the unconditional field carries the signal |

### 2.2 Main detection run

```bash
python scripts/run_detection.py --config configs/experiments/detection_likephys.yaml \
    data__limit=120 inversion__num_steps=<best from sweep>
python scripts/report.py /data/experiments/phaselock_geophys/detection_likephys --figures
```

120 balanced pairs, ~200 distinct clips after deduplication. Then scale to all 800 pairs if
the signal is there.

**The three cells that matter:**

- **`latent` vs `hidden_states`.** Geometry at chance on VAE latents while working on
  hidden states is the clean positive result — and says PhaseLock's latent delta operates
  in a representation carrying no physical signal.
- **Where in depth the peak sits.** The Invisible Hand found probes peaking in the middle
  third. Does the geometric readout agree?
- **`drift_*` vs `phi_*`.** Does the rate of change beat the value?

### 2.3 IntPhys2

```bash
python scripts/run_detection.py --config configs/experiments/detection_intphys2.yaml \
    data__window=null      # and 0.5, and 0.25
```

Expect this to be harder. 636 → 49 frames is a 13× decimation that can step straight over a
brief violation, which is exactly what the window sweep is testing.

### 2.4 Generation and step sweep

```bash
python scripts/run_generation.py --config configs/experiments/generation_likephys.yaml \
    generation__num_candidates=4 --save-videos
python scripts/run_step_sweep.py --config configs/experiments/step_sweep_likephys.yaml
```

For the step sweep, the reproduction check is `phase_difference_corr` at σ=16: PhaseLock
reports 0.358 at K=2 against 0.100 at K=50. Landing near that validates the spectral
implementation. Then the actual question — whether the *geometric* statistics show the same
K ordering, and whether it survives the blur.

---

## Phase 3 — correctness gates, consolidated

Nothing downstream of a failed gate is reportable.

| gate | check | criterion |
|---|---|---|
| G1 | CPU tests | `python -m pytest tests/ -q` → all pass |
| G2 | Wan layout | Rung 2 completes without a shape error |
| G3 | inversion sanity | reconstruction PSNR is plausible, not near-zero |
| G4 | **GeoPhys implementation** | DINOv2 on LikePhys in 70–88% |
| G5 | inversion fidelity | accuracy varies with `num_steps` — flatness means the trajectory is not being read |
| G6 | CFG reassembly | recorded `velocity` differs between `guidance_scale` 1.0 and 6.0 |
| G7 | spectral implementation | `phase_difference_corr` favours K=2 at σ=16, near 0.358 vs 0.100 |

G4 and G5 are the two that would invalidate the whole study. G4 says the metric is right;
G5 says we are actually measuring the model's trajectory rather than noise.

---

## Phase 4 — what is genuinely uncertain

Distinguishing "will crash" from "will quietly mislead" matters, because only the first
kind announces itself.

**Will crash, and that is fine** — offload/device handling, hook signatures, token-count
mismatches. All loud, all localisable, all cheap to fix.

**Could quietly mislead:**

1. **Inversion under an empty prompt.** If CogVideoX's unconditional velocity field is
   degenerate, hidden states may carry little signal, and a null result would be about the
   prompt rather than about geometry. Mitigation: the prompt sweep in 2.1.
2. **Letterbox dilution.** ~33% static black bars entering the spatial mean lowers SNR
   without changing the pairwise ranking. A weak-but-nonzero result could be this rather
   than the representation.
3. **Drift on hidden states is a coarse secant**, resolution-limited by `record_steps`. A
   null `drift_*` result on hidden states may be a resolution artefact. Raise
   `record_steps` before concluding anything.
4. **n=120 pairs and five statistics × ~42 blocks × 10 steps** is thousands of
   comparisons. Some will look significant by chance. Trust the *structure* — a smooth peak
   across neighbouring blocks and steps — over any single winning cell, which is exactly
   what the heatmaps are for.

---

## Phase 5 — deferred

- **Physics-IQ (Stage 8).** Dataset class and motion-mask metrics are built and tested;
  only the driver is missing. Deferred until the simulated-data story is settled.
- **Wan 14B backends.** Registered and layout-correct, never run, will not fit 46 GB.
- **True attention maps.** Only the attention sublayer *output* is recorded; the matrix is
  32760² per block at native resolution and fused SDPA never materialises it.

---

## Quick reference

```bash
# 0. is the GPU free?
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# 1-2. Wan deliverable
python scripts/inference.py --backend wan21_t2v_1_3b --prompt "a ball bouncing" \
    --output /tmp/wan.mp4 --full-steps 4 --no-phaselock
python scripts/inference.py --backend wan21_t2v_1_3b --prompt "a ball bouncing" \
    --output /tmp/wan_pl.mp4 --few-steps 2 --full-steps 8

# 3-4. inversion path  (add backend__offload=false at the first sign of trouble)
python scripts/run_detection.py --config configs/experiments/pilot_likephys.yaml \
    data__limit=1 inversion__reconstruction_check=true
python scripts/run_detection.py --config configs/experiments/pilot_likephys.yaml
python scripts/report.py /data/experiments/phaselock_geophys/pilot_likephys

# 5. THE GATE
python scripts/run_external.py --config configs/experiments/detection_likephys.yaml \
    data__limit=60

# 6. generation path
python scripts/run_generation.py --config configs/experiments/generation_likephys.yaml \
    data__limit=1 generation__num_steps=8 --save-videos

# then Phase 2
```
