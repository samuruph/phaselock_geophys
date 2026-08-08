# Results

What has been measured, what the numbers mean, and what should not yet be believed.

## Which model produced which number

**Every number in this document is model-specific.** An accuracy measured by inverting
with Wan says nothing about CogVideoX, and the two are never mixed in a table. Results are
filed under `<backend>/<dataset>/<stage>/` so the path carries the provenance.

| section | backend | mode | dataset | stage | status |
|---|---|---|---|---|---|
| §3–§6 (everything below) | **Wan2.1-T2V-1.3B** | T2V, inversion | LikePhys, 96 pairs | detection | **complete** |
| §3 external baseline | DINOv2-large (frozen) | encoder, no diffusion | LikePhys, 96 pairs | detection | **complete** |
| — | CogVideoX-5B-I2V | I2V, inversion | LikePhys | detection | in progress |
| — | CogVideoX-5B-I2V | I2V, generation | LikePhys | generation | in progress |

Written 2026-08-08 against
`/data/experiments/phaselock_geophys/wan21_t2v_1_3b/likephys/detection`, reproducible with
`python scripts/report.py <run_dir> --figures`.

### Why detection runs on Wan and generation on CogVideoX

Not a preference — a measured constraint. Inversion fidelity is load-bearing for
detection, and DDIM inversion of a Blender render is far off CogVideoX's training
distribution. On the same LikePhys clip:

| backend | VAE ceiling | inversion | gap |
|---|---|---|---|
| CogVideoX-5B-T2V (DDIM) | 47.4 dB | **6.9 dB** | 40.5 dB |
| Wan2.1-1.3B (flow matching) | 51.1 dB | **43.9 dB** | 7.2 dB |

Generation needs image-to-video, and the only I2V checkpoint that fits a 46 GB card is
CogVideoX-5B-I2V — Wan's I2V is 14B. Generation runs the model *forward*, so inversion
fidelity never enters and the constraint above does not apply. The cost is that detection
and generation sit on different models, which is why a CogVideoX-5B-I2V detection run is
in progress: without it, Stage 6 would have to pick its verifier from Wan numbers.

> **Status.** Stage 5 (detection on labelled pairs) is complete, including the temporally
> matched DINOv2 baseline. Stages 6–8 (generation, step sweep, IntPhys2) have not run.

---

## 1. What the experiment does

The question is not "does trajectory geometry detect implausible physics" — GeoPhys
already answered that for frozen external encoders. It is **which representation carries
the geometry**, and in particular whether a video diffusion model's own internals carry
it better than an external encoder does.

For each of 173 LikePhys clips (96 matched plausible/violated pairs):

1. decode, resample 60 → 81 frames, letterbox 512×512 → 480×832
2. VAE-encode → 21 latent frames × 16 channels (Wan's causal VAE folds 4 video frames
   into each latent after the first)
3. **invert** the flow-matching sampler from the clean latent toward noise, 50 steps,
   recording at 10 of them — a real video has no denoising trajectory, so one is
   manufactured by running the sampler backwards
4. capture four sources at each recorded step, spatially mean-pooled inside the forward
   hook to `(21, D)`
5. compute 12 scalars per trajectory
6. score each `(source, block, step, statistic, kind)` cell as a detector over the pairs

| source | probe cells | what it is |
|---|---|---|
| `hidden_states` | 30 blocks × 10 steps = 300 | the Invisible Hand signal |
| `latent` | 1 × 10 | `x_t` — the paper's negative control, and PhaseLock's space |
| `x0_hat` | 1 × 10 | the model's clean estimate |
| `velocity` | 1 × 10 | `u_θ`, the flow field |

Scoring is the GeoPhys rule verbatim: within a pair, the clip with the larger statistic is
called violated; accuracy is the fraction of pairs where that is correct. Confidence
intervals are 1000-resample bootstraps grouped by scenario, because LikePhys pairs within
a scenario share their `valid` clip and are not independent.

**Configuration.** Wan2.1-T2V-1.3B, bfloat16, 50-step inversion, empty prompt, guidance
off — so the recorded field is the unconditional velocity and no text confound enters the
comparison. **No PhaseLock guidance anywhere**; this is the plain baseline.

---

## 2. The metrics

### The five GeoPhys statistics

On a pooled trajectory `Z = (z̄₁ … z̄_T) ∈ R^{T×D}`:

```
v_t = z̄_{t+1} − z̄_t          s_t = ‖v_t‖        θ_t = ∠(v_t, v_{t+1})
a_t = z̄_{t+2} − 2z̄_{t+1} + z̄_t                  ε_t = z̄_{t+1} − ẑ_{t+1}

φ_speed = std({s_t})     φ_curv = mean({θ_t})    φ_ang  = std({θ_t})
φ_accel = mean({‖a_t‖²})                         φ_perr = mean({‖ε_t‖})
```

All five are oriented so **larger = less regular = less plausible**. `s_t` and `θ_t` are
per-frame intermediates, not statistics — the statistics are their temporal summaries.
Note `φ_accel` uses a *squared* norm and `φ_perr` an unsquared one; that asymmetry is the
paper's.

#### Notation: `z̄` versus `ẑ`

Easy to confuse, and they mean quite different things.

| symbol | name | what it is |
|---|---|---|
| `z̄_t` | **bar** — the pooled feature | The *observed* representation of frame `t`. Take the raw activation (a DiT hidden state, a VAE latent, a DINOv2 patch grid), average it over space, and you have one vector per frame. `z̄` is data. |
| `ẑ_t` | **hat** — the prediction | Where frame `t` *should have been*, predicted from the frames before it. `ẑ` is a forecast, never observed. |

So the trajectory `Z = (z̄₁ … z̄_T)` is what the clip actually did, and `ẑ_{t+1}` is what
the previous few frames implied it was about to do.

#### `φ_perr`: the prediction residual

`ε_t = z̄_{t+1} − ẑ_{t+1}` is the gap between the two — **how much the clip surprised a
short-horizon predictor fitted to its own recent past**. `φ_perr = mean({‖ε_t‖})` averages
that surprise over the clip.

The intuition for why it detects implausible physics: real dynamics are locally smooth, so
the next state is largely determined by the last few. A ball passing through a wall, an
object teleporting, or gravity reversing all break that determination, and the residual
spikes. Unlike `φ_curv` or `φ_speed`, which ask whether the trajectory *looks* regular in
isolation, `φ_perr` asks whether it is *self-consistent over time* — which is why it
transfers to representations that were trained to predict.

#### How `ẑ` is actually computed

The paper says it fits a linear autoregressive predictor `P̂_H : R^{H·D} → R^D` on past
windows. Taken literally that is unusable here: with `H = 3` and `D = 3072`, one clip
offers 10 windows to fit 9216 unknowns, so the system is underdetermined and the
in-sample residual is **exactly zero** for every clip, plausible or not. There is a
regression test asserting that degeneracy so the trap stays documented.

The default (`residual_fit="span"`) uses the paper's own *geometric* wording instead —
`ε_t` is the component of `z̄_{t+1}` orthogonal to the span of the previous `H` frames.
Per window:

1. take the `H = 3` preceding vectors `z̄_{t−2}, z̄_{t−1}, z̄_t`
2. form the **affine** span through them (their differences from `z̄_t`, so a
   constant-velocity trajectory is predicted exactly)
3. project `z̄_{t+1}` onto that span by least squares — the projection is `ẑ_{t+1}`
4. `ε_t` is the leftover perpendicular component

This is well-posed, training-free, per-clip, and needs no data beyond the clip itself.
`ridge` (a regularised version of the literal fit, with `λ` scaled by the Gram diagonal so
it is not swamped) and `scalar` (a single AR coefficient) are available for comparison.
`H` is never stated in the paper; default 3.

See [METHOD.md](METHOD.md) for the full derivation and the other three deviations.

### The three new metrics

The motivating observation: GeoPhys's velocity runs along the **frame** axis, the flow
model's along the **denoising** axis, and they are linked *exactly*, because the
frame-difference operator `D` and spatial mean-pooling are linear and therefore commute
with the sampler ODE:

```
d(D z̄)/dτ  =  D (dz̄/dτ)  =  D ū_θ(z, τ)
```

From which:

| metric | definition | reading |
|---|---|---|
| **geometric drift** | `ġ_σ(τ) = ⟨∇_z̄ φ_σ(z̄), ū_θ⟩` | `< 0`: this step is *regularising* the trajectory; `> 0`: eroding it |
| **transport alignment** | `ρ_f = cos((Dz̄)_f, (Dū)_f)` | is the step growing the motion already present, or rewriting it? |
| **erosion rate** | `‖Dū_f‖ / (‖Dz̄_f‖ + ε)` | how fast motion is restructured relative to how much exists |

Geometric drift is exact **only for the pooled VAE latent**, which is the actual ODE
state. `x0_hat` and `velocity` look like they qualify but do not — both depend on the
network output, so their τ-derivative drags in a Jacobian that is never formed. Those and
hidden states fall back to a finite difference across consecutive recorded steps, tagged
`empirical` in the schema so the two are never averaged together.

### The selection null — read this before reading any number

3800 signals were scored on 96 pairs. Reporting the best one is not a result; with that
many candidates a sizeable accuracy arises from noise alone.

So the labels are shuffled — which member of each pair is "violated" is randomised — every
signal is rescored, and the whole procedure is repeated 300 times. That gives the accuracy
a *selection process* reaches by chance:

| what you are reading | floor it must clear |
|---|---|
| a **maximum** over cells (diamonds in the figures) | **68.8%** |
| a **mean** over cells (bars in the figures) | **52.8%** |

They differ because a maximum selects for noise while a mean cancels it. Comparing a
maximum against the mean floor is the single easiest way to misread these results.

---

## 3. Correctness gate

Before any internal number: does this implementation reproduce GeoPhys on the path GeoPhys
actually published?

**DINOv2-large, layer 10, `φ_speed`, all 800 LikePhys pairs: 72.2%.**
Published single-backbone range: 77.6–80.8%.

Close enough to proceed, and the gap is expected — the local LikePhys release has 800 pairs
where the paper cites 650, so the scenario mix differs.

A second, qualitative check reproduces the paper's Figure 3 directly: on DINOv2, violated
sits above plausible at **every** readout layer for mean turning angle and angle
consistency (`figures/dinov2/04_signal_profile_depth.png`). That is their reported result,
reproduced.

---

## 4. Headline numbers

Pairwise accuracy, mean ± s.d. across probe cells, `n = 96` pairs.
**Bars must clear 52.8%; the best-cell column must clear 68.8%.**

### The five statistics on internal representations

**Backend: Wan2.1-T2V-1.3B**, inverted. LikePhys, 96 pairs.

| source | φ_perr | φ_accel | φ_curv | φ_ang | φ_speed |
|---|---|---|---|---|---|
| **DiT hidden states** (300 cells) | **72.1 ± 3.0** | 66.7 ± 6.7 | 49.3 ± 7.0 | 49.4 ± 5.6 | 43.7 ± 6.4 |
| **VAE latent `x_t`** (10) | **69.5 ± 1.9** | 63.0 ± 4.9 | 52.0 ± 4.3 | 46.8 ± 2.9 | 37.7 ± 4.8 |
| **flow velocity `u_θ`** (10) | **68.9 ± 4.1** | 66.1 ± 5.4 | 55.8 ± 3.6 | 50.1 ± 5.3 | 42.2 ± 4.8 |
| **clean estimate `x̂₀`** (10) | **65.5 ± 1.6** | 61.9 ± 6.3 | 59.8 ± 3.2 | 52.3 ± 7.3 | 46.5 ± 4.1 |

Best single cell overall: **hidden states, block 6, step 4, `φ_accel` — 85.4%**
[77.1, 92.7], against a 68.8% selection floor, `p < 0.0001`.

### The same five on the external DINOv2 baseline

**No diffusion model involved** — frozen DINOv2-large on the decoded frames.

Temporally matched: DINOv2 features averaged over each latent's 4 video frames, so both
paths have 21 trajectory points with the same spacing. Same 96 pairs. 25 readout layers.

| source | φ_perr | φ_accel | φ_curv | φ_ang | φ_speed |
|---|---|---|---|---|---|
| **DINOv2-large** (25 cells/stat) | 54.4 ± 4.9 | 63.9 ± 8.1 | 63.5 ± 5.1 | 64.5 ± 4.0 | **65.4 ± 6.7** |

Best DINOv2 cell: layer 7, `φ_speed`, **74.0%** over 125 cells.

### The three new metrics

**Backend: Wan2.1-T2V-1.3B.**

| metric | best source | mean | best cell |
|---|---|---|---|
| geometric drift (`φ_accel`) | `x0_hat` | 58.7 ± 10.5 | 75.0 |
| geometric drift (`φ_perr`) | `latent` / `velocity` | 58.0 | 68.8 |
| transport alignment | `latent` | 54.7 ± 3.9 | 59.4 |
| erosion rate | `latent` | 54.2 ± 8.3 | 66.7 |

---

## 5. What the numbers say

### 5.1 The signal is real, and it is carried by the prediction residual

`φ_perr` on hidden states averages **72.1%** across all 300 probe cells with a standard
deviation of **3.0**. That tight spread is the important part: it is not one lucky cell but
a property of nearly every block and denoising step. A best-of-N artifact does not look
like that.

`φ_accel` reaches a higher single cell (85.4%) but varies far more (±6.7), so it is the
more selection-sensitive of the two.

### 5.2 The five statistics swap roles between representations

`φ_speed` lands **below chance on every internal source** — 43.7%, 37.7%, 42.2%, 46.5%.
Below chance is not noise; it means the ranking is systematically inverted, and violated
clips have *more regular* pooled step sizes than plausible ones. `φ_curv` and `φ_ang` sit
at chance internally.

On DINOv2, at the same 21-point temporal resolution, the ordering is almost exactly
reversed:

| statistic | DiT hidden states | DINOv2 |
|---|---|---|
| `φ_perr` | **72.1** | 54.4 *(its worst)* |
| `φ_accel` | 66.7 | 63.9 |
| `φ_ang` | 49.4 | 64.5 |
| `φ_curv` | 49.3 | 63.5 |
| `φ_speed` | 43.7 *(its worst)* | **65.4** |

Each representation's **best** statistic is close to the other's **worst**. This was
initially suspicious — the unpooled DINOv2 run was 4× finer in time, and `φ_speed` is the
statistic most sensitive to sampling rate — so it could have been a pooling artifact. It
is not: the table above is temporally matched.

DINOv2 also spreads its signal fairly evenly across four statistics (63.5–65.4), while the
internal path concentrates it in one (72.1) and actively harms itself with another. The
GeoPhys five are not a portable set; *which* geometric channel carries plausibility is a
property of the representation, not of physics.

A plausible reading, not tested here: DINOv2 features encode appearance, so a trajectory
through them moves when the *image* changes, and irregular image change reads as
irregular motion — which `φ_speed` measures directly. Diffusion internals are optimised to
predict the next state, so their trajectory is closer to a dynamical model, and what
breaks under implausible physics is *predictability* — which is what `φ_perr` measures.

### 5.3 The VAE latent is not physics-free

The Invisible Hand reports linear probes at 48–53% on VAE latents — chance — and that
result is load-bearing for their argument. Here, **geometry on the same latents reaches
69.5 ± 1.9%** with `φ_perr`.

That is not a contradiction, it is a difference in readout: a linear probe asks whether
plausibility is linearly separable in latent space; `φ_perr` measures a nonlinear
functional of the whole trajectory. The signal is present in the VAE latent but not
linearly available.

**This matters for PhaseLock.** PhaseLock's latent delta `T(z)` *is* the GeoPhys
first-order velocity, computed in exactly this space. The result above says that space is
not empty — but it also says the first-order term is the *worst* place to look
(`φ_speed`, 37.7%), while the residual and acceleration terms PhaseLock never touches
carry 69.5% and 63.0%.

### 5.4 The new metrics do not pay off

Stated plainly: **geometric drift, transport alignment and erosion rate all
underperform the plain statistics they were built from.** The best drift cell reaches
58.7% mean where the plain `φ_perr` reaches 72.1%; alignment and erosion sit at 54.7% and
54.2% against a 52.8% floor.

The commutation identity is still correct and still the cleanest way to relate the two
velocity notions. But as *detectors*, on this dataset and this backend, the coupled
quantities carry less than the uncoupled ones. Two readings are plausible and this run
cannot separate them: either the flow field genuinely does not distinguish plausible from
violated trajectories, or the finite-difference estimator across only 10 recorded steps is
too coarse to resolve a derivative. Recording more steps would test the second.

### 5.5 Where it works and where it fails

Best cell (hidden states / b6 / s4 / `φ_accel`), broken out by scenario:

| | accuracy | n |
|---|---|---|
| `block_slide`, `cloth_drape`, `faucet`, `shadow` | **100%** | 8 each |
| `ball_collision`, `ball_drop`, `flag`, `pyramid`, `shadow_camera` | 87.5% | 8 each |
| `pendulum` | 75.0% | 8 |
| `fluid` | 62.5% | 8 |
| `river` | **50.0%** | 8 |

The failure mode is coherent: **fluids**. `river` is at chance, `fluid` near it, and
`non_conservation_fluid` scores 25% (n=4) across violation types. This makes physical
sense — every statistic here encodes "regular is plausible", and turbulent flow is
irregular *when correct*, so the prior points the wrong way.

Rigid bodies and discrete events are where it works. This also disposes of a concern I
had from watching the videos: `block_slide` has a very small moving object against 42%
black letterbox padding, and scores 100%, so spatial dilution is not limiting.

Per-violation counts are 3–5, too small to rank individually; the per-scenario numbers
(n=8) inherit the selection of the best cell and should be read as a pattern, not as
calibrated accuracies.

### 5.6 Depth and time

`φ_accel` on hidden states peaks at **blocks 3–22, step 4** — 13 of the top 20 signals are
`φ_accel` at step 4, spanning that band at 78–85%. A coherent band across most of the
network is a much stronger claim than a single cell.

The peak sits at the **input side** (block 6 of 30). The Invisible Hand reports its linear
probes peaking in the middle third. Geometry and linear decodability therefore do not peak
in the same place, which is a real difference between the two readouts rather than a
disagreement about the model.

Step 4 of 10 corresponds to a mid-noise level; the signal is weakest at both the clean and
the fully-noised ends.

---

## 6. Threats to these conclusions

Listed worst-first.

1. **The internal and external paths select from very different pools.** The matched
   comparison of *means* (§5.2) is sound. The comparison of *maxima* is not symmetric:
   the internal best cell is chosen from 1500 candidates and DINOv2's from 125, so the
   85.4% and 74.0% figures carry different selection burdens even though both clear
   their own null. Prefer the means when quoting a margin.

2. **Inverted video is not generated video.** GeoPhys evaluates real and generated clips
   through a feature extractor. This stage evaluates *inverted* real clips, which requires
   the recovered trajectory to be the one the model would have taken. Reconstruction is
   42.6 dB against a 49.7 dB VAE ceiling, which rules out a broken inversion but does not
   establish trajectory fidelity — the Invisible Hand explicitly reports probe accuracy
   collapsing 0.82 → 0.57 between 100 and 20 steps while reconstruction still looks fine.
   **`inversion__num_steps` has not been swept.**

3. **One backend, one dataset.** Wan2.1-1.3B on LikePhys. IntPhys2 has not run, and its
   636 → 81 frame decimation is a 7.85× subsample that can step over a brief violation
   entirely — unlike LikePhys, whose 60 → 81 is an *upsample* that drops nothing.

4. **96 pairs.** Individual cells carry ±8–10 point confidence intervals. The 300-cell
   means are far tighter, which is why they are the headline.

5. **Preprocessing is shared but not neutral.** Letterboxing leaves 42% of every frame
   black. Black bars are temporally constant so they contribute ~0 to `v_t` and largely
   cancel within a pair, and §5.5 gives direct evidence they are not limiting — but the
   DiT still attends over off-distribution hard edges. Measured alternative: native
   512×512 has zero padding and runs 40% faster, but inverts to 41.3 dB against a higher
   52.8 dB ceiling, so it is *further* from its ceiling. Not settled; deliberately parked.

---

## 7. Where this leaves the research question

**Does GeoPhys geometry transfer to internal representations?** Partly, and selectively.
Two of the five statistics transfer and one of those is strong and stable across the whole
network; two are useless and one is inverted. The transfer is not a property of "the
GeoPhys method" but of specific statistics.

**Which internal representation carries it best?** DiT hidden states, but the margin over
the VAE latent is smaller than the Invisible Hand's linear-probe results would predict
(72.1% vs 69.5%). The interesting finding is not the ranking but that the VAE latent is
*not* at chance under a geometric readout.

**Is the coupled-velocity family worth it?** On this evidence, no. Reported as a negative
result rather than dropped.

**Does internal beat external?** Yes, on the strongest channel of each, temporally
matched on the same 96 pairs: **72.1 ± 3.0** (`φ_perr` on hidden states, averaged over 300
cells) against **65.4 ± 6.7** (`φ_speed` on DINOv2, averaged over 25 layers). The best
single cells are 85.4% and 74.0%, though those are selected from unequal pools.

The more interesting answer is that the two do not merely differ in strength — they read
plausibility through *different geometric channels* (§5.2), and each is near-useless in
the other's channel. That points at combining them rather than choosing between them,
which nothing here has tested.

---

## 8. Reproducing

```bash
# the run behind this document
python scripts/run_detection.py --config configs/experiments/detection_likephys_wan.yaml

# tables + every figure
python scripts/report.py /data/experiments/phaselock_geophys/detection_likephys_wan --figures

# the correctness gate, temporally matched
python scripts/run_external.py --config configs/experiments/detection_likephys_wan.yaml \
    --temporal-pool latent output__name=gate_likephys_pooled
```

Figures land in `<run>/figures/`: `01_source_comparison.png` is the headline, and each
source has its own folder with the per-statistic, depth-profile, depth×time and
GeoPhys-Figure-3-analogue plots. `<run>/videos/` holds side-by-side mp4s for eyeballing
what the pipeline actually did.

See [RUNNING.md](RUNNING.md) for the full command surface and [METHOD.md](METHOD.md) for
the places the source papers could not be followed literally.
