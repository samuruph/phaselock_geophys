# Running the analysis

Operational guide: exact commands, what each stage does, what it reads and writes, and how
to read the output. For *why* each quantity is measured, see [METHOD.md](METHOD.md).

> **Status.** Every stage below is implemented and covered by 246 CPU tests, but **no GPU
> run has completed yet**, so no timings here are measured — they are step-count
> arithmetic, labelled as estimates. The pilot in Stage 0 exists to replace them with real
> numbers on your box.

---

## 0. The run order

Each stage depends on the one before it being believable.

| | stage | command | needs GPU | why it comes here |
|---|---|---|---|---|
| 0 | pilot | `run_detection.py --config .../pilot_likephys.yaml` | yes | proves the path end to end in minutes |
| 1 | **correctness gate** | `run_external.py --config .../detection_likephys.yaml` | yes | **nothing downstream is interpretable until this passes** |
| 2 | detection | `run_detection.py --config .../detection_likephys.yaml` | yes | the main result |
| 3 | report | `report.py <run_dir> --figures` | no | turns CSVs into tables and heatmaps |
| 4 | generation | `run_generation.py --config .../generation_likephys.yaml` | yes | Stage 6 |
| 5 | step sweep | `run_step_sweep.py --config .../step_sweep_likephys.yaml` | yes | Stage 7, with the blur control |

Stage 1 is not optional. It runs the same five statistics on frozen DINOv2 features — the
representation GeoPhys designed them for — and must land near the published **77.6–80.8%**
single-backbone pairwise accuracy on LikePhys. If it does not, the statistics, the pairing,
the preprocessing or the scoring rule is wrong, and every internal-representation number is
noise you cannot distinguish from signal.

---

## 1. Prerequisites

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 246 tests, CPU only, no weights, ~2 s
```

**Data** (paths are the defaults; override with `data__root=...`):

| dataset | path | what is used |
|---|---|---|
| LikePhys | `/data/datasets/LikePhys-Benchmark/data` | 800 pairs, 12 scenarios × 10 subgroups |
| IntPhys2 | `/data/datasets/IntPhys2` | 506 pairs from `Main/metadata.csv` |
| Physics-IQ | `/data/datasets/physics-IQ-benchmark-verified` | 198 take-1 scenarios (Stage 8, deferred) |

**Weights:**

| backend | source | note |
|---|---|---|
| `cogvideox_5b_t2v` | `/data/weights/CogVideoX-5b-Diffusers` | inversion default — T2V, so no image conditioning to confound the trajectory |
| `cogvideox_5b_i2v` | `THUDM/CogVideoX-5B-I2V` (HF cache) | generation default |
| `wan21_t2v_1_3b` | `/data/weights/Wan2.1-T2V-1.3B-Diffusers` | validated path |
| `wan21_*_14b*` | HF | **registered but unvalidated** — will not fit a 46 GB card |
| DINOv2 | `facebook/dinov2-large` (~2.4 GB, auto-downloads) | the correctness gate |

---

## 2. How configuration works

One YAML per experiment in `configs/experiments/`, plus `section__key=value` overrides on
the command line:

```bash
python scripts/run_detection.py --config configs/experiments/detection_likephys.yaml \
    data__limit=48 inversion__num_steps=100 probe__block_stride=4
```

**Unknown keys raise.** A typo that silently falls back to a default produces a
plausible-looking result under settings nobody chose, and after a ten-hour run there is no
way to tell. `data__limitt=5` fails immediately, as does a stray key in the YAML.

The resolved config is written to `config.json` next to the results it produced, so any
output can be traced back to the exact settings.

### The sections

| section | key | meaning |
|---|---|---|
| `backend` | `name`, `model_id`, `dtype`, `offload` | which model; `model_id` overrides the registry default |
| `data` | `name`, `root`, `split`, `scenarios`, `violations`, `limit`, `seed` | which clips |
| | `categories` | Physics-IQ category filter (paired datasets use `scenarios` instead) |
| | `window` | centred fraction of each clip to keep — mainly for IntPhys2 |
| | `blur_sigma` | Gaussian blur applied to every arm |
| `probe` | `sources` | `hidden_states`, `latent`, `x0_hat`, `velocity`, `attention` |
| | `blocks` / `block_stride` | which DiT blocks to record |
| | `pooling` | `mean` (matches GeoPhys) or `flatten` |
| | `record_steps` | how many denoising steps to probe, spread evenly |
| | `save_trajectories` | keep full `(T, D)` arrays, not just the statistics |
| `inversion` | `num_steps`, `prompt` | integration steps; prompt is empty by default |
| | `reconstruction_check` | invert one clip, resample, report PSNR to `reconstruction.json` |
| `metrics` | `ar_order`, `residual_fit`, `ridge_lambda`, `bootstrap_resamples` | statistic options |
| `generation` | `num_steps`, `step_sweep`, `blur_sweep`, `guidance_scale`, `num_candidates`, `seed` | sampling |
| | `negative_prompt` | passed through to the pipeline; `null` by default |
| `output` | `root`, `name` | artefacts land in `{root}/{name}/` |

`limit` is a **balanced** draw across scenarios, not the first N — the first 60 LikePhys
pairs are all `ball_collision`, so a pilot using them would measure one kind of physics.

---

## 3. Stage 0 — pilot

```bash
python scripts/run_detection.py --config configs/experiments/pilot_likephys.yaml
```

2 pairs, 20 inversion steps, 5 recorded steps, every 8th block, trajectories saved. The
numbers are meaningless at n=2; the point is to confirm the path runs and to **measure the
per-step cost on your GPU** so the estimates below can be replaced with real ones.

Time one clip, then everything else is arithmetic:

```
detection wall-clock  ≈  (distinct clips) × (inversion num_steps) × (seconds per step)
```

---

## 4. Stage 1 — the correctness gate

```bash
python scripts/run_external.py --config configs/experiments/detection_likephys.yaml
```

Decodes each clip, resamples it to the **backend's** native frame count (so the external
and internal paths see identical temporal sampling), runs frozen DINOv2 over the frames,
mean-pools patch tokens per frame into a `(F, D)` trajectory, and computes the five
statistics at **every layer** — all layers come from one forward pass, so the readout sweep
is free.

Reads: the dataset. Writes to `{root}/{name}/external/`:

| file | contents |
|---|---|
| `external_statistics.csv` | `sample_id, label, scenario, violation, encoder_layer, phi_*` — one row per clip per layer |
| `external_signals.csv` | `layer, statistic, accuracy, ci_low, ci_high, auc, n_pairs` |

Prints a layer × statistic accuracy table, the best readout, and an explicit verdict:

```
best readout: layer 12 on 'ang' -> 79.2% [74.0, 84.0], AUC 0.810 (n=24)
correctness gate: consistent with the published 77.6-80.8% single-backbone range
```

If instead it says `OUTSIDE the published range`, stop and debug before running anything
else. The most likely causes, in order: temporal resampling destroying the violation (check
`data__window`), the wrong readout layer, or too few pairs for the estimate to mean
anything.

Cost driver: one DINOv2 forward per frame. No diffusion model is loaded, so this is by far
the cheapest GPU stage.

---

## 5. Stage 2 — detection

```bash
python scripts/run_detection.py --config configs/experiments/detection_likephys.yaml
```

### What happens per clip

```
decode  ──▶  temporal window  ──▶  uniform resample to 49 frames  ──▶  letterbox 480×720
                                                                              │
                                                                       [optional blur σ]
                                                                              │
                                                              VAE encode (posterior mode)
                                                                              │
                                                          (13, 16, 60, 90) normalised latent
                                                                              │
                                            invert the sampler: 50 steps, data ──▶ noise
                                                       recording at 10 evenly spaced steps
                                                                              │
                    ┌─────────────────────────────────────────────────────────┤
                    ▼                                                          ▼
        hidden states per block                              latent / x0_hat / velocity
        (1, 13·30·45, 3072)  ≈100 MB                            (13, 16, 60, 90)
                    │  pooled inside the hook                              │  mean over h,w
                    ▼                                                       ▼
            (13, 3072) ≈80 KB                                        (13, 16)
                    └──────────────────▶  five statistics + flow coupling  ◀──┘
```

Pooling happens **inside the forward hook** — that 100 MB → 80 KB reduction is the only
reason recording every block at every step is affordable.

**Deduplication matters.** LikePhys pairs every violation against one valid clip per
subgroup, so 800 pairs cover only 920 distinct videos. Clips are inverted once each, not
once per pair; inverting per pair would waste roughly 40% of a multi-hour run.

**Resumable.** Clips already present in `statistics.csv` are skipped, so an interrupted run
picks up where it stopped. Use `--score-only` to re-score without re-inverting.

**Reconstruction check.** With `inversion__reconstruction_check=true`, the first clip is
inverted, its recovered noise resampled forward, and the result compared to the source:

```
reconstruction check on ball_drop/0/valid: PSNR 28.4 dB at 50 steps
```

This rules out an inversion that is simply *broken*. It cannot tell you the trajectory is
good enough — the Invisible Hand reports probe accuracy collapsing 0.82 → 0.57 between 100
and 20 steps while the reconstruction stays visually faithful. Only the
`inversion__num_steps` sweep answers that. Written to `reconstruction.json`.

### Outputs — `{root}/{name}/`

| file | contents |
|---|---|
| `config.json` | the resolved config, for provenance |
| `statistics.csv` | one row per `(clip, source, block, step)` — the raw measurements |
| `signals.csv` | one row per `(source, block, step, statistic, kind)` — pairwise accuracy, CI, AUC |
| `trajectories/*.npz` + `*.json` | full pooled trajectories, only if `probe.save_trajectories` |
| `figures/*.png` | written by `report.py --figures` |

`statistics.csv` columns:

```
sample_id, label, group, scenario, violation,      # what clip
source, block, step, tau,                          # where in the model
phi_speed, phi_curv, phi_ang, phi_accel, phi_perr, # the five statistics
drift_speed, ... drift_perr, drift_estimator,      # rate of change under the flow
alignment, erosion                                 # first-order coupling (latent only)
```

The schema is fixed, and absent values are written blank rather than omitted: the last
recorded step legitimately has no drift, and only the latent carries `alignment`/`erosion`.
`drift_estimator` is `exact` or `empirical` — **never aggregate across the two**.

`signals.csv` has `kind ∈ {phi, drift}`: `phi_curv` scored as a detector is a different
signal from `drift_curv`, and both are reported.

### Cost and storage

Per clip: `num_steps` transformer forwards (recording adds no forwards, only storage).

Storage, CogVideoX-5B with `record_steps=10`:

| `block_stride` | blocks recorded | rows in `statistics.csv` per clip | `trajectories/` per clip |
|---|---|---|---|
| 1 | 42 | 450 | ≈ 34 MB |
| 4 | 11 | 140 | ≈ 9 MB |
| 8 | 6 | 90 | ≈ 5 MB |

`latent`, `x0_hat` and `velocity` are `(13, 16)` — a few KB, negligible. Only hidden states
cost anything. With `save_trajectories: false` (the default), just the CSV is kept, at
roughly 40 KB per clip.

---

## 6. Stage 3 — reading the results

```bash
python scripts/report.py /data/experiments/phaselock_geophys/detection_likephys --figures
python scripts/report.py <run_dir> --statistic curv --source hidden_states --kind drift
```

Four blocks of output.

**The correctness gate**, restated up front, including a loud note if it was never run.

**Best signal per source** — the comparison the project exists to make:

```
source           block  step   statistic   kind              accuracy
hidden_states       24     4        curv    phi       80.5% [75.5, 85.5]
latent               -     4       accel    phi       52.9% [47.9, 57.9]
```

**Ranked signals**, top N by pairwise accuracy with CI and AUC.

**Block × step heatmaps**, ASCII in the terminal and PNG with `--figures`:

```
hidden_states / phi_ang   (accuracy 48.2% ' ' .. 80.5% '@')
   block | steps 0 -> 8
       0 |        max 51.7%
      12 | :-:--  max 59.8%
      24 | %%%@%  max 80.5%
      36 | .:...  max 55.5%
```

This is the geometric analogue of the Invisible Hand's Figure 4, which found linear-probe
accuracy peaking in the middle third of the network. Whether the *geometric* readout peaks
in the same place is a question about how physical information is organised, not just about
a readout. PNG heatmaps are centred on chance (50%) so above and below read differently at
a glance.

### The cells to look at first

- **`latent` vs `hidden_states`.** The Invisible Hand reports linear probes at chance
  (48–53%) on VAE latents. Geometry *also* landing near chance there while working on
  hidden states is a clean positive result — and says PhaseLock's latent delta operates in a
  representation carrying no physical signal.
- **Where in depth the signal peaks**, and whether it survives across denoising time.
- **`drift_*` vs `phi_*`.** Does the rate of change separate pairs better than the value? A
  violated clip should be one the model's own flow is actively fighting.

---

## 7. Stage 4 — generation

```bash
python scripts/run_generation.py --config configs/experiments/generation_likephys.yaml
python scripts/run_generation.py --config ... generation__num_candidates=4 --save-videos
```

Conditions CogVideoX-5B-I2V on the **first frame of each LikePhys `valid` clip**, so the
real continuation is a physically plausible reference. Sampling is plain baseline —
PhaseLock guidance is never applied.

Each generation is scored two independent ways:

1. **Fidelity** to the real continuation via the Physics-IQ motion-mask family. LikePhys is
   rendered with a static camera, so the protocol's assumptions hold directly.
2. **The Stage 2 detector** applied to the generation itself.

Having both matters: (1) needs no assumption that geometry measures physics, so it
arbitrates when they disagree.

With `num_candidates > 1`, candidates are ranked by a verifier score
(`--verifier-source`, `--verifier-statistic`) and compared against the no-verifier baseline
and the oracle:

```
  no verifier (first draw)  0.4120
  selected                  0.4780   (+0.0660)
  oracle (upper bound)      0.5310   (+0.1190 headroom)
  closed 55.5% of the gap to the oracle
```

Writes `candidates.csv` (`sample_id, scenario, seed, num_steps, blur_sigma, spatial_iou,
spatiotemporal_iou, weighted_spatial_iou, mse, raw_score, path`), `selection.csv`, and
`videos/*.mp4` with `--save-videos`.

Cost: `clips × candidates × num_steps` transformer forwards.

---

## 8. Stage 5 — step sweep and the blur control

```bash
python scripts/run_step_sweep.py --config configs/experiments/step_sweep_likephys.yaml
```

Generates each clip at **K ∈ {2, 10, 30, 50}** from the same seed with guidance off, then
measures three things at every `(K, σ)` cell with **σ ∈ {0, 8, 16}** blur applied to *both*
the generation and the real reference.

PhaseLock's claim is that a 2-step output is more physically consistent than a 50-step one
(Physics-IQ 34.02 → 30.82) while visual quality moves the other way. The obvious confound is
that a 2-step output is just blurrier, so any feature trajectory could look "more regular"
from having less texture to move. Blurring only the sharp arm would test a different
hypothesis; the paper blurs all arms and checks the *ranking* survives.

The three measurements are deliberately not interchangeable:

| measurement | what it is | why |
|---|---|---|
| DINOv2 GeoPhys `phi_*` | the published external path | the primary result |
| `phase_difference_corr` | frame-wise 2D FFT, inter-frame phase difference, Pearson r vs the reference | PhaseLock's own Fig. 3a metric — doubles as a reproduction check (they report 0.358 at K=2 vs 0.100 at K=50, at σ=16) |
| `raw_score` | motion-mask fidelity to the real continuation | ground-truth based, so it arbitrates |

Writes `sweep.csv` and prints a survival table:

```
metric                      sigma       K=2      K=50       gap favoured
phi_curv                        0    0.3120    0.4410    0.1290      yes
phi_curv                        8    0.3380    0.4290    0.0910      yes
phi_curv                       16    0.4020    0.3910   -0.0110       no      <-- sharpness
phase_difference_corr          16    0.3580    0.1000    0.2580      yes
```

**A `no` at σ=16 is a real and reportable outcome**, not a failure: it says that geometric
statistic was measuring sharpness rather than physics. It is informative precisely because
PhaseLock's spectral metric *does* survive the same control, so the two families separate.

---

## 9. Output tree

```
/data/experiments/phaselock_geophys/
├── pilot_likephys/
│   ├── config.json
│   ├── statistics.csv
│   ├── signals.csv
│   └── trajectories/*.npz + *.json
├── detection_likephys/
│   ├── config.json  statistics.csv  signals.csv
│   ├── external/    external_statistics.csv  external_signals.csv
│   └── figures/     <source>_<kind>_<statistic>.png
├── generation_likephys/
│   ├── config.json  candidates.csv  selection.csv
│   └── videos/*.mp4
└── step_sweep_likephys/
    ├── config.json
    └── sweep.csv
```

Change `output.root` / `output.name` to separate runs. Two runs sharing a `name` will
**append** to `statistics.csv` — which is what makes resuming work, but means a config
change under the same name silently mixes settings. Rename when you change anything.

---

## 10. How the metrics are computed

Full derivations and the places the papers were ambiguous are in [METHOD.md](METHOD.md).
The computational summary:

### The five statistics

Every source reduces to a trajectory `Z ∈ R^{T×D}` — one vector per latent frame. Then:

```
v_t = z_{t+1} − z_t                                   (T−1, D)   displacement
s_t = ‖v_t‖₂                                          (T−1,)     per-frame speed
θ_t = 2·atan2(‖v̂_t − v̂_{t+1}‖, ‖v̂_t + v̂_{t+1}‖)        (T−2,)     per-frame turning angle
a_t = v_{t+1} − v_t                                   (T−2, D)   acceleration
ε_t = z_{t+1} − proj_{span(z_{t−H+1..t})}(z_{t+1})     (T−H,)     AR residual

φ_speed = std({s_t})        φ_curv = mean({θ_t})       φ_ang = std({θ_t})
φ_accel = mean({‖a_t‖²})    φ_perr = mean({‖ε_t‖})
```

Note `φ_accel` uses the **squared** norm and `φ_perr` the **unsquared** one. `s_t` and `θ_t`
are per-frame intermediates; the statistics are their temporal summaries — which is why
`speed` is a standard deviation, not a mean.

Two deliberate deviations, both because the literal version does not work:

- **`φ_perr`**: fitting `P_H : R^{H·D} → R^D` on one video's windows is underdetermined
  (CogVideoX: 10 windows, 9216 unknowns) so the in-sample residual is *identically zero*.
  The default is the paper's own geometric reading — projection onto the affine span of the
  previous `H` frames. `residual_fit: ridge|scalar` for comparison.
- **`θ_t`**: `arccos` loses half its precision near 0 and π, exactly where a near-straight
  trajectory sits, and its gradient diverges there. The algebraically identical half-angle
  form above is used instead.

### Flow coupling

`D` (frame difference) and mean pooling are linear, so they commute with the sampler ODE:
`d(Dz̄)/dτ = D ū_θ`. Hence

```
ġ_σ(τ) = ⟨ ∇_z̄ φ_σ(z̄), ū_θ ⟩          negative ⇒ the step is regularising the trajectory
```

taken through the statistic only, never the transformer. **Exact only for `latent`**, which
is the ODE state; `x0_hat` and `velocity` depend on the network output too, so their
τ-derivative would need a transformer Jacobian. Those use a finite difference across
recorded steps, tagged `empirical`.

### Scoring

For a matched pair, `δ = φ(violated) − φ(plausible)`, positive when correctly ordered
(every statistic is larger for less regular motion). Ties count 0.5. Confidence intervals
are a 1000-resample bootstrap **grouped by scenario**, because LikePhys shares one valid
clip across a subgroup's violations and those pairs are not independent. AUC is reported
alongside because pairwise accuracy tests only within-pair ranking, while AUC tests whether
one global threshold separates the classes — the stronger claim, and the one that matters
if the score is ever used as a verifier.

---

## 11. Sweeps worth running

| sweep | override | why |
|---|---|---|
| inversion steps | `inversion__num_steps=20,50,100` | **the load-bearing assumption.** The Invisible Hand reports probe accuracy collapsing 0.82 → 0.57 from 100 to 20 steps while reconstruction still looks fine |
| AR order | `metrics__ar_order=2,3,4` | `H` is never stated in either paper |
| residual fit | `metrics__residual_fit=span,ridge,scalar` | the ambiguity above |
| IntPhys2 window | `data__window=null,0.5,0.25` | 636 → 49 frames is a 13× decimation that can step over a violation |
| pooling | `probe__pooling=mean,flatten` | mean matches GeoPhys; flatten keeps spatial detail |

---

## 12. Gotchas

- **`run_external.py` first.** An internal number without the gate is uninterpretable.
- **Appending under a changed config.** Same `output.name` appends to `statistics.csv`.
  Rename when you change settings.
- **`drift_estimator`.** Never average `exact` and `empirical` rows together.
- **14B Wan backends** load and are layout-correct but have never been run; they do not fit
  a 46 GB card.
- **`attention`** records the attention sublayer *output*, not attention matrices — at
  native resolution a single block's video self-attention is 32760², about 10⁹ entries, and
  both backends use fused SDPA which never materialises it.
- **LikePhys here is 800 pairs**, where the paper cites 650. Different release; absolute
  numbers will not match published ones even with a correct implementation.
- **Everything is correlational.** Geometry separating violated from plausible says the
  representation carries a usable statistical signature, not that the model represents
  physics.
