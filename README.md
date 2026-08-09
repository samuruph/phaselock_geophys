# PhaseLock + GeoPhys: trajectory geometry in video diffusion internals

Two things live in this repository.

**PhaseLock** — the training-free method from *Physics in 2-Steps: Locking Motion Priors
Before Visual Refinement Erases Them*, now backend-agnostic so it runs on Wan2.1 as well
as CogVideoX.

**GeoPhys on internal representations** — the research question this repo exists to
answer. [GeoPhys](context/GEOPHYS_The_Geometry_of_Physical_Plausibility.pdf) shows that
physical plausibility is readable from five geometric statistics of a per-frame feature
trajectory, using *frozen external image encoders*. [The Invisible Hand of
Physics](<context/The Invisible Hand of Physics- When Video Diffusion Models Know More Than They Show.pdf>)
shows that plausibility is linearly decodable from a video diffusion model's *own DiT
hidden states*, and — crucially — that it is **absent from the VAE latent input**
(48–53%, chance).

So: **does GeoPhys geometry transfer to internal representations, and which one carries
it?**

That question matters for PhaseLock directly. PhaseLock's latent delta
`T(z) = z[2:F] − z[1:F−1]` **is** GeoPhys's first-order velocity `v_t = z̄_{t+1} − z̄_t`,
computed in VAE latent space — the one space reported as physics-free. PhaseLock
constrains the first-order term of a GeoPhys trajectory, never looks at curvature,
acceleration or the prediction residual, and does so in a representation that may not
carry the signal at all.

> **Scope.** PhaseLock is not the object of study here. Every experiment runs on plain
> baseline sampling with guidance off.

## Documentation

| | |
|---|---|
| **[docs/RUNNING.md](docs/RUNNING.md)** | **Start here to run anything.** Exact commands, what each stage reads and writes, CSV schemas, cost and storage, how to read the output, which sweeps matter. |
| [docs/METHOD.md](docs/METHOD.md) | What is measured and why, and the four places the papers could not be followed literally. |

## Install

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 301 tests, CPU only, no weights needed
```

## The five statistics

On a pooled per-frame trajectory `Z = (z̄₁ … z̄_T) ∈ R^{T×D}`:

```
v_t = z̄_{t+1} − z̄_t          s_t = ‖v_t‖        θ_t = ∠(v_t, v_{t+1})
a_t = z̄_{t+2} − 2z̄_{t+1} + z̄_t                  ε_t = z̄_{t+1} − ẑ_{t+1}

φ_speed = std({s_t})     φ_curv = mean({θ_t})    φ_ang = std({θ_t})
φ_accel = mean({‖a_t‖²})                          φ_perr = mean({‖ε_t‖})
```

Larger means less regular, hence less plausible. `s_t` and `θ_t` are per-frame
intermediates, not statistics — the statistics are their temporal summaries.

Two places the source material is unusable as literally written, both handled explicitly
and documented at the call site:

- **`φ_perr` is underdetermined.** Fitting `P_H : R^{H·D} → R^D` on one video's windows
  is underdetermined whenever `H·D` exceeds the window count, which it always does here
  (CogVideoX: 10 windows, 9216 unknowns), so the in-sample residual collapses to exactly
  zero. The default follows the paper's own *geometric* reading instead — the component
  of `z̄_{t+1}` orthogonal to the affine span of the previous `H` frames — which is
  well-posed and training-free. `ridge` and `scalar` fits are available for comparison.
  `H` is never stated in the paper; default 3.
- **`arccos` is the wrong formula numerically.** It loses roughly half its precision near
  0 and π, exactly where a near-straight trajectory sits, and its derivative diverges
  there. A perfectly straight float32 trajectory returns ~2e-4 rather than 0. The
  algebraically identical half-angle form is used instead, which matters because the
  flow-coupling metrics differentiate these statistics.

## Coupling the two velocities

GeoPhys's velocity runs along the **frame** axis; a flow sampler's runs along the
**denoising** axis. They are linked exactly, because the frame-difference operator `D`
and spatial mean pooling are linear and so commute with the sampler ODE:

```
d(D z̄)/dτ  =  D (dz̄/dτ)  =  D ū_θ(z, τ)
```

*The flow velocity of the GeoPhys motion field is the GeoPhys motion field of the flow
velocity.* From that follows the headline metric, **geometric drift**:

```
ġ_σ(τ) = ⟨ ∇_z̄ φ_σ(z̄(τ)), ū_θ(z, τ) ⟩
```

`ġ_σ < 0` means this denoising step is making the trajectory **more** geometrically
regular; `ġ_σ > 0` means it is **eroding** it — PhaseLock's erosion thesis stated per
step and generalised past the first-order term. The gradient goes through the statistic
only, never the transformer, so it is nearly free.

**One caveat, enforced in the schema.** The identity gives `dr/dτ = u` only when the
recorded signal *is* the ODE state, i.e. the pooled VAE latent. `x0_hat` and `velocity`
look like they should qualify but do not — both depend on the network's output, so their
τ-derivative drags in a Jacobian that is never formed. Hidden states are further removed
still. Those all fall back to a finite-difference estimator, tagged `empirical` so the
two are never averaged together.

## Layout

```
phaselock/
  backends/     LatentSpec + VideoBackend; CogVideoX (BTCHW) and Wan2.1 (BCTHW)
  datasets/     LikePhys, IntPhys2, Physics-IQ, and shared video preprocessing
  pipelines/    inversion (real video -> latent trajectory), generation, phaselock
  probes/       forward hooks, per-latent-frame pooling, trajectory storage
  encoders/     frozen DINOv2 -- the published GeoPhys path, kept as the yardstick
  metrics/      geophys, flow_geometry, spectral, motion_mask, scoring
  experiments/  detection, external, generation, step_sweep
configs/experiments/    pilot, detection_{likephys,intphys2},
                        generation_likephys, step_sweep_likephys
scripts/        run_inversion, run_external, run_generation, run_step_sweep,
                report, inference
```

Everything internal works in a canonical `(T, C, H, W)` latent, so guidance, probing and
the metrics have one implementation each rather than one per backend.

## Backends

| name | layout | native | notes |
|---|---|---|---|
| `cogvideox_5b_t2v` | BTCHW | 49f @ 8fps, 480×720 | inversion default |
| `cogvideox_5b_i2v` | BTCHW | 49f @ 8fps, 480×720 | generation default |
| `wan21_t2v_1_3b` | BCTHW | 81f @ 16fps, 480×832 | validated |
| `wan21_t2v_14b`, `wan21_i2v_14b_480p`, `wan21_i2v_14b_720p` | BCTHW | 81f @ 16fps | wired, **not validated** — will not fit a 46GB card |

Wan support required fixing three CogVideoX-specific assumptions that all failed
*silently*: the guidance unpacked `B,T,C,H,W` and would have differenced the channel
axis; the encoder used `vae.config.scaling_factor`, which is `null` for
`AutoencoderKLWan` (it normalises per-channel); and the VAE encode sampled the posterior,
making the motion prior irreproducible between PhaseLock's two stages.

## Datasets

| dataset | pairs | clip | note |
|---|---|---|---|
| LikePhys | **800** | 60f @ 30fps, 512² | 12 scenarios × 10 subgroups; primary — 60→49 frames is nearly 1:1 |
| IntPhys2 | **506** | 636f @ 60fps, 512² | secondary; 636→49 is a 13× decimation that can step over a violation |
| Physics-IQ | 198 scenarios | 4K @ 30fps | generation benchmark, deferred |

The local LikePhys release has 800 pairs where the paper cites 650, so absolute
accuracies will not match published numbers even with a correct implementation.

## Running

Each stage depends on the previous one being believable, so run them in order. Full detail
in [docs/RUNNING.md](docs/RUNNING.md).

```bash
# 0. Smallest end-to-end check: 2 pairs, 20 steps. Minutes, not hours.
python scripts/run_inversion.py --config configs/experiments/pilot_likephys.yaml

# 1. CORRECTNESS GATE. The same five statistics on frozen DINOv2 features, which must
#    land near GeoPhys's published 77.6-80.8% on LikePhys. Until this passes, nothing
#    measured on internal representations can be distinguished from noise.
python scripts/run_external.py --config configs/experiments/inversion_likephys_cog_t2v.yaml

# 2. Detection: geometry on internal representations. The main result.
python scripts/run_inversion.py --config configs/experiments/inversion_likephys_cog_t2v.yaml
python scripts/run_inversion.py --config configs/experiments/inversion_likephys_cog_t2v.yaml \
    data__limit=48 inversion__num_steps=100        # overrides; unknown keys raise

# 3. Read it: per-source comparison, ranked signals, block x step heatmaps
python scripts/report.py /data/experiments/phaselock_geophys/wan21_t2v_1_3b/likephys/inversion --figures

# 4. Generation on labelled data, and the step sweep with PhaseLock's blur control
python scripts/run_generation.py --config configs/experiments/generation_likephys.yaml
python scripts/run_step_sweep.py --config configs/experiments/step_sweep_likephys.yaml

# Wan2.1 smoke test on local weights
python scripts/inference.py --backend wan21_t2v_1_3b \
    --prompt "a ball bouncing on a table" --output /tmp/wan.mp4
```

Artefacts land in `{output.root}/{output.name}/`: `statistics.csv` (raw per-clip
measurements), `signals.csv` (pairwise accuracy per signal), `config.json` (provenance),
plus `external/`, `figures/`, `videos/` and `trajectories/` as applicable.

Extraction is resumable — clips already in `statistics.csv` are skipped.

## Status

The library and all six drivers are implemented and covered by 301 CPU tests.

**Stage 5 detection has completed** on LikePhys with Wan2.1-T2V-1.3B: 173 clips, 96 matched
pairs, 3800 scored signals, 50-step inversion, no PhaseLock guidance. The correctness gate
passed first (DINOv2 at 72.2% against the published 77.6-80.8%), so the internal numbers are
interpretable. Headline: prediction residual on DiT hidden states averages 72% across all 300
probe cells, against a 53% label-shuffled floor for a mean; the best single cell is
acceleration at block 6 / step 4, 85.4%, against a 69% floor for a maximum.

Two caveats travel with those numbers. The DINOv2 comparison is still the *unpooled* run,
which is 4x finer in time than the latents and so not directly comparable — rerun with
`--temporal-pool latent` before quoting any internal-versus-external margin. And Stages 6-8
(generation, step sweep, IntPhys2) have not run.

## What to watch for

- **Inversion fidelity is the load-bearing assumption.** The Invisible Hand reports probe
  accuracy collapsing from 0.82 to 0.57 when integration steps drop from 100 to 20, while
  the reconstruction still *looks* fine. A good reconstruction is necessary and nowhere
  near sufficient; sweep `inversion__num_steps`.
- **The VAE-latent control is the interesting cell.** Geometry landing at chance there
  while working on hidden states would be a clean positive result — and would say
  PhaseLock guides in a physics-free representation.
- **Blur is a confound in the step sweep.** A 2-step output is blurrier, so any feature
  trajectory could look "more regular" purely from having less texture to move. The
  σ ∈ {0, 8, 16} control is applied to *every* arm including the real reference, per
  PhaseLock Fig. 3a. If the K=2 advantage vanishes under blur, the ordering is a
  sharpness artefact and is not a physics result.

## License

Apache 2.0. Builds on diffusers, CogVideoX and Wan2.1.
