# Method

What is measured, how, and where the source papers had to be interpreted.

> **Scope.** Definitions and derivations only — self-contained, with no numbers from any
> run. For measured results see [RESULTS.md](RESULTS.md); to run anything see
> [RUNNING.md](RUNNING.md). Links into the code are a convenience, not a dependency:
> everything here can be reimplemented from this document alone.

## 1. The question

GeoPhys establishes that physical plausibility is readable from the geometry of a
per-frame feature trajectory produced by a **frozen external image encoder**. The
Invisible Hand establishes that plausibility is linearly decodable from a video diffusion
model's **own DiT hidden states**, recovered for real videos by inverting the sampler, and
that the same signal is **absent from the VAE latent input** (48–53%, chance).

Neither paper crosses over. This repo asks whether GeoPhys's *geometric readout* works on
the *internal representations* the Invisible Hand identified, and which representation
carries it.

The answer has a direct consequence for PhaseLock. Its latent delta operator
`T(z) = z[2:F] − z[1:F−1]` is, term for term, GeoPhys's first-order velocity
`v_t = z̄_{t+1} − z̄_t`. PhaseLock therefore:

1. constrains only the **first-order** term of a GeoPhys trajectory, and
2. does so in **VAE latent space**, the representation reported as physics-free.

If geometry works on hidden states but not on latents, that is a concrete statement about
where the method is operating.

## 2. Pipeline

```
video ──decode──▶ temporal window ──▶ uniform resample ──▶ letterbox ──▶ [blur σ]
                                                                            │
                                                                       VAE encode
                                                                            │
                                                            invert sampler, record
                                                                            │
                            ┌───────────────────────────────────────────────┤
                            ▼                                               ▼
                    per-latent-frame pooling                        DINOv2 (external)
                            │                                               │
                            └──────────────▶ five statistics ◀──────────────┘
                                                    │
                                        pairwise accuracy + AUC + CI
```

### Preprocessing

Source clips and model inputs disagree. LikePhys is 60 frames at 30 fps and 512²;
IntPhys2 is 636 frames at 60 fps; CogVideoX wants 49 frames at 480×720.

- **Temporal resampling** is nearest-index, never interpolation. Blending neighbouring
  frames manufactures motion blur, which the geometric statistics would read as smoother
  dynamics — biasing precisely the quantity under study.
- **Spatial handling** is letterbox, not stretch. Squashing 512² into 480×720 changes
  every apparent velocity by a different factor per axis. Padding is identical within a
  matched pair, so it cannot affect a within-pair comparison.
- **IntPhys2's 13× decimation** (636 → 49) is the main correctness risk in the whole
  pipeline: a brief violation can fall between sampled frames. `data.window` narrows to
  the centre of the clip, trading coverage for temporal resolution. Sweep it.

### Inversion

Diffusion models expose no latent trajectory for a video they did not generate, so the
internal states of a *real, labelled* clip are never produced. Following the Invisible
Hand, we integrate the learned velocity field backward from the clean latent to noise.

The exact reverse of an Euler step is implicit and needs an iterative solver per step. The
explicit approximation — evaluate the denoiser at the known endpoint, then step — costs
one network evaluation per step, the same as forward sampling.

Both backends reduce to two operations: estimate `(x0, eps)` at the current point, then
place that pair at the next noise level. So one loop serves both parameterisations:

| | recover `(x0, eps)` | renoise to level `t` |
|---|---|---|
| CogVideoX (VP, v-pred) | `x0 = √ᾱ·z − √(1−ᾱ)·v`, `ε = √ᾱ·v + √(1−ᾱ)·z` | `√ᾱ_t·x0 + √(1−ᾱ_t)·ε` |
| Wan (rectified flow) | `x0 = z − σ·v`, `ε = z + (1−σ)·v` | `(1−σ_t)·x0 + σ_t·ε` |

Sampling is the same loop toward *lower* noise, which is what makes the reconstruction
check meaningful.

**Inversion fidelity is the load-bearing assumption.** The Invisible Hand reports probe
accuracy collapsing 0.82 → 0.57 when steps drop 100 → 20, while the reconstruction still
looks fine. Reconstruction is necessary and nowhere near sufficient. Sweep
`inversion__num_steps` over {20, 50, 100}.

The inversion prompt is empty with CFG off, so the recorded trajectory is the
unconditional velocity field and no text confound enters a plausible-vs-violated
comparison.

### Pooling

GeoPhys pools spatially to one vector per frame. The same reduction applied to internals
is what lets the geometry transfer.

Pooling happens **inside the forward hook**. A raw CogVideoX hidden state is
`(1, 13·30·45, 3072)` — about 100 MB in fp16; the pooled trajectory is `(13, 3072)`, about
80 KB. Recording every block at every step is only affordable because of that.

Both backends patchify with `patch_size=(1, 2, 2)` and flatten row-major `(t, h, w)`, so
each latent frame owns a contiguous, equal-length run of tokens and the reshape is exact.
A backend with temporal patching would break this; the tests assert `patch_size[0] == 1`.

## 3. The statistics

On a pooled trajectory `Z ∈ R^{T×D}`:

| statistic | definition | reads |
|---|---|---|
| `φ_speed` | `std({‖v_t‖})` | erratic step sizes — jumping, disappearing |
| `φ_curv` | `mean({θ_t})` | how much the trajectory bends |
| `φ_ang` | `std({θ_t})` | whether the bending is consistent |
| `φ_accel` | `mean({‖a_t‖²})` | full second-order instability. **squared** |
| `φ_perr` | `mean({‖ε_t‖})` | departure from local linearity. **unsquared** |

`s_t` and `θ_t` are per-frame intermediates. The statistics are their temporal summaries —
a distinction easy to lose, and the reason `speed` is a standard deviation rather than a
mean.

### Two deviations from the papers, both deliberate

**`φ_perr` as written is degenerate.** The paper fits `P_H : R^{H·D} → R^D` on one video's
past windows. With `H·D ≫ T` that system is underdetermined and the in-sample residual is
*identically zero* — for CogVideoX, 10 windows against 9216 unknowns. The statistic would
carry no signal at all. The paper's own geometric sentence is well-posed and
training-free: `ε_t` is "the component of `z̄_{t+1}` orthogonal to the H-step linear span".
That is the default (`fit="span"`). `ridge` (dual-form, scale-relative penalty) and
`scalar` (coefficients shared across feature dimensions) are implemented for comparison.
**`H` is never stated in either paper**; default 3, sweep {2, 3, 4}.

**`arccos` is numerically wrong for this use.** It loses roughly half the available
precision near 0 and π — exactly where a near-straight trajectory sits — and its
derivative diverges there. A perfectly straight float32 trajectory returns ~2e-4 instead
of 0, and its gradient is unusable. The algebraically identical half-angle form
`θ = 2·atan2(‖â−b̂‖, ‖â+b̂‖)` is well conditioned across the whole range. This matters
because the flow-coupling metrics differentiate `φ_curv` and `φ_ang`.

One consequence worth knowing: a perfectly straight trajectory is a **kink** minimum of
curvature, not a smooth one — `θ` grows like `|δ|` near zero. Its directional derivative
is one-sided and strictly positive in every direction, so the drift value autograd returns
exactly at the kink is an artefact of the epsilon guard, not a slope. A static scene
produces such a trajectory, so this is a real input, not a corner case.

## 4. Coupling the two velocities

GeoPhys's velocity runs along the **frame** axis. A flow sampler's runs along the
**denoising** axis. The frame-difference operator `D` and spatial mean pooling are both
linear, so they commute with the sampler ODE:

```
d(D z̄)/dτ  =  D (dz̄/dτ)  =  D ū_θ(z, τ)
```

*The flow velocity of the GeoPhys motion field is the GeoPhys motion field of the flow
velocity.* Three metrics follow, all free because `u_θ` is already evaluated.

**Geometric drift** (headline):

```
ġ_σ(τ) = ⟨ ∇_z̄ φ_σ(z̄(τ)), ū_θ(z, τ) ⟩
```

`ġ_σ < 0` — this step is making the trajectory more geometrically regular.
`ġ_σ > 0` — it is eroding it. PhaseLock's erosion thesis, stated per step and generalised
past the first-order term. The gradient goes through the statistic only, a `(T, D)`
tensor, never through the transformer.

**Transport alignment** `ρ_f = cos((D z̄)_f, (D ū)_f)` — is the step *growing* the motion
already present or *rewriting* it? The exact first-order special case, and the quantity
that maps one-to-one onto PhaseLock's latent delta.

**Erosion rate** `‖D ū‖ / ‖D z̄‖` — how fast motion is restructured relative to how much
exists.

### Which sources support the exact form

The identity gives `dr/dτ = u` **only when the recorded signal is the ODE state**, i.e.
the pooled VAE latent.

`x0_hat` and `velocity` look like they should qualify and do not: both are functions of
the network's *output* as well as the state, so differentiating them in `τ` drags in a
Jacobian of the transformer that is never formed. Hidden states are further removed still.

Those all use the finite-difference estimator across consecutive recorded steps, which
inherits the recording stride as its resolution — a coarse secant, not a tangent. Every
record is tagged `exact` or `empirical`, and the two are never averaged together.

This is a real limitation: the headline metric is exact precisely on the source
(`latent`) that the Invisible Hand says carries no signal, and approximate on the source
(`hidden_states`) that does.

## 4b. The full signal inventory

Every scored signal is one number per clip, addressed by
**`(source, block, step, kind, statistic)`**. Nothing else is scored, and nothing scored
is missing from this table.

### The four axes

| axis | values | what it selects |
|---|---|---|
| **source** | `hidden_states`, `latent`, `x0_hat`, `velocity` | *which tensor* inside the model the trajectory is read from |
| **block** | `0 … L-1` for hidden states, `-1` otherwise | *how deep* in the transformer. Wan-1.3B has 30 blocks, CogVideoX-5B has 42. The other three sources exist once per step, not per block |
| **step** | `0 … 9` | *when* along the denoising trajectory, `step 0` = clean, `step 9` ≈ pure noise. `tau = 1 - t/1000` is recorded alongside |
| **kind** | `phi`, `drift`, `coupling` | *what kind of quantity* — a value, its rate of change, or a flow-coupling measure |

### What each source is

| source | tensor | shape after pooling | why it is in the study |
|---|---|---|---|
| `hidden_states` | output of DiT block `b` at step `s` | `(T, D)`, D = 1536 (Wan-1.3B) / 3072 (CogVideoX-5B) | the signal "The Invisible Hand of Physics" found linearly decodable |
| `latent` | the sampler state `x_t` itself | `(T, C)`, C = 16 | the paper's **negative control** — reported at chance for linear probes. Also PhaseLock's space |
| `x0_hat` | the model's estimate of the clean latent from `x_t` | `(T, C)` | what the model currently "believes" the video is |
| `velocity` | `u_θ`, the PF-ODE drift | `(T, C)` | the flow field itself, and the input to every coupling metric |

All four are spatially mean-pooled **inside the forward hook**, per latent frame, giving
one vector per latent frame. A raw hidden state is ~100 MB; pooled it is ~64 KB.

### The 12 quantities computed on each trajectory

| kind | name | formula | reads as |
|---|---|---|---|
| `phi` | `speed` | `std({‖v_t‖})` | erratic step sizes |
| `phi` | `curv` | `mean({θ_t})` | how sharply the path turns |
| `phi` | `ang` | `std({θ_t})` | how *inconsistently* it turns |
| `phi` | `accel` | `mean({‖a_t‖²})` | abrupt changes of motion |
| `phi` | `perr` | `mean({‖ε_t‖})` | how much the clip surprises a predictor of its own past |
| `phi` | `or` | `argmax_b │z_b│` over the five | ensemble: trust the most confident statistic |
| `phi` | `majority` | `Σ_b z_b` over the five | ensemble: vote across all five |
| `drift` | one per statistic | `ġ_σ = ⟨∇_z̄ φ_σ, ū_θ⟩` | is this denoising step *regularising* the trajectory (`<0`) or eroding it (`>0`)? |
| `coupling` | `alignment` | `cos((Δz̄)_f, (Δū)_f)` | is the step growing the motion already there, or rewriting it? |
| `coupling` | `erosion` | `‖Δū_f‖ / ‖Δz̄_f‖` | how fast motion is restructured relative to how much exists |

`v_t`, `θ_t`, `a_t`, `ε_t` are the per-frame intermediates defined in §3. The two
ensembles are GeoPhys's own; they combine the five `phi` statistics and so exist only for
`kind = phi`.

### Where the count comes from

For a Wan run with 30 blocks and 10 recorded steps — the exact composition of the
**3820** signals in `signals.csv`:

| source | kind | cells | × quantities | signals | why this many cells |
|---|---|---|---|---|---|
| `hidden_states` | `phi` | 30 × 10 = 300 | 7 | **2100** | every block, every step |
| `hidden_states` | `drift` | 30 × 9 = 270 | 5 | **1350** | 9, not 10: an *empirical* drift is a finite difference and needs the next step, so the last one has no successor |
| `latent` | `phi` | 1 × 10 = 10 | 7 | 70 | not per-block |
| `latent` | `drift` | 1 × 10 = 10 | 5 | 50 | 10, not 9: **exact** drift is an analytic gradient and needs no successor |
| `latent` | `coupling` | 1 × 10 = 10 | 2 | 20 | only the ODE state gets coupling metrics |
| `x0_hat` | `phi` | 10 | 7 | 70 | |
| `x0_hat` | `drift` | 9 | 5 | 45 | empirical |
| `velocity` | `phi` | 10 | 7 | 70 | |
| `velocity` | `drift` | 9 | 5 | 45 | empirical |
| | | | | **3820** | |

Two asymmetries in that table are the exact/empirical distinction made concrete, and both
are worth noticing:

- **`latent` has 10 drift cells where the others have 9.** The latent *is* the ODE state,
  so `ġ_σ` is a gradient evaluated at a point. Everywhere else it is a secant between two
  recorded steps, and the last step has nothing to pair with.
- **only `latent` has `coupling` rows at all.** Transport alignment and erosion rate are
  defined against the flow field acting on the state; for a non-state source there is no
  corresponding exact quantity, so they are not computed rather than approximated.

On CogVideoX-5B, substitute 42 blocks for 30: `phi` becomes 42 × 10 × 7 = 2940 and
`drift` 42 × 9 × 5 = 1890, for **5200** signals total.

### Reading a signal name

```
hidden_states/b22/s4/phi_accel
└─ source ──┘ └─┘ └┘ └─┘ └───┘
              │    │   │    └─ statistic
              │    │   └────── kind
              │    └────────── recorded step 4 of 10
              └─────────────── DiT block 22
```

`latent/s1/drift_curv` has no block segment, because that source is not per-block.

> **This is a lot of signals for a small number of pairs**, and reporting the best one is
> a selection procedure, not a result. See §6 for the permutation null that separates the
> two, and note it needs *different* floors for a mean and for a maximum.

## 4c. Exactly how each one is computed

Enough detail to reimplement without reading the code. Links go to the implementation,
which carries the *why* at each call site.

### Getting to a trajectory

Everything below operates on `Z ∈ R^{T×D}`, one vector per latent frame. Both paths
reduce to that shape before any statistic is touched:

- **internal** — the raw activation is `(1, T_tok·H_tok·W_tok, D)`. Both backends patchify
  with `patch_size=(1,2,2)` and flatten row-major, so each latent frame owns a contiguous
  equal-length run of tokens. Reshape to `(T, H_tok·W_tok, D)` and `mean(dim=1)`. Exact,
  no interpolation.
- **external** — DINOv2 gives one vector per *video* frame, `(F, D)`. Since the causal VAE
  folds 4 video frames into each latent after the first, the frames belonging to each
  latent are averaged, giving `(T, D)`. Without this the two paths differ in both
  trajectory length and spacing, and every statistic depends on both.

### The five statistics, step by step

**Per-frame intermediates first.** With `v_t = z̄_{t+1} − z̄_t` for `t = 1 … T−1`:

| | shape | computation |
|---|---|---|
| `v_t` [`displacements`](../phaselock/metrics/geophys.py#L72) | `(T−1, D)` | `Z[1:] − Z[:−1]` |
| `s_t` [`speeds`](../phaselock/metrics/geophys.py#L82) | `(T−1,)` | `‖v_t‖₂` |
| `a_t` [`accelerations`](../phaselock/metrics/geophys.py#L107) | `(T−2, D)` | `v[1:] − v[:−1]`, i.e. the second difference of `Z` |
| `θ_t` [`turning_angles`](../phaselock/metrics/geophys.py#L87) | `(T−2,)` | see below |
| `ε_t` [`prediction_residuals`](../phaselock/metrics/geophys.py#L125) | `(T−order,)` | see below |

**`θ_t`, the turning angle.** Normalise both displacements, then take the half-angle form:

```
â = v_t / ‖v_t‖     b̂ = v_{t+1} / ‖v_{t+1}‖
θ_t = 2 · atan2( ‖â − b̂‖ , ‖â + b̂‖ )
```

Algebraically identical to `arccos(⟨â, b̂⟩)`, numerically far better. `arccos` loses about
half its precision near 0 and π — exactly where a near-straight trajectory sits — and its
derivative diverges there, which matters because §4's drift differentiates this. Norms use
an epsilon floor so a stationary segment gives 0 rather than NaN.

**`ε_t`, the prediction residual** (default `fit="span"`). For each sliding window of
`order = 3` consecutive frames predicting the next one:

1. **anchor** on the most recent frame of the window, `p = z̄_t`
2. **basis** = the earlier frames as offsets from it, `B = [z̄_{t−2} − p, z̄_{t−1} − p]`,
   shaped `(D, order−1)`. Anchoring makes the span *affine*, so a constant-velocity
   trajectory is predicted exactly and scores zero.
3. **target offset** `o = z̄_{t+1} − p`
4. **project** `o` onto `span(B)` by least squares: `c = pinv(B) · o`, giving `ẑ = B·c`
5. **residual** `ε_t = ‖o − ẑ‖₂` — the component orthogonal to the span

`pinv` rather than `lstsq`: it is differentiable and tolerant of rank-deficient windows,
which a stationary segment produces. `ridge` and `scalar` are the alternative fits; see §3
for why the paper's literal global fit is degenerate.

**The five summaries.** `φ_speed = std({s_t})`, `φ_curv = mean({θ_t})`,
`φ_ang = std({θ_t})`, `φ_accel = mean({‖a_t‖²})`, `φ_perr = mean({‖ε_t‖})` —
[`geophys_statistics`](../phaselock/metrics/geophys.py#L250). Standard deviations are
**population**, not sample. Note again `accel` is squared and `perr` is not; that
asymmetry is the paper's.

### The three coupled metrics, step by step

All take the pooled trajectory `Z` and the pooled flow `U = pool(u_θ)` at the same point,
the same shape. Pooling is linear, so the pooled drift *is* the drift of the pooled
trajectory — that is the whole reason this works.

> **Which way time runs.** The plots put `t = 0` (clean video) on the left and
> `t = 1000` (pure noise) on the right, because that is the order inversion visits them
> in. The drift below runs the other way: it is `dz/dτ` oriented **noise → data**, the
> direction the model actually *generates*, which is right to left on those axes. So
> `ġ_σ < 0` at `t = 700` means "as the model refines through this point toward a clean
> video, it is making the trajectory more regular". Reading it left to right inverts the
> claim, and the claim is the whole point — PhaseLock's thesis is about what *refinement*
> does to the motion prior.

**Geometric drift, exact** — [`geometric_drift`](../phaselock/metrics/flow_geometry.py#L61)

1. detach `Z`, clone it, set `requires_grad_(True)`
2. compute all five statistics on it, building an autograd graph over a `(T, D)` tensor
3. for each statistic: `g = autograd.grad(φ_σ, Z, retain_graph=True)`
4. `ġ_σ = Σ (g ⊙ U)` — the directional derivative of the statistic along the flow

The graph covers the statistic only, never the transformer, so this is a few thousand
flops on top of a sampler step already computed. Sign convention: **negative means this
step is making the trajectory more regular**.

**Geometric drift, empirical** — [`geometric_drift_empirical`](../phaselock/metrics/flow_geometry.py#L105)

`ġ_σ ≈ [φ_σ(Z_{k+1}) − φ_σ(Z_k)] / (τ_{k+1} − τ_k)` across consecutive *recorded* steps.
Same quantity, but a secant whose resolution is the recording stride rather than a tangent.
This is what every non-ODE-state source uses.

**Transport alignment** — [`transport_alignment`](../phaselock/metrics/flow_geometry.py#L127)

Per frame, the cosine between the motion and how the flow is changing that motion:
`ρ_f = cos((ΔZ)_f, (ΔU)_f)`. Returned twice — a plain mean, and a mean weighted by
`‖(ΔZ)_f‖` so that near-static frames, where the cosine is dominated by noise, do not
count equally with moving ones.

**Erosion rate** — [`erosion_rate`](../phaselock/metrics/flow_geometry.py#L154)

`mean(‖(ΔU)_f‖ / (‖(ΔZ)_f‖ + ε))` — how fast motion is being restructured relative to how
much motion exists. Unlike alignment it is unsigned and unbounded.

### From a number to an accuracy

[`statistics_from_record`](../phaselock/experiments/detection.py#L74) turns a probe record
into the rows of `statistics.csv`, and is where the exact-versus-empirical choice is made
per source. Then, per signal, over matched pairs:

1. `δ_b = u_b(V⁻) − u_b(V⁺)` — [`signed_deltas`](../phaselock/metrics/scoring.py#L41).
   Positive when the signal ordered the pair correctly.
2. `z_b = δ_b / scale` — [`scale_normalize`](../phaselock/metrics/scoring.py#L71).
   **Scale only, never mean-centred**: subtracting the mean would flip the sign of every
   below-average pair, and the sign is the prediction. See §5.
3. accuracy = fraction of positive `δ`, ties 0.5 —
   [`pairwise_accuracy`](../phaselock/metrics/scoring.py#L96)
4. interval = 1000-resample bootstrap **grouped by scenario** —
   [`bootstrap_ci`](../phaselock/metrics/scoring.py#L133)
5. ensembles over the five: [`majority_ensemble`](../phaselock/metrics/scoring.py#L300)
   (`Σ z_b`) and [`or_ensemble`](../phaselock/metrics/scoring.py#L311) (`argmax |z_b|`)
6. and, once for the whole run,
   [`selection_null`](../phaselock/metrics/scoring.py#L246) — shuffle which member of each
   pair is labelled violated, rescore every signal, repeat 300×. Yields two floors, one
   for a mean and a higher one for a maximum, because a maximum selects for noise while a
   mean cancels it.

### How to check the implementations are right

The tests are executable specifications, and are more precise than prose. They run on CPU
in seconds with no weights:

```bash
python -m pytest tests/test_geophys.py tests/test_flow_geometry.py -v
```

The statistics are pinned against analytic ground truth rather than golden values — a
straight line gives zero for all five, a discretised circle gives constant curvature with
zero speed variation, an injected teleport spikes acceleration and residual at exactly the
frame it was injected at. Two tests exist specifically to document traps: that the naive
global OLS residual is degenerate, and that the turning angle stays stable at both
endpoints of its range where `arccos` does not.

The new metrics are checked against the identity they are derived from: frame differencing
commutes with a linear flow, `geometric_drift` matches a central finite difference of the
statistic, the empirical estimator converges to the exact one as the step shrinks, and a
flow that provably straightens a trajectory must give `ġ_curv < 0`.

## 5. Scoring

GeoPhys's rule, verbatim. For a matched pair `(V⁺, V⁻)` with `V⁻` violated:

```
δ_b = u_b(V⁻) − u_b(V⁺)
```

positive when the signal orders the pair correctly, since every statistic is larger for
less regular motion. Ties count 0.5.

**One ambiguity resolved.** The paper says δ is "z-normalised to `z_b`, whose sign
identifies the violated video". Those clauses conflict: mean-centring flips the sign of
every pair below the dataset mean, so a signal that ranks *all* pairs correctly would be
scored wrong on the below-average ones. Only scale-only normalisation keeps the sign
meaningful while still putting signals on a comparable footing. `scale_normalize` is what
the ensembles consume; `zscore` is kept for cross-pair work.

Ensembles: **Majority** (`Σ_b z_b`) and **OR** (`argmax_b |z_b|`).

AUC is reported alongside pairwise accuracy because they test different things. Pairwise
accuracy tests within-pair ranking only; AUC tests whether one global threshold separates
violated from plausible across the dataset, which is the stronger claim and the one that
matters if the score is ever used as a verifier.

Confidence intervals: 1000-resample bootstrap, **grouped by scenario** for LikePhys and by
pair for IntPhys2. LikePhys shares one valid clip across a subgroup's violations, so those
pairs are not independent; resampling videos rather than scenarios would understate the
interval.

## 6. Reading the results

**Check the correctness gate first.** `scripts/run_external.py` runs the five statistics
on frozen DINOv2 features — the representation GeoPhys designed them for. It must land
near the published 77.6–80.8% single-backbone range on LikePhys. If it does not, the
statistics, the pairing, the preprocessing or the scoring rule is wrong, and no internal
number means anything.

Then the cells that matter:

- **`latent` vs `hidden_states`.** The Invisible Hand reports linear probes at chance on
  VAE latents. Geometry also landing at chance there while working on hidden states is a
  clean positive result, and says PhaseLock guides in a physics-free representation.
- **Block × step structure.** The Invisible Hand found probe accuracy peaking in the
  middle third of the network. Whether the *geometric* readout peaks in the same place is
  a question about how physical information is organised, not just about a readout.
- **`ġ_σ` vs `φ_σ`.** Does the rate of change separate pairs better than the value? A
  violated clip should be one the model's own flow is actively fighting.
- **The step sweep's blur control.** A 2-step output is blurrier, so any feature
  trajectory could look "more regular" from having less texture to move. Blur at
  σ ∈ {0, 8, 16} is applied to *every* arm including the real reference, per PhaseLock
  Fig. 3a. If the K=2 advantage disappears under blur, the ordering is a sharpness
  artefact and is not a physics result — a negative worth reporting, since PhaseLock's
  *spectral* metric does survive the same control.

## 7. Known limits

- Local LikePhys has **800 pairs** where the paper cites 650, so this is a different
  release and absolute numbers will not match published ones.
- The 14B Wan checkpoints are wired but **unvalidated** — they do not fit a 46 GB card.
- `attention` records the attention sublayer *output*, not attention matrices. At native
  resolution a single block's video self-attention is 32760², about 10⁹ entries, and both
  backends route through fused SDPA which never materialises it.
- Everything here is correlational. Geometry separating violated from plausible says the
  representation carries a usable statistical signature, not that the model represents
  physics.
