# Results

> **Scope.** What was measured and what it means. For how each signal is defined and
> computed, [METHOD.md](METHOD.md); for how to run any of it, [RUNNING.md](RUNNING.md).
>
> Updated 2026-08-10 against the n=100 native-resolution run. Earlier runs are under
> `_archive/`.

## In three sentences

**A video diffusion model's own internals carry physical plausibility better than a frozen
image encoder does — but only through two of the eight geometric channels, and only on
LikePhys.** `φ_perr` on DiT hidden states reaches 75.3% pairwise accuracy against DINOv2's
best of 65.3%, while three of GeoPhys's five statistics sit at chance internally and score
61–65% externally. On IntPhys2 nothing clears 57%, so the headline is a LikePhys result
rather than a general one.

## Which model produced which number

**Every number here is model-specific.** An accuracy measured by inverting with Wan says
nothing about CogVideoX, and the two are never mixed in a table.

| track | backend | dataset | status |
|---|---|---|---|
| **primary** | Wan2.1-T2V-1.3B, inversion | LikePhys, 100 pairs @ 512² | **complete** |
| baseline | DINOv2-large, frozen, no diffusion | LikePhys, 100 pairs @ 512² | **complete** |
| held-out | Wan2.1-T2V-1.3B, inversion | IntPhys2, 100 pairs | **complete** |
| second model | CogVideoX-5B-I2V, inversion | LikePhys, 100 pairs @ 480×720 | running |
| generation | CogVideoX-5B-I2V | LikePhys | running |
| step sweep | CogVideoX-5B-I2V | LikePhys | running |

The `φ_energy`, `φ_momentum` and `φ_jerk` columns for the primary track were rebuilt from
its saved trajectories with `scripts/rescore_trajectories.py` — those statistics did not
exist when it started — and live in `*_rescored.csv` beside the originals.

**CogVideoX runs at 480×720, Wan at 512×512.** Forced: diffusers refuses any other geometry
for that checkpoint. The two are therefore not resolution-matched.

## The headline table

Pairwise accuracy, mean ± s.d. over each source's probe grid, `n = 100`. The figure is
`figures/00_overall.png`; the same numbers are in `overall.xlsx`.

**Floors, from a 300-round label permutation over the same 2640 signals: a mean must clear
52.1%, a single best cell 68.0%.** Comparing a maximum against the mean floor is the
easiest way to misread everything below.

### LikePhys — Wan2.1-1.3B

| source | φ_perr | φ_accel | φ_jerk | φ_mom | φ_curv | φ_ang | φ_speed | φ_energy |
|---|---|---|---|---|---|---|---|---|
| **DINOv2** *(external, 25 cells)* | 54.9 | 63.9 | 63.8 | 53.2 | 61.8 | 63.4 | **65.3** | 63.8 |
| **DiT hidden states** *(300)* | **75.3** | 70.7 | 68.8 | **60.0** | 49.6 | 49.8 | 49.4 | 28.5 |
| **VAE latent `x_t`** *(10)* | 72.6 | 67.3 | 67.2 | 55.4 | 54.0 | 51.0 | 46.6 | 31.5 |
| **clean estimate `x̂₀`** *(10)* | 64.8 | 66.0 | 69.0 | 58.5 | **63.1** | **53.2** | **50.9** | 29.7 |
| **flow velocity `u_θ`** *(10)* | 68.9 | **73.1** | **72.2** | 59.8 | 53.1 | 52.0 | 47.4 | 31.1 |

Best single cells, all on hidden states: `φ_accel` and `φ_jerk` at 87.0%, `φ_perr` at 82.0%.

### IntPhys2 — same backend, same statistics

| source | φ_perr | φ_accel | φ_jerk | φ_mom | φ_curv | φ_ang | φ_speed | φ_energy |
|---|---|---|---|---|---|---|---|---|
| **DINOv2** | 50.8 | 48.5 | 47.6 | 46.1 | 46.0 | 48.1 | 47.1 | 45.6 |
| **DiT hidden states** | 51.3 | 46.0 | 45.0 | 44.1 | 42.8 | 44.0 | 54.4 | 55.0 |
| **VAE latent** | 55.8 | 47.5 | 48.0 | 42.6 | 43.1 | 42.9 | 53.9 | 55.5 |
| **clean estimate** | 52.0 | 46.2 | 47.6 | 39.1 | 41.4 | 49.6 | 55.9 | 56.7 |
| **flow velocity** | 49.0 | 47.2 | 47.1 | 40.6 | 43.7 | 46.1 | 56.9 | **57.1** |

---

## What the numbers say

### 1. The signal is real, and it is the prediction residual

`φ_perr` on hidden states averages **75.3% ± 3.2** across all 300 probe cells. The tight
spread is the important part: not one lucky cell but a property of nearly every block and
denoising step. A best-of-N artefact does not look like that.

`φ_accel` (70.7) and `φ_jerk` (68.8) follow, and both reach higher single cells (87.0) with
more spread — so they are the more selection-sensitive.

### 2. The internal and external profiles are shaped differently, and nearly inverted

DINOv2 is **flat**: every statistic lands in 53–65%, nothing excels, nothing fails. The DiT
is **spiky**: two channels well above anything DINOv2 achieves, three at chance.

Each representation's best statistic is close to the other's worst — `φ_perr` is 75.3
internal against 54.9 external; `φ_speed` is 65.3 external against 49.4 internal.

A plausible reading, not tested here: DINOv2 features encode appearance, so a trajectory
through them moves when the *image* changes, and irregular image change reads as irregular
motion — which `φ_speed` measures. Diffusion internals are trained to predict the next
state, so what breaks under implausible physics is *predictability* — which is `φ_perr`.

That points at **combining** them rather than choosing, which nothing here has tested.

### 3. `φ_jerk` is the one addition that paid off

68.8 / 67.2 / 72.2 / 69.0 across the four internal sources — ahead of every GeoPhys shape
statistic, and the only signal that is strong on *all four* sources rather than one. That
is what you would expect if discontinuity is what violations actually introduce.

`φ_momentum` is a weak positive (55–60%). The three **coupling** metrics — geometric drift,
transport alignment, erosion rate — are a measured negative at 54–59%, below the plain
statistics they were built from. The commutation identity is still correct; as detectors the
coupled quantities carry less than the uncoupled ones.

### 4. `φ_energy` is inverted on LikePhys — and IntPhys2 says do not flip it

28.5 / 31.5 / 31.1 / 29.7 across four readouts that share no parameters: roughly 20 points
*below* chance, in the same direction. A statistic with no signal sits at 50 with scatter;
this does not. It carries as much as a good detector and points the wrong way.

The reading: **plausible motion varies its kinetic energy more than violated motion does.**
Real dynamics trade energy continuously — a ball decelerating into a bounce, cloth settling.
LikePhys injects violations by editing motion toward something uniform, which *flattens* the
energy profile.

**The obvious fix is wrong.** Flipping the sign would give ≈70% on LikePhys, third-best
overall. On IntPhys2 `φ_energy` is 55–57%, *above* chance in the same four readouts — so
flipping would have turned a 55% signal into a 45% one on held-out data. The inversion is a
property of how LikePhys builds violated clips, not of the statistic. Reported unflipped.

DINOv2 scores 63.8% on the same statistic over the same clips, so the sign is also a
property of the readout, not of the videos.

### 5. The VAE latent is not physics-free

The Invisible Hand reports linear probes at 48–53% on VAE latents — chance — and that result
is load-bearing for their argument. Geometry on the same latents reaches **72.6%** with
`φ_perr`.

Not a contradiction, a difference in readout: a linear probe asks whether plausibility is
linearly separable; `φ_perr` is a nonlinear functional of the whole trajectory. The signal is
present but not linearly available.

**This matters for PhaseLock**, whose latent delta `T(z)` *is* the GeoPhys first-order
velocity in exactly this space. That space is not empty — but the first-order term is the
worst place to look (`φ_speed`, 46.6%), while the residual, acceleration and jerk terms
PhaseLock never touches carry 72.6%, 67.3% and 67.2%.

### 6. Where it works is not uniform

`figures/06_by_family.png` and `06_by_scenario.png`. `φ_perr` on hidden states ranges from
**91% on soft body to 57% on fluid**; `φ_curv` and `φ_ang` work on fluid (75–79%) and fail on
rigid and soft (22–38%). Two methods with the same mean can differ completely here.

### 7. It does not transfer to IntPhys2

Nothing clears 57%, against 75.3% on LikePhys. Consistent with the published difficulty —
V-JEPA 2 at 57.5%, GeoPhys at 59.5%, human 96.4% — but it means the headline is a LikePhys
result. What weak signal survives is carried by the diffusion model (54–57% on
`φ_speed`/`φ_energy`) and not by DINOv2 (45–47%).

---

## What not to believe yet

Worst first.

1. **Inversion fidelity is unverified.** `inversion__num_steps` has never been swept. The
   Invisible Hand reports probe accuracy collapsing 0.82 → 0.57 between 100 and 20 steps
   while reconstruction still looks fine. 41.3 dB against a 49.7 dB VAE ceiling rules out a
   *broken* inversion, not an unfaithful trajectory. **This is the largest open threat to
   everything above** — it is gate G5 in [RUNNING.md](RUNNING.md).

2. **Inverted video is not generated video.** GeoPhys evaluates real and generated clips
   through a feature extractor. This evaluates *inverted* real clips, which only means
   something if the recovered trajectory is the one the model would have taken.

3. **The two paths select from very different pools.** The comparison of *means* is sound.
   The comparison of *maxima* is not: the internal best cell is chosen from 2640 candidates
   and DINOv2's from 200. Quote the means.

4. **One backend for the headline.** CogVideoX is still running, and at a different
   resolution when it lands.

5. **Orientation is assumed, not fitted.** Every signal is scored as "larger = less
   plausible". That is justified for the five shape statistics; `φ_energy` shows what happens
   when it is not, and the drift and alignment values have no principled orientation at all.
   A below-50% number is discrimination with an inverted sign, not absence of signal.

6. **100 pairs.** Individual cells carry ±8–10 point confidence intervals. Trust the
   *structure* — a smooth peak across neighbouring blocks and steps — over any single
   winning cell, which is what the depth heatmaps are for.

7. **Everything is correlational.** Geometry separating violated from plausible says the
   representation carries a usable statistical signature, not that the model represents
   physics.

## Where this leaves the research question

**Does GeoPhys's geometric readout transfer to a video diffusion model's internals?**
Partly. Two of the five transfer, plus `φ_jerk` of the three additions. The transfer is not
a property of "the GeoPhys method" but of specific statistics.

**Which representation carries it best?** DiT hidden states for `φ_perr`, but the **flow
velocity** `u_θ` wins on `φ_accel` and `φ_jerk` — so the answer depends on the statistic.
The margin over the VAE latent is smaller than the Invisible Hand's linear-probe results
would predict (75.3 vs 72.6).

**Does internal beat external?** Yes on the strongest channel of each, temporally matched on
the same pairs: **75.3 ± 3.2** against **65.3 ± 6.7**. On IntPhys2 the ordering survives but
the margin nearly vanishes (57.1 vs 48.5).

**Open, in the order worth doing:** sweep `inversion__num_steps` (threat 1); finish the
CogVideoX and generation tracks; test the internal + external combination that §2 points at.

## Reproducing

```bash
# every table and figure in this document, from a finished run
python scripts/report.py <run_dir> --figures

# the primary track's numbers, which needed rescoring for the three new statistics
python scripts/report.py <run_dir> --figures \
    --statistics statistics_rescored.csv --signals signals_rescored.csv
```

Background on the two terms that carry weight above: **dB is PSNR**, logarithmic, ~6 dB per
halving of error; the **VAE ceiling** is encode-then-decode with no inversion at all (47–51
dB here), so judge inversion by the *gap* to it rather than the absolute number — a backend
with a worse VAE shows a lower number without inverting any worse. Wan's gap is 7 dB,
CogVideoX-I2V's 17 dB, and CogVideoX-T2V's 40 dB is why detection runs on Wan.
