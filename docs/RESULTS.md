# Results

> **Scope.** What was measured and what it means. For how each signal is defined and
> computed, [METHOD.md](METHOD.md); for how to run any of it, [RUNNING.md](RUNNING.md).
>
> Updated 2026-08-10 against the n=100 native-resolution run. Earlier runs are under
> `_archive/`.

## In four sentences

**A video diffusion model's own internals separate physically implausible video from
plausible video better than a frozen image encoder does — but most of that separation is
one effect, not eight.** A single scalar, how far the pooled trajectory moves per frame,
reaches 73.8% on DiT hidden states; of GeoPhys's five statistics only `φ_perr` measures
anything beyond it (58.2% after the scale is divided out, against a 52.1% floor), and
three sit at chance. The headline number is `φ_perr` at **75.3%** against DINOv2's best of
**65.3%**, temporally matched on the same 100 pairs. On IntPhys2 nothing clears 57%, so
all of this is a LikePhys result rather than a general one.

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
more spread — so they are the more selection-sensitive. **§4b shows both are mostly the
scale effect**; `φ_perr` is the only one that survives dividing it out.

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

### 3. `φ_jerk` is the strongest addition, but see §4b before believing it

68.8 / 67.2 / 72.2 / 69.0 across the four internal sources — ahead of every GeoPhys shape
statistic, and strong on *all four* sources rather than one. The scale control in §4b says
almost all of that is the magnitude effect rather than discontinuity, so it is not
independent evidence.

`φ_momentum` is a weak positive (55–60%). The three **coupling** metrics — geometric drift,
transport alignment, erosion rate — are a measured negative at 54–59%, below the plain
statistics they were built from. The commutation identity is still correct; as detectors the
coupled quantities carry less than the uncoupled ones.

### 4. Most of the signal is one thing: violated clips travel further in feature space

This is the finding that reframes the rest, and it came from asking why `φ_energy` reads
*below* chance. Reproduce with `scripts/scale_control.py <run_dir>`.

Decompose the trajectory into how far it moves and what shape it makes. On hidden states,
over the probe grid:

| quantity | accuracy | what it is |
|---|---|---|
| `mean(‖v_t‖)` | **73.8 ± 6.2** | the scale — mean step length, equivalently path length |
| `std(‖v_t‖)` = `φ_speed` | 49.2 ± 6.4 | the spread, with the scale removed |
| `std(‖v_t‖)/mean(‖v_t‖)` | **28.6 ± 6.7** | the same spread, divided by the scale |
| `φ_energy` = `std(E)/mean(E)` | **28.1 ± 6.1** | the same ratio on `E = ½‖v‖²` |

**A single scalar — how far the pooled trajectory moves per frame — is a 73.8% detector**,
and it is not one of the eight. The *spread* around it carries nothing (49.2%, chance).

That settles `φ_energy`. It is a coefficient of variation: a numerator at chance over a
denominator that is a strong detector. Dividing by a discriminative quantity inverts the
ratio, mechanically. **No energy-exchange story is needed, and the one this document gave
earlier was wrong** — `φ_energy` and `std(‖v‖)/mean(‖v‖)` agree to within 0.5 points, and
the latter contains no notion of energy at all.

It also explains `φ_speed`, which GeoPhys defines as `std({‖v_t‖})`. **That definition
discards the mean, which is where all the signal is** — hence chance, on every source.

### 4b. What survives the scale control

Dividing each statistic by the power of `mean(‖v_t‖)` it carries by definition:

| statistic | raw | scale-normalised | verdict |
|---|---|---|---|
| `φ_perr` | 75.4 | **58.2** | keeps a real margin over the 52.1% floor |
| `φ_accel` | 70.5 | 44.4 | scale, and the correction overshoots |
| `φ_jerk` | 68.7 | 51.6 | scale |
| `φ_momentum` | 59.2 | 59.2 | already scale-free |
| `φ_curv`, `φ_ang` | ~49.7 | unchanged | already scale-free, and at chance |

**`φ_perr` is the only statistic that measures something beyond how far the trajectory
moved** — and even it loses 17 points to the control. `φ_accel` and `φ_jerk` are the scale
effect re-measured; their apparent strength in the headline table is not independent
evidence.

This does not make the detector worse — 73.8% from one scalar is a real result, and
`φ_perr` still adds to it. It makes the *interpretation* narrower: the dominant effect is
that these clips move differently in magnitude, not that the model's geometry encodes
plausibility in the way GeoPhys's five statistics were designed to capture.

**Whether the scale effect is itself a confound is untested.** Violated clips could move
further in feature space because a violation is an abrupt event, or because the edit that
created them changed the clips in some visually trivial way. Separating those needs a
control this run does not have.

### 4b bis. Energy continuity was tested properly, and does not work

The natural repair for `φ_energy` is to stop dividing by the mean and ask instead whether
the energy is *continuous* — real motion changes its energy through discrete events (an
impact, a bounce), and editing physics away should smooth them out. Eighteen candidates
were scored on LikePhys and rescored on IntPhys2 without refitting
(`scripts/candidate_statistics.py`), covering spreads, crest factors, autocorrelation,
total variation, local relative jumps, second differences and burst counts.

**Nothing survives.** Every candidate that is far from chance on LikePhys lands at chance
on IntPhys2 or flips sign, including `mean(‖v_t‖)` itself (73.8 → 46.4). The best
order-sensitive candidate, `1 − autocorr(E)`, is 52.5 and 50.2.

One candidate looked like the exception and was not. A **burst fraction** — the share of
frames whose `|ΔE|` exceeds twice the clip's median — read 39.5% on LikePhys and 38.9% on
IntPhys2: far from chance, same sign, same size, on two unrelated datasets. It was an
artefact of the scoring rule. A fraction over twelve frames takes about six distinct
values, so it **ties on 22% of LikePhys pairs and 30% of IntPhys2 pairs**, and `violated >
plausible` counts a tie as a miss. Excluding ties it is 50.6% and 55.8% — nothing.

Two things follow. The tool now reports the tie rate and the tie-excluded accuracy in
every row, because that artefact is invisible in the headline number and points the wrong
way by construction. And the production statistics are unaffected: all eight are
continuous functions of the trajectory and tie at a rate of 0%.

So `φ_energy` is left as it is — a documented control that shows what a scale ratio looks
like — rather than replaced by something that only appears to work.

### 4c. Where along the denoising trajectory the signal lives

Accuracy on hidden states, meaned over all 30 blocks, by recorded step. `t = 0` is the
clean video, `t = 993` is nearly pure noise:

| t | 0 | 251 | 459 | 586 | **702** | 779 | 853 | 904 | 956 | 993 |
|---|---|---|---|---|---|---|---|---|---|---|
| `φ_perr` | 72.7 | 75.0 | 75.9 | 76.1 | 72.9 | 73.9 | 74.9 | 76.6 | 77.1 | **78.0** |
| `φ_accel` | 69.9 | 65.0 | 64.9 | 70.2 | **81.2** | 77.4 | 74.4 | 71.3 | 67.0 | 65.7 |
| `φ_jerk` | 67.7 | 64.2 | 64.2 | 69.2 | **80.4** | 75.6 | 72.5 | 69.0 | 63.7 | 61.8 |

**Not at the clean end.** `φ_accel` and `φ_jerk` peak in the *middle* of the trajectory
(81.2% and 80.4% at `t ≈ 702`), and `φ_perr` is at its weakest there and strongest near
noise. The derivative statistics want a partly-noised latent; the residual wants either
end.

`05_signal_profile_time` appears to say otherwise — its plausible/violated gap looks widest
near `t = 0` — and it does not. Two reasons, both of which make that figure unusable for
judging effect size:

- **It plots the raw statistic, which is largest near the clean end.** Every one of these
  grows with the trajectory's scale (§4), and the trajectory moves furthest while the
  latent still holds real content. Absolute gap and absolute value shrink together toward
  noise; the *relative* gap does not follow.
- **The line and band are unpaired; the accuracy is paired.** Bands that overlap almost
  entirely still give 75%, because a pair's two clips share a scenario and cancel most of
  that spread. The figure shows between-clip variance, which pairing removes.

Read `02_depth_profile` and `03_depth_vs_time` for separation by depth and step. The
caption on `05_signal_profile_time` now says this.

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

5. **The dominant effect may be a confound.** §4 shows 73.8% comes from how far the
   trajectory moves, and nothing here establishes *why* violated clips move further. An
   abrupt physical event would do it; so would a systematic visual difference introduced by
   whatever edit produced the violated clip. LikePhys renders its violated clips rather
   than editing footage, which makes the second less likely but does not rule it out. The
   test is a matched control on clips that differ in motion magnitude without differing in
   plausibility.

6. **Orientation is assumed, not fitted.** Every signal is scored as "larger = less
   plausible". That is justified for the five shape statistics; `φ_energy` shows what happens
   when it is not, and the drift and alignment values have no principled orientation at all.
   A below-50% number is discrimination with an inverted sign, not absence of signal.

7. **100 pairs.** Individual cells carry ±8–10 point confidence intervals. Trust the
   *structure* — a smooth peak across neighbouring blocks and steps — over any single
   winning cell, which is what the depth heatmaps are for.

8. **Everything is correlational.** Geometry separating violated from plausible says the
   representation carries a usable statistical signature, not that the model represents
   physics.

## Where this leaves the research question

**Does GeoPhys's geometric readout transfer to a video diffusion model's internals?**
Weakly, and less than the headline table suggests. After the scale control only `φ_perr`
measures anything beyond trajectory magnitude, at 58.2% against a 52.1% floor. The
representation does separate the classes well — 75.3% — but mostly through a quantity none
of GeoPhys's five statistics is designed to capture, and `φ_speed` explicitly discards.

**Which representation carries it best?** DiT hidden states for `φ_perr`, but the **flow
velocity** `u_θ` wins on `φ_accel` and `φ_jerk` — so the answer depends on the statistic.
The margin over the VAE latent is smaller than the Invisible Hand's linear-probe results
would predict (75.3 vs 72.6).

**Does internal beat external?** Yes on the strongest channel of each, temporally matched on
the same pairs: **75.3 ± 3.2** against **65.3 ± 6.7**. On IntPhys2 the ordering survives but
the margin nearly vanishes (57.1 vs 48.5).

**Open, in the order worth doing:** control the magnitude effect (threat 5) — it is now
the load-bearing result and the least understood; sweep `inversion__num_steps` (threat 1);
finish the CogVideoX and generation tracks; test the internal + external combination that
§2 points at.

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
