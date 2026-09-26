# Running the experiments

> **Scope.** How to run everything and what to check when it finishes. For what is being
> measured and why, see [METHOD.md](METHOD.md); for what the runs found,
> [RESULTS.md](RESULTS.md).

## One command

```bash
tmux new -s phaselock
cd /home/ec2-user/code/phaselock
scripts/experiments/run_all.sh
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t phaselock`.

**The run logs itself** to `<root>/<run_id>/run.log` while still printing to the terminal.
Piping through `tee` by hand is easy to forget, and the one run you forget it on is the one
you need the log for. Progress bars go to stderr; set `PHASELOCK_NO_PROGRESS=1` to drop
them.

## Knobs

Everything is an environment variable, so nothing needs editing:

| variable | default | meaning |
|---|---|---|
| `N` | `100` | matched pairs (or clips, for generation) |
| `SWEEP_N` | `30` | clips for the step sweep |
| `VISUALS` | `4` | pairs to render videos for |
| `RESOLUTION` | `native` | `native` (512×512) or `letterbox` (480×832) |
| `STAGES` | `A B C D E` | which tracks to run |
| `RUN_ID` | timestamped | the output folder name |
| `ROOT` | `/data/experiments/phaselock_geophys` | where everything lands |

```bash
N=5 scripts/experiments/run_all.sh                         # fast end-to-end check, ~1 h
STAGES="A" scripts/experiments/run_all.sh                  # just the primary track
STAGES="C D" RUN_ID=20260810_1107_n100_native scripts/experiments/run_all.sh   # fill in two tracks
```

**`N=2` is the floor.** Below two pairs a "pairwise accuracy" is 0% or 100% by
construction, so scoring drops every signal and no figure gets produced. Generation needs
**3 clips** for the same reason: a rank correlation over two points is meaningless.
**`N=12` is the floor for a per-scenario table** — below that some scenario has one pair
and gets dropped.

## The stages

| | track | model | dataset | what it answers |
|---|---|---|---|---|
| **A** | inversion | Wan2.1-1.3B | LikePhys | **the primary result** — which representation carries the geometry |
| **B** | inversion | CogVideoX-5B-I2V | LikePhys | does it hold on a second model? |
| **C** | generation | CogVideoX-5B-I2V | LikePhys | do the signals predict a faithful generation? |
| **D** | step sweep | CogVideoX-5B-I2V | LikePhys | does the few-step effect survive the blur control? |
| **E** | inversion | Wan2.1-1.3B | IntPhys2 | does it transfer to a second dataset? |

Each track runs gate → work → report → overlays. Stages are **guarded, not `set -e`**: one
failure does not cancel the rest, because the point of a long unattended run is to have
something for every track when you look.

**CogVideoX-5B-I2V ignores `RESOLUTION`.** diffusers refuses any geometry but its native
480×720 for that checkpoint, so B, C and D run there whatever is asked. Wan honours it.
This means A and B are not resolution-matched; say so when comparing them.

## Cost

Measured on one L40S at 480×832; native 512×512 is about 40% faster.

| per clip | |
|---|---|
| Wan2.1-1.3B inversion, 50 steps | 108 s |
| CogVideoX-5B, same | ~190 s |
| CogVideoX generation, 50 steps | ~330 s |
| DINOv2 gate | ~1 s |

At `N=100` that is roughly **35 h at letterbox, 21 h at native**. LikePhys's 100 pairs are
only ~180 distinct clips, because a subgroup's violations share one valid clip.

Two defaults exist because the extra clips buy nothing:

- **`SWEEP_N=30`.** The step sweep is a `(K, σ)` grid and clips are averaged *inside* each
  cell, so 30 already gives tight error bars. At 100 it would be the longest stage at 17 h.
- **`VISUALS=4`.** Each visual pair costs an extra full inversion plus six VAE decodes,
  about 4 min. At 100 that is 6.7 h of pure video rendering per track.

## Output

```
$ROOT/<run_id>/<backend>/<dataset>/<stage>/
```

`run_id` is computed **once per invocation** and passed to every stage, so a run's five
tracks sit together and a later run cannot overwrite an earlier one. It is deliberately not
a per-process default: each stage would then get its own timestamp and the report would not
find its own gate.

The **stage name says where the latent trajectory came from**, which is the thing that
actually differs between them — not the question being asked, since GeoPhys statistics and
pairwise scoring happen in all three.

| stage | trajectory source | labels |
|---|---|---|
| `inversion` | a **real** video, sampler run backwards to recover the trajectory it never had | ground truth, from the dataset |
| `generation` | the model's **own** forward sampling from a first frame | the real continuation is the reference |
| `step_sweep` | generation repeated at each `K`, under the blur control | as above, per `(K, σ)` cell |

```
20260810_1107_n100_native/
├── run.log
├── wan21_t2v_1_3b/likephys/inversion/
│   ├── config.json  statistics.csv  signals.csv  reconstruction.json
│   ├── overall.xlsx  category_family.xlsx  category_scenario.xlsx
│   ├── external_latent/   the temporally matched DINOv2 gate
│   ├── figures/           00_overall.png, 01_source_comparison.png, …, <source>/…
│   ├── trajectories/      full (T, D) arrays, needed for the timeline figures
│   └── visuals/           *_pair.mp4  *_inversion.mp4  *_roundtrip.mp4  *_timeline.png
├── cogvideox_5b_i2v/likephys/{inversion,generation,step_sweep}/
└── wan21_t2v_1_3b/intphys2/inversion/
```

Two runs sharing a full path **append** to `statistics.csv` — which is what makes resuming
work, but means a config change under the same name silently mixes settings. Change
`output.name` when you change anything else.

### What each video shows

The suffix says what the video shows, and suffixes do not overlap between stages.

| file | stage | what it shows | the question it answers |
|---|---|---|---|
| `*_pair.mp4` | inversion | plausible \| violated | Did the violation survive preprocessing? If the two look identical, no statistic downstream can recover it. |
| `*_pair_signals.mp4` | inversion | the pair, with the per-frame statistics running underneath | Where in the clip does each signal separate them? |
| `*_inversion.mp4` | inversion | original, then one panel per recorded step | How fast does the clean estimate stop tracking the real motion as the trajectory walks toward noise? |
| `*_roundtrip.mp4` | inversion | original \| VAE-only \| inverted | Did inversion return to the video it came from? The middle panel is the ceiling. |
| `*_generation.mp4` | generation | real continuation \| generated | Both start from the same first frame, so column 0 should match; later divergence is the model's own dynamics. |

## Resuming and re-reporting

Extraction is resumable: clips already in `statistics.csv` are skipped. To continue an
interrupted run, pass its `RUN_ID` and re-issue the same command.

To redo the analysis without recomputing anything — both CPU-only, seconds:

```bash
python scripts/report.py <run_dir> --figures        # tables and figures
python scripts/annotate_videos.py <run_dir>         # signal overlays on the videos
```

**To add a statistic to a run that predates it**, without re-inverting. Every statistic is
a pure function of the saved `(T, D)` trajectories, so a run with
`probe__save_trajectories=true` can be rebuilt on CPU in minutes:

```bash
python scripts/rescore_trajectories.py <run_dir>    # writes *_rescored.csv beside the originals
python scripts/report.py <run_dir> --figures \
    --statistics statistics_rescored.csv --signals signals_rescored.csv
```

It writes beside the originals rather than over them, so the numbers a run actually
reported stay recoverable.

## What to check when it finishes

1. **The correctness gate**, printed at the top of every inversion report. DINOv2 must land
   near GeoPhys's published 77.6–80.8% on LikePhys. If it does not, nothing downstream is
   interpretable.
2. **The reconstruction PSNR** in `reconstruction.json`, against the VAE ceiling. The
   *gap* is what inversion cost; Wan's is ~7 dB, CogVideoX-I2V's ~17 dB.
3. **The selection null** in `01_source_comparison.png`. A bar must clear the grey band and
   a diamond the dashed line — different floors for different statistics, and comparing a
   maximum against the mean floor is the easiest way to misread the results.
4. **A `_pair.mp4`**, to confirm the violation survives preprocessing at all.
5. **`figures/00_overall.png`** for the whole result on one grid, then `06_by_family.png`
   and `06_by_scenario.png` for where each signal fails. The aggregate hides that: at
   n=100 `φ_perr` on hidden states scores 91% on soft body and 57% on fluid.

### The gates that would invalidate the study

| gate | check | criterion |
|---|---|---|
| G1 | CPU tests | `python -m pytest tests/ -q` → all pass |
| G2 | layout | inversion completes without a shape error |
| G3 | inversion sanity | reconstruction PSNR is plausible, not near-zero |
| G4 | **GeoPhys implementation** | DINOv2 on LikePhys in 70–88% |
| G5 | **inversion fidelity** | accuracy varies with `num_steps` — flatness means the trajectory is not being read |
| G6 | CFG reassembly | recorded `velocity` differs between `guidance_scale` 1.0 and 6.0 |
| G7 | spectral implementation | `phase_difference_corr` favours K=2 at σ=16, near 0.358 vs 0.100 |

G4 and G5 are the two that matter most. G4 says the metric is right; G5 says we are reading
the model's trajectory rather than noise. **G5 has not been run** — `inversion__num_steps`
is still unswept, and it is the largest open threat to the headline.

## Configuration

One YAML per experiment in `configs/experiments/`, plus `section__key=value` overrides on
the command line:

```bash
python scripts/run_inversion.py --config configs/experiments/inversion_likephys_wan.yaml \
    data__limit=48 inversion__num_steps=100 probe__block_stride=4
```

**Unknown keys raise.** A typo that silently falls back to a default produces a
plausible-looking result under settings nobody chose, and after a ten-hour run there is no
way to tell. `data__limitt=5` fails immediately, as does a stray key in the YAML.

The resolved config is written to `config.json` next to the results it produced, so any
output can be traced back to the exact settings.

For a running-momentum sample, enable the denoising dashboard with the dedicated switch:

```bash
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml \
    --sample-id 0001 --guidance motion --diagnostics \
    diagnostics__record_steps=0,5,10,20,30,40,49
```

Choose the motion source with `--source latent` or `--source x0_hat`, or set
`phaselock.running_momentum_source` in YAML. `latent` preserves the established
behavior: the frame difference is measured from callback latents after the
scheduler update. `x0_hat` uses the model's clean prediction `x0_hat(z_t, t)`
from the current step, before that update. In both modes the correction is added
to the callback's post-step latents, and no extra denoiser pass is made. The
dashboard labels the current-motion panel with the selected source and its timing.
The latent setting remains `videos/motion_on_latent/`; x0 prediction runs use
`videos/motion_on_x0_hat_pred_at_t/`. To compare sources, use the same seed and
sample ID.

Diagnostics are grouped under `diagnostics/<video-stem>/<setting>/`. That folder contains
one `dashboard.mp4` joining the selected denoising steps, one full temporal movie per
selected step in `steps/`, and `dashboard.json` with moment summaries and alignment metadata.
Each movie plays the predicted clean video beside spatial maps of the selected-source
frame difference, bias-corrected mean, bias-corrected variance, and schedule-scaled guidance actually applied. These corrected moments are the values used to compute guidance; raw `m1` and `m2` are retained in summaries and optional raw traces. The bottom plots show those signals
across normalized diffusion time and across decoded video frames. Latent maps are held over
the decoded frames belonging to each latent transition, and are resized to the decoded
frame dimensions without stretching the source aspect ratio. The maps use fixed run-level
99th-percentile scales. The latent difference, corrected mean, and guidance maps show RMS over channels; corrected variance shows its channel mean. Guidance
is displayed as the RMS magnitude of `lambda * correction`. `m2` stays a latent statistic
and is not decoded as RGB. Frame zero is the conditioning anchor and has no latent transition.
Use `diagnostics__save_raw_tensors=true` only when you need the full latent tensors later.

| section | key | meaning |
|---|---|---|
| `backend` | `name`, `model_id`, `dtype`, `offload` | which model; `model_id` overrides the registry default |
| `data` | `name`, `root`, `split`, `scenarios`, `violations`, `limit`, `seed` | which clips |
| | `categories` | Physics-IQ category filter (paired datasets use `scenarios` instead) |
| | `window` | centred fraction of each clip to keep — mainly for IntPhys2 |
| | `blur_sigma` | Gaussian blur applied to every arm |
| | `height` / `width` | override the backend's frame size; `null` uses the spec. Ignored by backends that cannot honour it (see the stage table). Setting a square source to its own size (LikePhys 512×512) drops Wan's 42% letterbox padding and ~40% of the per-clip cost with it |
| `probe` | `sources` | `hidden_states`, `latent`, `x0_hat`, `velocity`, `attention` |
| | `blocks` / `block_stride` | which DiT blocks to record |
| | `pooling` | `mean` (matches GeoPhys) or `flatten` |
| | `record_steps` | how many denoising steps to probe, spread evenly |
| | `save_trajectories` | keep full `(T, D)` arrays, not just the statistics. **Set it** — it is what makes a run rescorable later |
| `inversion` | `num_steps`, `prompt` | integration steps; prompt is empty by default |
| | `reconstruction_check` | invert one clip, resample, report PSNR to `reconstruction.json` |
| `metrics` | `ar_order`, `residual_fit`, `ridge_lambda`, `bootstrap_resamples` | statistic options |
| `generation` | `num_steps`, `step_sweep`, `blur_sweep`, `guidance_scale`, `num_candidates`, `seed`, `negative_prompt` | sampling |
| `diagnostics` | `enabled`, `record_steps`, `save_raw_tensors`, `fps`, `preview_height`, `output_subdir` | opt-in moment and applied-guidance dashboard, with one replayable movie per selected denoising step |
| `exploration` | `enabled`, `mask_mode`, `noise_ratio`, `floor`, `seed`, `structured` | post-momentum noise, with an independent seed. Masks: `uniform`, `motion`, `disagreement`, `motion_disagreement`. Defaults: disabled, motion × disagreement, 0.01, 0.05, 0, true. `structured=false` is a matched-RMS noise control. |
| `refinement` | `enabled`, `steps_per_timestep`, `confidence_gate`, `seed` | CogVideoX I2V DDIM P&P. `steps_per_timestep` is the extra predictions per guided outer step. Defaults: disabled, 1, true, 0. |
| `phaselock` | `few_steps`, `guidance_strength`, `guide_start`, `guide_end` | Latent Delta Guidance, at the paper's defaults. `guide_end: null` resolves to half of `generation.num_steps` |
| | `prior_mode` | `few_step` uses the original two-step prior; `running_momentum` builds a moving reference from each denoising step |
| | `few_step_prior_source` | which tensor the prior is measured on: `latent` (the sampler state, PhaseLock's own), `x0_hat` or `velocity` |
| | `few_step_prior_type` | which quantity the few-step prior is built from and the full pass is held to: `motion` (PhaseLock's own first difference), `accel`, `jerk` or `perr` |
| | `running_momentum_source`, `running_momentum_mode` | source: `latent`, `x0_hat`, or `blend`; mode: `residual` or `snr` |
| | `beta1`, `beta2`, `velocity_decay` | first- and second-moment EMA coefficients and extra first-moment decay |
| `output` | `root`, `run_id`, `name` | artefacts land in `{root}/{run_id}/{backend}/{dataset}/{name}/` |
| | `backend` / `dataset` | filled in automatically from `backend.name` and `data.name`; set by hand only to file a run elsewhere |

### Running-momentum extensions (CogVideoX I2V)

The new experiments are opt-in and use separate run IDs. Each line in the scripts is
a standalone command that can be copied and run separately. The listed commands use
the same seven Physics-IQ clips; change the sample list to `--sample-id 0001` for a
one-sample smoke test, or remove `--sample-id` and choose new run IDs for the full set:

```bash
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 --diagnostics -- exploration__enabled=true output__run_id=explore_smoke_g42
python scripts/run_physics_iq.py --config configs/experiments/physics_iq_running_momentum.yaml --guidance motion --source latent --sample-id 0001 --diagnostics -- refinement__enabled=true refinement__steps_per_timestep=1 output__run_id=pnp_smoke_g42
```

The full sets of plain commands are in `scripts/experiments/run_momentum_exploration.sh`
and `scripts/experiments/run_momentum_refinement.sh`. To try another source, change
`--source latent` to `--source x0_hat` or `--source blend` and choose a new run ID.
The exploration script now separates mask effects with noise-only arms, sweeps noise
strength and floor, checks the cap, structure, auxiliary seeds, generation seeds, and
momentum source. The refinement script compares K=1 and K=2, includes matched 75 and
100 prediction DDIM controls, varies the P&P seed, and checks momentum source. These are
staged one-factor comparisons rather than the full Cartesian product of all settings.
For P&P, `refinement__steps_per_timestep` is the number of extra predictions per active
step; with the default window `[0,25)` and 50 outer steps, K=1 spends 75 predictions. All
P&P arms, including `K=0` controls, use deterministic CogVideoX DDIM; compare them
with each other rather than with old DPM results as a matched sampler comparison.
As in the Wan I2V reference, each inner prediction re-noises the entire preceding
DDIM-returned clean estimate at the same timestep. CogVideoX's image condition is supplied in
separate model-input channels, so P&P does not freeze the first sampled latent frame.
The running-momentum correction still leaves that frame unchanged. `x0_hat` for
momentum and diagnostics remains the backend's FP32 reconstruction from the final
model prediction, which can differ slightly from DDIM's BF16 clean estimate.

The stochastic exploration method uses the ordinary sampler. Its motion mask comes
from the current clean prediction, while the other map comes from momentum's
outer-step residual history. These maps are heuristics, not calibrated physical
uncertainty. P&P's separate within-step motion disagreement gates only the applied
momentum correction, leaving moment updates unchanged. Both extensions keep their
own random generators. They cannot be enabled together in this initial experiment.

Each run records `extension_run.json` for reuse checks, per-sample metadata under
`sampling/`, and optional `diagnostics/.../extensions.mp4` with maps and intervention
magnitudes. The evaluator-facing `videos/` directories contain only the expected
video files. For paired physics, motion, saturation, clipping, compute, and auxiliary
seed diversity comparisons, pass completed run directories to:

```bash
python scripts/report_sampling_extensions.py RUN_DIR_A RUN_DIR_B --out sampling_comparison.csv
```

The report restricts every score to the shared sample IDs. Auxiliary-seed diversity
is pairwise mean absolute RGB difference; it measures variation, not plausibility.

`limit` is a **balanced** draw across scenarios, not the first N — the first 60 LikePhys
pairs are all `ball_collision`, so a pilot using them would measure one kind of physics.

## Gotchas

- **The gate runs first.** An internal number without it is uninterpretable.
- **`drift_estimator`.** Never average `exact` and `empirical` rows together.
- **Overrides go in one run at the end of the command line.** argparse fills the positional
  `overrides` list from the first group it meets and rejects any later one.
- **14B Wan backends** load and are layout-correct but have never been run; they do not fit
  a 46 GB card.
- **`attention`** records the attention sublayer *output*, not attention matrices — a single
  block's video self-attention is 32760² at native resolution, and fused SDPA never
  materialises it.
- **LikePhys here is 800 pairs**, where the paper cites 650. Different release; absolute
  numbers will not match published ones even with a correct implementation.
- **Everything is correlational.** Geometry separating violated from plausible says the
  representation carries a usable statistical signature, not that the model represents
  physics.
