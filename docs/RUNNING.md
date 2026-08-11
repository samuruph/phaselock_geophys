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
| `generation` | `num_steps`, `step_sweep`, `blur_sweep`, `guidance_scale`, `num_candidates`, `seed` | sampling |
| `phaselock` | `few_steps`, `guidance_strength`, `guide_start`, `guide_end` | Latent Delta Guidance, at the paper's defaults. `guide_end: null` resolves to half of `generation.num_steps` |
| | `few_step_prior_type` | which quantity the few-step prior is built from and the full pass is held to: `motion` (PhaseLock's own first difference), `accel`, `jerk` or `perr` |
| | `negative_prompt` | passed through to the pipeline; `null` by default |
| `output` | `root`, `run_id`, `name` | artefacts land in `{root}/{run_id}/{backend}/{dataset}/{name}/` |
| | `backend` / `dataset` | filled in automatically from `backend.name` and `data.name`; set by hand only to file a run elsewhere |

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
