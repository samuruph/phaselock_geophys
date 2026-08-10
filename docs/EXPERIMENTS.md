# Running the experiments yourself

> **Scope.** The one-command path and what to expect while it runs. For per-stage detail,
> CSV schemas and config keys see [RUNNING.md](RUNNING.md); for what is measured and why,
> [METHOD.md](METHOD.md).

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
| `SWEEP_N` | `30` | clips for the step sweep — see below |
| `VISUALS` | `4` | pairs to render videos for |
| `RESOLUTION` | `native` | `native` (512×512) or `letterbox` (480×832) |
| `STAGES` | `A B C D E` | which tracks to run |
| `RUN_ID` | timestamped | the output folder name |
| `ROOT` | `/data/experiments/phaselock_geophys` | where everything lands |

```bash
N=5 scripts/experiments/run_all.sh                    # fast end-to-end check, ~1 h
STAGES="A" scripts/experiments/run_all.sh             # just the primary track
RESOLUTION=letterbox N=20 scripts/experiments/run_all.sh
RUN_ID=20260810_1030_n100 scripts/experiments/run_all.sh   # resume into an existing run
```

**`N=2` is the floor.** Below two pairs a "pairwise accuracy" is 0% or 100% by
construction, so scoring drops every signal and no figure gets produced. Generation needs
**3 clips** for the same reason: a rank correlation over two points is meaningless.

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

```
20260810_1030_n100_native/
├── wan21_t2v_1_3b/likephys/inversion/
│   ├── config.json  statistics.csv  signals.csv  reconstruction.json
│   ├── external_latent/   the temporally matched DINOv2 gate
│   ├── figures/           01_source_comparison.png, …, <source>/…
│   ├── trajectories/      full (T, D) arrays, needed for the timeline figures
│   └── visuals/           *_pair.mp4  *_inversion.mp4  *_roundtrip.mp4  *_timeline.png
├── cogvideox_5b_i2v/likephys/{inversion,generation,step_sweep}/
└── wan21_t2v_1_3b/intphys2/inversion/
```

## Resuming

Extraction is resumable: clips already in `statistics.csv` are skipped. To continue an
interrupted run, pass its `RUN_ID` and re-issue the same command — it picks up where it
stopped. To re-report without recomputing anything:

```bash
python scripts/report.py <run_dir> --figures        # tables and figures
python scripts/annotate_videos.py <run_dir>         # signal overlays on the videos
```

Both are CPU-only and take seconds.

## What to check first when it finishes

1. **The correctness gate**, printed at the top of every inversion report. It must land
   near GeoPhys's published 77.6–80.8% on DINOv2. If it does not, nothing downstream is
   interpretable.
2. **The reconstruction PSNR** in `reconstruction.json`, against the VAE ceiling. The
   *gap* is what inversion cost; Wan's is ~7 dB, CogVideoX-I2V's ~17 dB.
3. **The selection null** in `01_source_comparison.png`. A bar must clear the grey band
   and a diamond the dashed line — they are different floors for different statistics.
4. **A `_pair.mp4`**, to confirm the violation survives preprocessing at all.
5. **`figures/06_by_scenario.png`** and `category_scenario.csv`, which say *where* each
   signal fails. The aggregate hides this: at n=96 the best cell scored 100% on four
   scenarios and 50% on `river`. Use `--category violation` for the finer split, though
   LikePhys's 56 violation types mostly have too few pairs each to mean anything.
