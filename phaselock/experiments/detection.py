"""Stage 5: does GeoPhys geometry separate plausible from violated video?

For every clip in a matched pair: decode, resample, letterbox, VAE-encode, invert the
sampler while recording internal states, pool, and compute the five geometric statistics
plus the flow-coupling metrics for every ``(source, block, step)``.

Then score. The result is a table of pairwise accuracies indexed by
``(source, block, step, statistic)``, which is what answers the project's question: not
"does geometry work" but *which internal representation* it works on, and where in depth
and denoising time.

Two reference points frame every internal number:

* the **DINOv2 external baseline**, which validates the statistics against the paper's
  published 78-81% on LikePhys;
* the **VAE latent control**, where "The Invisible Hand of Physics" reports linear probes
  at chance. Geometry landing at chance there while working on hidden states would be a
  clean positive result, and would say PhaseLock guides in a physics-free space.
"""

from __future__ import annotations

import csv
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch

from ..backends.base import VideoBackend, to_canonical
from ..config import Config
from ..datasets import VideoPair, VideoSample, load_video
from ..metrics.flow_geometry import (
    DriftEstimator,
    erosion_rate,
    geometric_drift,
    geometric_drift_empirical,
    transport_alignment,
)
from ..metrics.geophys import STATISTICS, geophys_statistics
from ..metrics.scoring import (
    PairwiseResult,
    evaluate_deltas,
    evaluate_pairs,
    majority_ensemble,
    or_ensemble,
    scale_normalize,
)
from ..pipelines.inversion import invert
from ..probes import LATENT, VELOCITY, ProbeRecord, StatisticRow, supports_exact_drift

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalKey:
    """Identifies one scalar signal that can be scored as a detector."""

    source: str
    block: int
    step: int
    statistic: str
    kind: str = "phi"
    """``"phi"`` for the statistic itself, ``"drift"`` for its rate of change."""

    def label(self) -> str:
        block = "" if self.block < 0 else f"/b{self.block}"
        return f"{self.source}{block}/s{self.step}/{self.kind}_{self.statistic}"


def statistics_from_record(
    record: ProbeRecord,
    sample: VideoSample,
    pair: Optional[VideoPair] = None,
    order: int = 3,
    fit: str = "span",
) -> list[StatisticRow]:
    """Turn one video's probe record into statistics rows.

    Exact flow coupling is computed only for the latent, which is the ODE state; every
    other source gets the finite-difference estimator across consecutive recorded steps,
    tagged so the two are never aggregated together.

    ``pair`` is optional because generation has no matched pair: it conditions on a valid
    clip and there is nothing to contrast against within the run. The scenario then comes
    from the sample itself and the violation field is empty, so a generation run produces
    exactly the same schema and every downstream reader keeps working.
    """
    scenario = pair.scenario if pair is not None else sample.meta.get("scenario", "")
    violation = pair.violation if pair is not None else ""
    rows: list[StatisticRow] = []
    steps = record.steps

    for source in record.sources:
        for block in record.blocks(source):
            for position, step in enumerate(steps):
                key = (source, step, block)
                try:
                    trajectory = record.get(*key)
                except KeyError:
                    continue

                stats = {
                    name: float(value)
                    for name, value in geophys_statistics(
                        trajectory, order=order, fit=fit
                    ).items()
                }

                drift: Optional[dict[str, float]] = None
                estimator: Optional[str] = None
                alignment: Optional[float] = None
                erosion: Optional[float] = None

                if supports_exact_drift(source) and VELOCITY in record.sources:
                    flow = record.get(VELOCITY, step)
                    drift = geometric_drift(trajectory, flow, order=order, fit=fit)
                    estimator = DriftEstimator.EXACT.value
                    alignment, _ = transport_alignment(trajectory, flow)
                    erosion = erosion_rate(trajectory, flow)
                elif position + 1 < len(steps):
                    next_step = steps[position + 1]
                    delta_tau = record.taus[next_step] - record.taus[step]
                    if abs(delta_tau) > 1e-9:
                        drift = geometric_drift_empirical(
                            trajectory,
                            record.get(source, next_step, block),
                            delta_tau,
                            order=order,
                            fit=fit,
                        )
                        estimator = DriftEstimator.EMPIRICAL.value

                rows.append(
                    StatisticRow(
                        sample_id=sample.sample_id,
                        label=sample.label,
                        group=sample.group,
                        scenario=scenario,
                        violation=violation,
                        source=source,
                        block=block,
                        step=step,
                        tau=record.taus[step],
                        statistics=stats,
                        drift=drift,
                        drift_estimator=estimator,
                        alignment=alignment,
                        erosion=erosion,
                    )
                )
    return rows


def image_conditioning(
    backend: VideoBackend, frames: torch.Tensor, prompt: str
) -> Optional[dict[str, Any]]:
    """Conditioning for an I2V backend, or None to let the caller build its own.

    An image-to-video transformer takes the conditioning frame concatenated onto its
    input channels, so inverting a real clip on one fails on shape unless those channels
    are supplied. The conditioning frame is the clip's own first frame -- the same frame
    generation would be given -- so the recovered trajectory is the one that model would
    have taken for this video.

    Text-to-video backends have no such input and are left alone.
    """
    encode = getattr(backend, "encode_image_condition", None)
    if encode is None or getattr(backend, "mode", "t2v") != "i2v":
        return None
    return backend.prepare_conditioning(prompt=prompt, image_latents=encode(frames))


def extract_sample(
    backend: VideoBackend,
    sample: VideoSample,
    pair: VideoPair,
    config: Config,
    trajectory_dir: Optional[Path] = None,
) -> list[StatisticRow]:
    """Invert one clip and reduce it to statistics rows."""
    spec = backend.spec
    frames = load_video(
        sample.path,
        num_frames=spec.default_num_frames,
        height=spec.default_height,
        width=spec.default_width,
        window=config.data.window,
        blur_sigma=config.data.blur_sigma,
    )

    sources = list(config.probe.sources)
    # Exact flow coupling for the latent needs the drift recorded alongside it.
    if LATENT in sources and VELOCITY not in sources:
        sources.append(VELOCITY)

    result = invert(
        backend,
        frames,
        num_steps=config.inversion.num_steps,
        record_steps=config.probe.record_steps,
        sources=sources,
        blocks=config.probe.blocks,
        block_stride=config.probe.block_stride,
        pooling=config.probe.pooling,
        prompt=config.inversion.prompt,
        conditioning=image_conditioning(backend, frames, config.inversion.prompt),
        provenance={
            "sample_id": sample.sample_id,
            "label": sample.label,
            "scenario": pair.scenario,
            "violation": pair.violation,
            "dataset": sample.dataset,
            "blur_sigma": config.data.blur_sigma,
            "window": config.data.window,
        },
    )

    if trajectory_dir is not None:
        result.record.save(trajectory_dir / sample.sample_id.replace("/", "_"))

    return statistics_from_record(
        result.record, sample, pair, order=config.metrics.ar_order, fit=config.metrics.residual_fit
    )


def save_visuals(
    backend: VideoBackend,
    pair: VideoPair,
    config: Config,
    directory: Path,
    video_steps: int = 6,
) -> list[Path]:
    """Write videos so a run can be looked at, not just trusted.

    Videos only. Most LikePhys violations are purely temporal -- a freeze, a jitter, a
    shuffled segment -- and simply do not exist in sampled stills, which is what the
    contact sheets these replaced could show.

    Two per pair:

    * ``_pair.mp4`` -- plausible and violated side by side on one clock. Does the
      violation survive preprocessing at all?
    * ``_inversion.mp4`` -- the original, then the model's clean estimate at each
      recorded step, all panels playing together, so the trajectory's decay is visible
      as motion. Decoding the noisy latent instead would show noise at every step and
      answer nothing.

    The reconstruction PSNR that the old sheet carried is logged instead, and
    ``reconstruction_check`` records it per run.
    """
    from ..analysis import video
    from ..metrics.motion_mask import psnr
    from ..pipelines.inversion import resample

    spec = backend.spec
    written: list[Path] = []
    load = lambda sample: load_video(
        sample.path, num_frames=spec.default_num_frames,
        height=spec.default_height, width=spec.default_width,
        window=config.data.window, blur_sigma=config.data.blur_sigma,
    )

    plausible, violated = load(pair.plausible), load(pair.violated)
    stem = pair.violated.sample_id.replace("/", "_")
    written.append(video.pair_video(
        plausible, violated, directory / f"{stem}_pair.mp4",
        scenario=pair.scenario, violation=pair.violation,
    ))

    # Decode the clean estimate at each recorded step. Held on CPU: a decoded 81-frame
    # clip is ~300 MB in fp32 and there are `video_steps` of them.
    steps: list[tuple[float, "torch.Tensor"]] = []

    def keep(index: int, tau: float, state) -> None:
        steps.append((tau, backend.decode(to_canonical(state.x0, spec)).cpu()))

    latents = backend.encode(plausible)
    vae_only = backend.decode(latents).cpu()
    conditioning = image_conditioning(backend, plausible, config.inversion.prompt)
    noise = invert(
        backend, latents=latents, num_steps=config.inversion.num_steps,
        record_steps=video_steps, sources=[LATENT], prompt=config.inversion.prompt,
        conditioning=conditioning, on_record=keep,
    ).noise
    written.append(video.inversion_video(
        steps, directory / f"{stem}_inversion.mp4", original=plausible, kind="x0_hat",
    ))

    recovered = backend.decode(
        resample(backend, noise, num_steps=config.inversion.num_steps,
                 prompt=config.inversion.prompt, conditioning=conditioning)
    ).cpu()
    # The middle panel is the ceiling: no inversion can beat what the VAE alone keeps.
    # Comparing against it separates "the inversion lost this" from "the VAE never had
    # it", which a single PSNR cannot.
    written.append(video.write_grid(
        {
            "original": plausible,
            f"VAE only  {psnr(vae_only, plausible):.1f} dB": vae_only,
            f"inverted  {psnr(recovered, plausible):.1f} dB": recovered,
        },
        directory / f"{stem}_roundtrip.mp4",
    ))
    return written


def reconstruction_check(
    backend: VideoBackend, sample: VideoSample, config: Config
) -> dict[str, float]:
    """Invert one clip, resample the recovered noise, and compare against the source.

    Guards against an inversion that is simply broken. It cannot tell you the trajectory
    is *good enough*: the Invisible Hand reports probe accuracy collapsing 0.82 -> 0.57
    between 100 and 20 steps while the reconstruction stays visually faithful. That is
    what the ``inversion__num_steps`` sweep is for.
    """
    from ..metrics.motion_mask import psnr
    from ..pipelines.inversion import resample

    spec = backend.spec
    frames = load_video(
        sample.path,
        num_frames=spec.default_num_frames,
        height=spec.default_height,
        width=spec.default_width,
        window=config.data.window,
    )
    conditioning = image_conditioning(backend, frames, config.inversion.prompt)
    result = invert(
        backend,
        frames,
        num_steps=config.inversion.num_steps,
        record_steps=1,
        sources=[LATENT],
        prompt=config.inversion.prompt,
        conditioning=conditioning,
    )
    recovered = backend.decode(
        resample(
            backend,
            result.noise,
            num_steps=config.inversion.num_steps,
            prompt=config.inversion.prompt,
            # Same conditioning coming back as going out, or an I2V backend rebuilds it
            # without the image channels and the transformer gets 16 where it wants 32.
            conditioning=conditioning,
        )
    )
    return {
        "psnr": psnr(recovered.cpu(), frames.cpu()),
        "mse": float((recovered.cpu() - frames.cpu()).pow(2).mean()),
        "num_steps": float(config.inversion.num_steps),
    }


def unique_samples(pairs: Sequence[VideoPair]) -> list[tuple[VideoSample, VideoPair]]:
    """Every distinct clip exactly once, with a pair for its metadata.

    LikePhys shares one valid clip across all violations of a subgroup -- 800 pairs cover
    only 920 distinct videos. Inverting per pair would waste roughly 40% of the compute.
    """
    seen: dict[str, tuple[VideoSample, VideoPair]] = {}
    for pair in pairs:
        for sample in (pair.plausible, pair.violated):
            seen.setdefault(sample.sample_id, (sample, pair))
    return list(seen.values())


def write_rows(rows: Iterable[StatisticRow], path: Path) -> int:
    """Append statistics rows to a CSV under a fixed schema.

    The header comes from :meth:`StatisticRow.fieldnames`, not from the first row, so
    that appends from separate batches stay consistent even when some rows have no drift.
    """
    rows = list(rows)
    if not rows:
        return 0
    exists = path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=StatisticRow.fieldnames())
        if not exists:
            writer.writeheader()
        writer.writerows(row.flatten() for row in rows)
    return len(rows)


def score_signals(
    rows: Sequence[dict[str, Any]],
    pairs: Sequence[VideoPair],
    resamples: int = 1000,
) -> dict[SignalKey, PairwiseResult]:
    """Pairwise accuracy and AUC for every scalar signal in the table.

    A signal is one ``(source, block, step, statistic)`` combination, scored by the
    paper's rule: the pair member with the larger value is predicted violated.
    """
    indexed: dict[tuple, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        cell = (row["source"], int(row["block"]), int(row["step"]))
        indexed[cell][row["sample_id"]] = row

    results: dict[SignalKey, PairwiseResult] = {}
    for cell, by_sample in indexed.items():
        source, block, step = cell
        usable = [
            pair
            for pair in pairs
            if pair.plausible.sample_id in by_sample and pair.violated.sample_id in by_sample
        ]
        if not usable:
            continue

        # "coupling" carries the two flow-coupling metrics, which live in bare columns
        # rather than under a phi_/drift_ prefix. They were measured from the start but
        # never scored, so they were absent from every figure.
        scorable = [(kind, name) for kind in ("phi", "drift") for name in STATISTICS]
        scorable += [("coupling", "alignment"), ("coupling", "erosion")]

        for kind, name in scorable:
            if True:
                column = name if kind == "coupling" else f"{kind}_{name}"
                # Drop only the pairs with a missing value, not the whole signal: the
                # last recorded step legitimately has no drift, and the latent carries
                # alignment where other sources do not.
                complete = [
                    (
                        _maybe_float(by_sample[pair.plausible.sample_id].get(column)),
                        _maybe_float(by_sample[pair.violated.sample_id].get(column)),
                        pair.scenario,
                    )
                    for pair in usable
                ]
                complete = [entry for entry in complete if entry[0] is not None and entry[1] is not None]
                if len(complete) < 2:
                    continue

                results[SignalKey(source, block, step, name, kind)] = evaluate_pairs(
                    [entry[0] for entry in complete],
                    [entry[1] for entry in complete],
                    groups=[entry[2] for entry in complete],
                    resamples=resamples,
                )
    return results


def delta_matrix(
    rows: Sequence[dict[str, Any]], pairs: Sequence[VideoPair], kind: str = "phi"
) -> "np.ndarray":
    """Signed deltas for every scorable signal, shaped ``(n_signals, n_pairs)``.

    Feeds :func:`~phaselock.metrics.scoring.selection_null`, which needs the same signal
    set that the reported maximum was selected from -- otherwise the null is measuring a
    different selection procedure than the one that produced the number.

    Only pairs complete across *every* signal are kept, so each column is one pair.
    """
    import numpy as np

    indexed: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        indexed[(row["source"], int(row["block"]), int(row["step"]))][row["sample_id"]] = row

    usable = [
        pair
        for pair in pairs
        if all(
            pair.plausible.sample_id in cell and pair.violated.sample_id in cell
            for cell in indexed.values()
        )
    ]
    if len(usable) < 2:
        return np.empty((0, 0))

    columns: list[list[float]] = []
    for cell in indexed.values():
        for name in STATISTICS:
            column = f"{kind}_{name}"
            values = [
                (
                    _maybe_float(cell[pair.violated.sample_id].get(column)),
                    _maybe_float(cell[pair.plausible.sample_id].get(column)),
                )
                for pair in usable
            ]
            if any(bad is None or good is None for bad, good in values):
                continue
            columns.append([bad - good for bad, good in values])
    return np.asarray(columns, dtype=np.float64) if columns else np.empty((0, 0))


def _maybe_float(value: Any) -> Optional[float]:
    """Parse a CSV cell, treating blanks and non-finite values as absent."""
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


@dataclass(frozen=True)
class SourceSummary:
    """How a source performs across its whole probe grid, not just at its best cell.

    Reporting only the maximum over a (block x step x statistic) grid is a selection
    procedure and reads high by construction. The mean and spread over the grid is the
    honest summary of "what this representation gives you"; the best is kept alongside
    only so the two can be compared, and it is meaningful only against the permutation
    null.
    """

    source: str
    statistic: str
    kind: str
    mean: float
    std: float
    n_cells: int
    best: float
    best_label: str

    def __str__(self) -> str:
        return (
            f"{100 * self.mean:.1f}% +/- {100 * self.std:.1f}  "
            f"(best {100 * self.best:.1f}% at {self.best_label}, {self.n_cells} cells)"
        )


def summarise_grid(
    results: dict[SignalKey, PairwiseResult], per_statistic: bool = True
) -> list[SourceSummary]:
    """Mean, spread and best accuracy per source, optionally split by statistic.

    Averaging over the grid mixes probe locations, which is the point: a representation
    that only works at one hand-picked cell out of hundreds is not a usable detector.
    """
    import statistics as stats_module

    buckets: dict[tuple[str, str, str], list[tuple[float, SignalKey]]] = defaultdict(list)
    for key, result in results.items():
        name = key.statistic if per_statistic else "all"
        buckets[(key.source, name, key.kind)].append((result.accuracy, key))

    summaries: list[SourceSummary] = []
    for (source, name, kind), entries in buckets.items():
        values = [value for value, _ in entries]
        best_value, best_key = max(entries, key=lambda item: item[0])
        summaries.append(
            SourceSummary(
                source=source, statistic=name, kind=kind,
                mean=sum(values) / len(values),
                std=stats_module.pstdev(values) if len(values) > 1 else 0.0,
                n_cells=len(values), best=best_value, best_label=best_key.label(),
            )
        )
    return sorted(summaries, key=lambda item: -item.mean)


def best_signals(
    results: dict[SignalKey, PairwiseResult], top: int = 20
) -> list[tuple[SignalKey, PairwiseResult]]:
    """Signals ranked by pairwise accuracy."""
    return sorted(results.items(), key=lambda item: item[1].accuracy, reverse=True)[:top]


def summarise_by_source(
    results: dict[SignalKey, PairwiseResult]
) -> dict[str, tuple[SignalKey, PairwiseResult]]:
    """The single best signal per source.

    This is the comparison the project exists to make: which internal representation
    carries the geometry.
    """
    best: dict[str, tuple[SignalKey, PairwiseResult]] = {}
    for key, result in results.items():
        current = best.get(key.source)
        if current is None or result.accuracy > current[1].accuracy:
            best[key.source] = (key, result)
    return best


def ensemble_over_statistics(
    rows: Sequence[dict[str, Any]],
    pairs: Sequence[VideoPair],
    source: str,
    block: int,
    step: int,
    kind: str = "phi",
) -> dict[str, PairwiseResult]:
    """Combine the five statistics at one probe location, by OR and by Majority.

    These are the paper's headline numbers, not an afterthought: OR (defer to whichever
    signal is most confident) reaches 98.3% on LikePhys against 77.6-80.8% for the best
    single backbone, on the grounds that different signals catch different violations.
    Reporting only single signals understates the method by roughly 18 points.

    Returns an empty dict when the location has too few complete pairs to score.
    """
    by_sample = {
        row["sample_id"]: row
        for row in rows
        if row["source"] == source and int(row["block"]) == block and int(row["step"]) == step
    }
    usable = [
        pair
        for pair in pairs
        if pair.plausible.sample_id in by_sample and pair.violated.sample_id in by_sample
    ]
    if len(usable) < 2:
        return {}

    normalised: dict[str, Any] = {}
    for name in STATISTICS:
        column = f"{kind}_{name}"
        deltas = [
            (
                _maybe_float(by_sample[pair.violated.sample_id].get(column)),
                _maybe_float(by_sample[pair.plausible.sample_id].get(column)),
            )
            for pair in usable
        ]
        if any(bad is None or good is None for bad, good in deltas):
            continue
        normalised[name] = scale_normalize([bad - good for bad, good in deltas])

    if not normalised:
        return {}

    groups = [pair.scenario for pair in usable]
    # Both ensemble outputs are already per-pair signed deltas, so they score directly.
    return {
        "or": evaluate_deltas(or_ensemble(normalised), groups=groups),
        "majority": evaluate_deltas(majority_ensemble(normalised), groups=groups),
    }


def score_ensembles(
    rows: Sequence[dict[str, Any]],
    pairs: Sequence[VideoPair],
    kind: str = "phi",
) -> dict[SignalKey, PairwiseResult]:
    """OR and Majority at every probe location, keyed like the single-signal results."""
    locations = {(row["source"], int(row["block"]), int(row["step"])) for row in rows}
    results: dict[SignalKey, PairwiseResult] = {}
    for source, block, step in sorted(locations):
        for rule, result in ensemble_over_statistics(
            rows, pairs, source, block, step, kind=kind
        ).items():
            results[SignalKey(source, block, step, rule, kind)] = result
    return results
