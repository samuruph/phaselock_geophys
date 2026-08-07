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

from ..backends.base import VideoBackend
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
    pair: VideoPair,
    order: int = 3,
    fit: str = "span",
) -> list[StatisticRow]:
    """Turn one video's probe record into statistics rows.

    Exact flow coupling is computed only for the latent, which is the ODE state; every
    other source gets the finite-difference estimator across consecutive recorded steps,
    tagged so the two are never aggregated together.
    """
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
                        scenario=pair.scenario,
                        violation=pair.violation,
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
    result = invert(
        backend,
        frames,
        num_steps=config.inversion.num_steps,
        record_steps=1,
        sources=[LATENT],
        prompt=config.inversion.prompt,
    )
    recovered = backend.decode(
        resample(
            backend,
            result.noise,
            num_steps=config.inversion.num_steps,
            prompt=config.inversion.prompt,
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

        for kind in ("phi", "drift"):
            for name in STATISTICS:
                column = f"{kind}_{name}"
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


def _maybe_float(value: Any) -> Optional[float]:
    """Parse a CSV cell, treating blanks and non-finite values as absent."""
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


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
) -> Optional[PairwiseResult]:
    """OR ensemble across the five statistics at one probe location.

    The paper's OR rule -- defer to whichever signal is most confident -- carries its
    headline numbers, on the grounds that different signals catch different violations.
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
    if not usable:
        return None

    normalised: dict[str, Any] = {}
    for name in STATISTICS:
        column = f"phi_{name}"
        deltas = [
            float(by_sample[p.violated.sample_id][column])
            - float(by_sample[p.plausible.sample_id][column])
            for p in usable
        ]
        normalised[name] = scale_normalize(deltas)

    # The ensemble output is already a per-pair signed delta, so it is scored directly.
    return evaluate_deltas(or_ensemble(normalised), groups=[pair.scenario for pair in usable])
