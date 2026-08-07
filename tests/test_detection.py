"""Tests for the detection driver's bookkeeping.

No GPU: statistics rows are constructed directly. What is checked here is the plumbing
that would silently corrupt a long run -- the CSV schema, the deduplication that decides
how much compute is spent, and the scoring's handling of legitimately absent values.
"""

from __future__ import annotations

import csv

import pytest
import torch

from phaselock.datasets import VideoPair, VideoSample
from phaselock.experiments.detection import (
    SignalKey,
    score_signals,
    statistics_from_record,
    summarise_by_source,
    unique_samples,
    write_rows,
)
from phaselock.metrics.geophys import STATISTICS
from phaselock.probes import LATENT, NO_BLOCK, ProbeRecord, StatisticRow, TrajectoryKey


def make_sample(sample_id: str, label: int, group: str) -> VideoSample:
    return VideoSample(
        sample_id=sample_id, path=f"/tmp/{sample_id}.mp4", label=label, group=group, dataset="test"
    )


def make_pair(index: int, scenario: str = "ball_drop") -> VideoPair:
    group = f"{scenario}/{index}"
    return VideoPair(
        plausible=make_sample(f"{group}/valid", 0, group),
        violated=make_sample(f"{group}/penetration", 1, group),
        scenario=scenario,
        violation="penetration",
    )


def make_row(sample_id: str, label: int, scenario: str, value: float, drift: bool = True) -> StatisticRow:
    return StatisticRow(
        sample_id=sample_id,
        label=label,
        group=scenario,
        scenario=scenario,
        violation="penetration",
        source=LATENT,
        block=NO_BLOCK,
        step=0,
        tau=0.5,
        statistics={name: value for name in STATISTICS},
        drift={name: -value for name in STATISTICS} if drift else None,
        drift_estimator="exact" if drift else None,
        alignment=0.3 if drift else None,
        erosion=1.2 if drift else None,
    )


# -- deduplication ----------------------------------------------------------


def test_a_shared_plausible_clip_is_inverted_only_once():
    """LikePhys pairs every violation against one valid clip per subgroup.

    Inverting per pair rather than per clip would waste roughly 40% of the compute on a
    full run, which is hours.
    """
    group = "ball_drop/0"
    valid = make_sample(f"{group}/valid", 0, group)
    pairs = [
        VideoPair(valid, make_sample(f"{group}/{kind}", 1, group), "ball_drop", kind)
        for kind in ("penetration", "teleportation", "over_bounce")
    ]

    samples = unique_samples(pairs)
    assert len(samples) == 4  # one valid + three violations, not six
    assert sum(1 for sample, _ in samples if sample.label == 0) == 1


def test_unique_samples_preserves_pair_metadata():
    pairs = [make_pair(0, "ball_drop"), make_pair(1, "pendulum")]
    scenarios = {sample.sample_id: pair.scenario for sample, pair in unique_samples(pairs)}
    assert scenarios["ball_drop/0/valid"] == "ball_drop"
    assert scenarios["pendulum/1/valid"] == "pendulum"


# -- CSV schema -------------------------------------------------------------


def test_csv_schema_is_fixed_even_when_rows_lack_drift(tmp_path):
    """The last recorded step has no drift; a header derived from row 0 would break.

    Written in the order that used to raise: a drift-less row first, then one with drift.
    """
    path = tmp_path / "statistics.csv"
    write_rows(
        [
            make_row("a", 0, "ball_drop", 1.0, drift=False),
            make_row("b", 1, "ball_drop", 2.0, drift=True),
        ],
        path,
    )

    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert set(rows[0]) == set(StatisticRow.fieldnames())
    assert rows[0]["drift_speed"] == ""       # absent, written blank
    assert rows[1]["drift_speed"] != ""


def test_appending_a_second_batch_keeps_one_header(tmp_path):
    """Resumed runs append; a second header row would poison every later read."""
    path = tmp_path / "statistics.csv"
    write_rows([make_row("a", 0, "ball_drop", 1.0, drift=False)], path)
    write_rows([make_row("b", 1, "ball_drop", 2.0, drift=True)], path)

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    assert sum(1 for line in lines if line.startswith("sample_id")) == 1


def test_write_rows_on_an_empty_batch_is_a_no_op(tmp_path):
    path = tmp_path / "statistics.csv"
    assert write_rows([], path) == 0
    assert not path.exists()


# -- scoring ----------------------------------------------------------------


def test_a_perfectly_separating_statistic_scores_one():
    pairs = [make_pair(index) for index in range(6)]
    rows = []
    for index, pair in enumerate(pairs):
        rows.append(make_row(pair.plausible.sample_id, 0, pair.scenario, 1.0 + index).flatten())
        rows.append(make_row(pair.violated.sample_id, 1, pair.scenario, 5.0 + index).flatten())

    results = score_signals(rows, pairs, resamples=50)
    key = SignalKey(LATENT, NO_BLOCK, 0, "speed", "phi")
    assert results[key].accuracy == pytest.approx(1.0)
    assert results[key].n_pairs == 6


def test_missing_drift_drops_only_the_affected_pairs():
    """A signal must not vanish because one step legitimately has no drift value."""
    pairs = [make_pair(index) for index in range(6)]
    rows = []
    for index, pair in enumerate(pairs):
        has_drift = index < 4
        rows.append(make_row(pair.plausible.sample_id, 0, pair.scenario, 1.0, has_drift).flatten())
        rows.append(make_row(pair.violated.sample_id, 1, pair.scenario, 5.0, has_drift).flatten())

    results = score_signals(rows, pairs, resamples=50)
    assert results[SignalKey(LATENT, NO_BLOCK, 0, "speed", "phi")].n_pairs == 6
    assert results[SignalKey(LATENT, NO_BLOCK, 0, "speed", "drift")].n_pairs == 4


def test_pairs_missing_a_member_are_skipped_not_mispaired():
    """An interrupted extraction leaves half-pairs; they must not be scored against noise."""
    pairs = [make_pair(index) for index in range(4)]
    rows = [make_row(pair.plausible.sample_id, 0, pair.scenario, 1.0).flatten() for pair in pairs]
    # Only the first two violations were extracted before the run stopped.
    rows += [
        make_row(pair.violated.sample_id, 1, pair.scenario, 5.0).flatten() for pair in pairs[:2]
    ]

    results = score_signals(rows, pairs, resamples=50)
    assert results[SignalKey(LATENT, NO_BLOCK, 0, "speed", "phi")].n_pairs == 2


def test_a_single_complete_pair_is_not_scored():
    """One pair cannot support a bootstrap interval; reporting it would be misleading."""
    pairs = [make_pair(index) for index in range(4)]
    rows = [make_row(pair.plausible.sample_id, 0, pair.scenario, 1.0).flatten() for pair in pairs]
    rows += [make_row(pairs[0].violated.sample_id, 1, pairs[0].scenario, 5.0).flatten()]

    assert SignalKey(LATENT, NO_BLOCK, 0, "speed", "phi") not in score_signals(rows, pairs, resamples=50)


def test_summarise_by_source_reports_the_best_signal_per_source():
    """This is the comparison the project exists to make."""
    results = {
        SignalKey("latent", -1, 0, "speed", "phi"): type("R", (), {"accuracy": 0.52})(),
        SignalKey("hidden_states", 20, 3, "curv", "phi"): type("R", (), {"accuracy": 0.81})(),
        SignalKey("hidden_states", 2, 3, "ang", "phi"): type("R", (), {"accuracy": 0.60})(),
    }
    best = summarise_by_source(results)
    assert set(best) == {"latent", "hidden_states"}
    assert best["hidden_states"][0].block == 20


def test_signal_label_is_readable():
    assert SignalKey("hidden_states", 20, 3, "curv", "phi").label() == "hidden_states/b20/s3/phi_curv"
    assert SignalKey("latent", -1, 0, "speed", "phi").label() == "latent/s0/phi_speed"


# -- record -> rows ---------------------------------------------------------


def test_statistics_from_record_covers_every_source_block_and_step():
    record = ProbeRecord()
    for step in range(3):
        record.taus[step] = 1.0 - 0.3 * step
        record.add(TrajectoryKey(LATENT, NO_BLOCK, step), torch.randn(8, 16))
        record.add(TrajectoryKey("velocity", NO_BLOCK, step), torch.randn(8, 16))
        for block in (0, 4):
            record.add(TrajectoryKey("hidden_states", block, step), torch.randn(8, 32))

    pair = make_pair(0)
    rows = statistics_from_record(record, pair.violated, pair)

    cells = {(row.source, row.block, row.step) for row in rows}
    assert len(cells) == 3 * (1 + 1 + 2)  # latent, velocity, two hidden-state blocks
    assert all(set(row.statistics) == set(STATISTICS) for row in rows)


def test_latent_rows_get_exact_drift_and_hidden_states_get_empirical():
    """The two estimators must be distinguishable in the output, never averaged."""
    record = ProbeRecord()
    for step in range(2):
        record.taus[step] = 1.0 - 0.4 * step
        record.add(TrajectoryKey(LATENT, NO_BLOCK, step), torch.randn(8, 16))
        record.add(TrajectoryKey("velocity", NO_BLOCK, step), torch.randn(8, 16))
        record.add(TrajectoryKey("hidden_states", 0, step), torch.randn(8, 32))

    pair = make_pair(0)
    rows = statistics_from_record(record, pair.violated, pair)
    estimators = {(row.source, row.drift_estimator) for row in rows if row.drift is not None}

    assert (LATENT, "exact") in estimators
    assert ("hidden_states", "empirical") in estimators
    assert not any(row.source == "hidden_states" and row.drift_estimator == "exact" for row in rows)


def test_only_the_latent_carries_alignment_and_erosion():
    """Those two are defined against the ODE drift, which only the latent trajectory is."""
    record = ProbeRecord()
    record.taus[0] = 0.5
    record.add(TrajectoryKey(LATENT, NO_BLOCK, 0), torch.randn(8, 16))
    record.add(TrajectoryKey("velocity", NO_BLOCK, 0), torch.randn(8, 16))
    record.add(TrajectoryKey("hidden_states", 0, 0), torch.randn(8, 32))

    pair = make_pair(0)
    rows = {row.source: row for row in statistics_from_record(record, pair.violated, pair)}
    assert rows[LATENT].alignment is not None
    assert rows["hidden_states"].alignment is None


# -- ensembles --------------------------------------------------------------


def test_or_ensemble_beats_its_best_component_when_signals_are_complementary():
    """The reason GeoPhys reports OR at 98.3% against ~80% for any single signal.

    Here 'speed' orders the first half of the pairs correctly and 'curv' the second half.
    Either alone is at chance; deferring to whichever is more confident gets both.
    """
    from phaselock.experiments.detection import ensemble_over_statistics

    pairs = [make_pair(index) for index in range(8)]
    rows = []
    for index, pair in enumerate(pairs):
        first_half = index < 4
        good = {name: 1.0 for name in STATISTICS}
        bad = {name: 1.0 for name in STATISTICS}
        # A large, correctly-signed margin on whichever statistic "sees" this pair.
        bad["speed"] = 5.0 if first_half else 0.9
        bad["curv"] = 0.9 if first_half else 5.0

        for sample, values in ((pair.plausible, good), (pair.violated, bad)):
            row = make_row(sample.sample_id, sample.label, pair.scenario, 1.0)
            rows.append(
                StatisticRow(
                    sample_id=row.sample_id, label=row.label, group=row.group,
                    scenario=row.scenario, violation=row.violation, source=LATENT,
                    block=NO_BLOCK, step=0, tau=0.5, statistics=values,
                ).flatten()
            )

    ensembles = ensemble_over_statistics(rows, pairs, LATENT, NO_BLOCK, 0)
    assert set(ensembles) == {"or", "majority"}
    assert ensembles["or"].accuracy == pytest.approx(1.0)


def test_score_ensembles_covers_every_probe_location():
    from phaselock.experiments.detection import score_ensembles

    pairs = [make_pair(index) for index in range(4)]
    rows = []
    for pair in pairs:
        for sample, value in ((pair.plausible, 1.0), (pair.violated, 5.0)):
            for source, block in ((LATENT, NO_BLOCK), ("hidden_states", 3)):
                row = make_row(sample.sample_id, sample.label, pair.scenario, value)
                rows.append(
                    StatisticRow(
                        sample_id=row.sample_id, label=row.label, group=row.group,
                        scenario=row.scenario, violation=row.violation, source=source,
                        block=block, step=0, tau=0.5,
                        statistics={n: value for n in STATISTICS},
                    ).flatten()
                )

    results = score_ensembles(rows, pairs)
    assert {(k.source, k.statistic) for k in results} == {
        (LATENT, "or"), (LATENT, "majority"), ("hidden_states", "or"), ("hidden_states", "majority")
    }
    assert all(result.accuracy == pytest.approx(1.0) for result in results.values())


def test_ensembles_decline_a_location_with_too_few_pairs():
    from phaselock.experiments.detection import ensemble_over_statistics

    pairs = [make_pair(0)]
    rows = [
        make_row(pairs[0].plausible.sample_id, 0, "ball_drop", 1.0).flatten(),
        make_row(pairs[0].violated.sample_id, 1, "ball_drop", 5.0).flatten(),
    ]
    assert ensemble_over_statistics(rows, pairs, LATENT, NO_BLOCK, 0) == {}
