"""Experiment drivers.

* :mod:`detection` -- Stage 5. GeoPhys geometry on internal representations of real,
  labelled clips recovered by sampler inversion.
* :mod:`external` -- the published GeoPhys path on frozen encoder features, which
  validates the implementation and provides the yardstick.
* :mod:`generation` -- Stage 6. Baseline I2V continuation of LikePhys valid clips, scored
  against the real continuation and by the detector from Stage 5.
* :mod:`step_sweep` -- Stage 7. Does trajectory geometry reproduce the few-step effect,
  and does the ordering survive PhaseLock's Gaussian blur control?
"""

from .detection import (
    SignalKey,
    best_signals,
    ensemble_over_statistics,
    extract_sample,
    reconstruction_check,
    score_ensembles,
    score_signals,
    statistics_from_record,
    summarise_by_source,
    unique_samples,
    write_rows,
)
from .external import ExternalRow, encode_sample, score_external
from .generation import Candidate, best_of_n, frames_to_tensor, generate_candidate, reference_continuation
from .step_sweep import DEFAULT_BLUR, DEFAULT_STEPS, SweepCell, blur_survival, measure_cell, ordering_by_steps

__all__ = [
    "Candidate",
    "DEFAULT_BLUR",
    "DEFAULT_STEPS",
    "ExternalRow",
    "SignalKey",
    "SweepCell",
    "best_of_n",
    "best_signals",
    "blur_survival",
    "encode_sample",
    "ensemble_over_statistics",
    "extract_sample",
    "frames_to_tensor",
    "generate_candidate",
    "measure_cell",
    "reconstruction_check",
    "ordering_by_steps",
    "reference_continuation",
    "score_external",
    "score_ensembles",
    "score_signals",
    "statistics_from_record",
    "summarise_by_source",
    "unique_samples",
    "write_rows",
]
