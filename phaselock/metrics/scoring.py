"""Turning per-video statistics into benchmark numbers.

Implements GeoPhys's decision rule as published. For a matched pair ``(V+, V-)``
sharing initial conditions, with ``V-`` the violated video, the per-signal scalar is
the signed delta::

    delta_b = u_b(V-) - u_b(V+)

which is z-normalised to ``z_b``; its sign identifies the violated video and ``|z_b|``
is the confidence. Backends or signals are combined by **Majority** (``sum_b z_b``) or
**OR** (``argmax_b |z_b|``). Confidence intervals come from a 1000-resample bootstrap,
grouped per scenario for LikePhys and per pair for IntPhys2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class PairwiseResult:
    """Detection performance for one signal on one paired dataset."""

    accuracy: float
    ci_low: float
    ci_high: float
    n_pairs: int
    auc: Optional[float] = None

    def __str__(self) -> str:
        auc = f", AUC {self.auc:.3f}" if self.auc is not None else ""
        return (
            f"{100 * self.accuracy:.1f}% [{100 * self.ci_low:.1f}, {100 * self.ci_high:.1f}]"
            f"{auc} (n={self.n_pairs})"
        )


def signed_deltas(plausible: Sequence[float], violated: Sequence[float]) -> np.ndarray:
    """``delta = u(V-) - u(V+)``, positive when the signal ranks the pair correctly.

    Every GeoPhys statistic is oriented so that larger means less regular, hence less
    plausible, so a correct signal gives a positive delta.
    """
    plausible = np.asarray(plausible, dtype=np.float64)
    violated = np.asarray(violated, dtype=np.float64)
    if plausible.shape != violated.shape:
        raise ValueError(
            f"paired arrays must have matching shapes, got {plausible.shape} and {violated.shape}"
        )
    if plausible.size == 0:
        raise ValueError("no pairs supplied")
    return violated - plausible


def zscore(values: Sequence[float]) -> np.ndarray:
    """Mean-centred, unit-variance standardisation.

    Use this for cross-pair comparisons. Do **not** use it before an ensemble vote --
    see :func:`scale_normalize`.
    """
    values = np.asarray(values, dtype=np.float64)
    spread = values.std()
    if spread < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / spread


def scale_normalize(deltas: Sequence[float]) -> np.ndarray:
    """Divide by the spread, preserving sign. This is what the ensembles consume.

    The paper says the signed delta is "z-normalised to ``z_b``, whose sign identifies
    the violated video". Those two clauses are in tension: subtracting the mean, as a
    literal z-score does, flips the sign of any pair whose delta falls below the dataset
    mean, so a signal that ranks *every* pair correctly would still be scored wrong on
    the below-average ones. Only a scale-only normalisation keeps the sign meaningful
    while still putting different signals on a comparable footing for Majority and OR,
    so that is what is implemented here.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    spread = deltas.std()
    if spread > 1e-12:
        return deltas / spread

    # Zero spread means the signal ordered every pair by an identical margin -- the best
    # possible detector, not a useless one. Dividing by the std would zero it out and
    # throw away the sign the ensembles vote on, so fall back to the magnitude.
    scale = np.abs(deltas).mean()
    if scale < 1e-12:
        return np.zeros_like(deltas)
    return deltas / scale


def pairwise_accuracy(deltas: Sequence[float]) -> float:
    """Fraction of pairs the signal orders correctly. Exact ties score 0.5."""
    deltas = np.asarray(deltas, dtype=np.float64)
    return float((deltas > 0).mean() + 0.5 * (deltas == 0).mean())


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUC under a single global threshold, via the rank (Mann-Whitney) identity.

    Pairwise accuracy only tests *within*-pair ranking. AUC additionally tests whether
    one threshold separates violated from plausible across the whole dataset, which is
    the stronger claim and the one that matters if the score is ever used as a verifier.

    ``labels`` are 1 for violated, 0 for plausible.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positives, negatives = int((labels == 1).sum()), int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("AUC needs at least one sample of each class")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

    # Average ranks within tied groups, otherwise ties bias the statistic.
    sorted_scores = scores[order]
    start = 0
    for index in range(1, len(sorted_scores) + 1):
        if index == len(sorted_scores) or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index

    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = pairwise_accuracy,
    groups: Optional[Sequence] = None,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI.

    Args:
        groups: Resampling unit. Pass scenario labels for LikePhys and pair ids for
            IntPhys2, matching the paper. Resampling individual videos when several
            share a scenario would understate the interval, because those samples are
            not independent.
    """
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)

    if groups is None:
        indices = [np.arange(len(values))]
    else:
        groups = np.asarray(groups)
        indices = [np.flatnonzero(groups == key) for key in np.unique(groups)]

    draws = np.empty(resamples, dtype=np.float64)
    for draw in range(resamples):
        chosen = rng.integers(0, len(indices), size=len(indices))
        sample = np.concatenate([indices[k] for k in chosen])
        draws[draw] = statistic(values[sample])

    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def evaluate_deltas(
    deltas: Sequence[float],
    groups: Optional[Sequence] = None,
    resamples: int = 1000,
    seed: int = 0,
) -> PairwiseResult:
    """Pairwise accuracy and CI from signed deltas alone.

    For a score that is already a per-pair signed quantity -- an ensemble combination,
    for instance -- there are no separate plausible and violated values to recover, so
    AUC is undefined and reported as ``None``.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    low, high = bootstrap_ci(deltas, groups=groups, resamples=resamples, seed=seed)
    return PairwiseResult(
        accuracy=pairwise_accuracy(deltas), ci_low=low, ci_high=high, n_pairs=len(deltas)
    )


def evaluate_pairs(
    plausible: Sequence[float],
    violated: Sequence[float],
    groups: Optional[Sequence] = None,
    resamples: int = 1000,
    seed: int = 0,
) -> PairwiseResult:
    """Pairwise accuracy, bootstrap CI and AUC for one signal on one paired dataset."""
    deltas = signed_deltas(plausible, violated)
    low, high = bootstrap_ci(deltas, groups=groups, resamples=resamples, seed=seed)

    scores = np.concatenate([np.asarray(plausible, float), np.asarray(violated, float)])
    labels = np.concatenate([np.zeros(len(plausible), int), np.ones(len(violated), int)])

    return PairwiseResult(
        accuracy=pairwise_accuracy(deltas),
        ci_low=low,
        ci_high=high,
        n_pairs=len(deltas),
        auc=roc_auc(scores, labels),
    )


def majority_ensemble(z_by_signal: dict[str, np.ndarray]) -> np.ndarray:
    """``sum_b z_b`` -- the continuous score the paper uses for Majority ROC curves.

    Its sign is the majority vote whenever the signals are comparably scaled, which
    z-normalisation ensures.
    """
    if not z_by_signal:
        raise ValueError("no signals supplied")
    return np.sum(np.stack(list(z_by_signal.values())), axis=0)


def or_ensemble(z_by_signal: dict[str, np.ndarray]) -> np.ndarray:
    """``z_{b*}`` where ``b* = argmax_b |z_b|`` -- defer to the most confident signal.

    This is the combination that carries GeoPhys's headline numbers (98.3% LikePhys,
    93.3% IntPhys2), on the grounds that different signals catch different violations.
    """
    if not z_by_signal:
        raise ValueError("no signals supplied")
    stacked = np.stack(list(z_by_signal.values()))
    picked = np.abs(stacked).argmax(axis=0)
    return stacked[picked, np.arange(stacked.shape[1])]


def majority_vote_accuracy(z_by_signal: dict[str, np.ndarray], threshold: float = 0.5) -> float:
    """Fraction of pairs the majority of signals order correctly.

    Each signal votes with the sign of its (scale-normalised) delta. The default is a
    plain majority; the paper reports ">= 3/4 agree" over its four backbones, which is
    ``threshold=0.75``. An exact tie scores 0.5, matching :func:`pairwise_accuracy`.
    """
    if not z_by_signal:
        raise ValueError("no signals supplied")
    stacked = np.stack(list(z_by_signal.values()))
    correct = (stacked > 0).mean(axis=0)
    return float((correct > threshold).mean() + 0.5 * (correct == threshold).mean())
