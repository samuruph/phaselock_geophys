"""Tests for the pairwise decision rule, ensembles and confidence intervals."""

from __future__ import annotations

import numpy as np
import pytest

from phaselock.metrics.scoring import (
    bootstrap_ci,
    evaluate_deltas,
    evaluate_pairs,
    majority_ensemble,
    majority_vote_accuracy,
    or_ensemble,
    pairwise_accuracy,
    roc_auc,
    scale_normalize,
    signed_deltas,
    zscore,
)


def test_signed_delta_is_violated_minus_plausible():
    """Every GeoPhys statistic is larger for less regular motion, so correct is positive."""
    deltas = signed_deltas(plausible=[1.0, 2.0], violated=[3.0, 1.0])
    assert list(deltas) == [2.0, -1.0]


def test_signed_deltas_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="matching shapes"):
        signed_deltas([1.0, 2.0], [1.0])


def test_signed_deltas_rejects_empty_input():
    with pytest.raises(ValueError, match="no pairs"):
        signed_deltas([], [])


def test_pairwise_accuracy_counts_ties_as_half():
    assert pairwise_accuracy([1.0, 1.0, -1.0, -1.0]) == pytest.approx(0.5)
    assert pairwise_accuracy([0.0, 0.0]) == pytest.approx(0.5)
    assert pairwise_accuracy([1.0, 2.0, 3.0]) == pytest.approx(1.0)


# -- normalisation ----------------------------------------------------------


def test_scale_normalize_preserves_sign_but_zscore_does_not():
    """The ensembles need sign-preserving normalisation.

    The paper says the delta is "z-normalised to z_b, whose sign identifies the violated
    video". Those clauses conflict: mean-centring flips the sign of any pair below the
    dataset mean, so a signal that ranks every pair correctly would be scored wrong on the
    below-average ones. Only scale-only normalisation keeps the sign meaningful.
    """
    deltas = [1.0, 2.0, 3.0, 10.0]  # all correct
    assert (scale_normalize(deltas) > 0).all()
    assert not (zscore(deltas) > 0).all()


def test_normalisers_handle_a_constant_input_without_nans():
    assert list(scale_normalize([2.0, 2.0, 2.0])) == [0.0, 0.0, 0.0]
    assert list(zscore([2.0, 2.0, 2.0])) == [0.0, 0.0, 0.0]


def test_scale_normalize_is_scale_invariant():
    deltas = np.array([1.0, -2.0, 3.0])
    assert np.allclose(scale_normalize(deltas), scale_normalize(deltas * 1000))


# -- AUC --------------------------------------------------------------------


def test_auc_endpoints():
    assert roc_auc([0.0, 1.0, 2.0, 3.0], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert roc_auc([3.0, 2.0, 1.0, 0.0], [0, 0, 1, 1]) == pytest.approx(0.0)


def test_auc_of_completely_tied_scores_is_one_half():
    """Ties must use average ranks; otherwise a constant score scores 0 or 1."""
    assert roc_auc([1.0] * 6, [0, 0, 0, 1, 1, 1]) == pytest.approx(0.5)


def test_auc_matches_the_hand_computed_value_with_partial_ties():
    # negatives {0, 1}, positives {1, 2}: the tie at 1 contributes half a win.
    assert roc_auc([0.0, 1.0, 1.0, 2.0], [0, 0, 1, 1]) == pytest.approx(0.875)


def test_auc_requires_both_classes():
    with pytest.raises(ValueError, match="one sample of each class"):
        roc_auc([1.0, 2.0], [1, 1])


# -- bootstrap --------------------------------------------------------------


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    deltas = rng.normal(0.5, 1.0, size=200)
    low, high = bootstrap_ci(deltas, resamples=500)
    assert low <= pairwise_accuracy(deltas) <= high


def test_bootstrap_is_deterministic_given_a_seed():
    deltas = np.random.default_rng(1).normal(0.3, 1.0, size=100)
    assert bootstrap_ci(deltas, resamples=200, seed=7) == bootstrap_ci(deltas, resamples=200, seed=7)


def test_grouped_bootstrap_widens_the_interval_for_correlated_samples():
    """LikePhys shares a valid clip across a subgroup's violations, so pairs from one
    scenario are not independent. Resampling videos rather than scenarios would understate
    the interval.
    """
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 20)
    # Strong per-scenario effect: within a group the outcome is nearly constant.
    deltas = np.repeat(rng.normal(0.4, 1.2, size=10), 20) + rng.normal(0, 0.05, size=200)

    ungrouped = bootstrap_ci(deltas, resamples=400, seed=0)
    grouped = bootstrap_ci(deltas, groups=groups, resamples=400, seed=0)
    assert (grouped[1] - grouped[0]) > (ungrouped[1] - ungrouped[0])


# -- end-to-end evaluation --------------------------------------------------


def test_evaluate_pairs_reports_a_perfect_signal_as_perfect():
    result = evaluate_pairs([0.0] * 20, [1.0] * 20, resamples=200)
    assert result.accuracy == pytest.approx(1.0)
    assert result.auc == pytest.approx(1.0)
    assert result.n_pairs == 20


def test_evaluate_pairs_reports_an_inverted_signal_as_zero():
    """A statistic pointing the wrong way should read 0%, not 100%."""
    result = evaluate_pairs([1.0] * 20, [0.0] * 20, resamples=200)
    assert result.accuracy == pytest.approx(0.0)


def test_evaluate_deltas_has_no_auc():
    """An ensemble output is already a signed delta; there are no class values to rank."""
    result = evaluate_deltas([1.0, 1.0, -1.0, 1.0], resamples=200)
    assert result.auc is None
    assert result.accuracy == pytest.approx(0.75)


def test_result_formats_as_a_readable_summary():
    text = str(evaluate_pairs([0.0] * 10, [1.0] * 10, resamples=100))
    assert "100.0%" in text and "n=10" in text


# -- ensembles --------------------------------------------------------------


def test_or_ensemble_defers_to_the_most_confident_signal():
    signals = {
        "a": np.array([0.5, -3.0]),
        "b": np.array([-2.0, 1.0]),
    }
    assert list(or_ensemble(signals)) == [-2.0, -3.0]


def test_majority_ensemble_sums_the_normalised_signals():
    signals = {"a": np.array([1.0, -1.0]), "b": np.array([2.0, 0.5])}
    assert list(majority_ensemble(signals)) == [3.0, -0.5]


def test_majority_vote_uses_signs_not_magnitudes():
    """Two weakly-correct signals must outvote one strongly-wrong one."""
    signals = {
        "a": np.array([0.1, 0.1]),
        "b": np.array([0.1, 0.1]),
        "c": np.array([-9.0, -9.0]),
    }
    assert majority_vote_accuracy(signals) == pytest.approx(1.0)


def test_majority_vote_threshold_matches_the_papers_three_of_four_rule():
    signals = {
        "a": np.array([1.0]),
        "b": np.array([1.0]),
        "c": np.array([-1.0]),
        "d": np.array([-1.0]),
    }
    # A 2-2 split is a tie under a plain majority and a failure under >= 3/4.
    assert majority_vote_accuracy(signals, threshold=0.5) == pytest.approx(0.5)
    assert majority_vote_accuracy(signals, threshold=0.75) == pytest.approx(0.0)


def test_ensembles_reject_empty_input():
    for combine in (or_ensemble, majority_ensemble, majority_vote_accuracy):
        with pytest.raises(ValueError, match="no signals"):
            combine({})
