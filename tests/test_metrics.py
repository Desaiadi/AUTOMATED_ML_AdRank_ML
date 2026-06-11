import numpy as np

from adrank.eval.metrics import classification_report, expected_calibration_error
from adrank.auction.ranking import dcg_at_k, ndcg_at_k


def test_ece_zero_for_perfectly_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=200_000)
    y = (rng.uniform(0, 1, size=p.size) < p).astype(int)  # y ~ Bernoulli(p)
    ece, _ = expected_calibration_error(y, p, n_bins=15)
    assert ece < 0.01  # well-calibrated by construction


def test_ece_large_for_miscalibrated():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, size=50_000)
    y = (rng.uniform(0, 1, size=p.size) < np.clip(p - 0.3, 0, 1)).astype(int)
    ece, _ = expected_calibration_error(y, p, n_bins=15)
    assert ece > 0.1  # systematic over-prediction -> large ECE


def test_classification_report_keys():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, size=10_000)
    y = (rng.uniform(0, 1, size=p.size) < p).astype(int)
    rep = classification_report(y, p)
    assert {"auc", "logloss", "ece", "base_rate", "mean_pred"} <= set(rep)
    # Oracle AUC for y~Bernoulli(p), p~Uniform(0,1) is ~0.83 (P[p_pos > p_neg]).
    assert 0.78 < rep["auc"] < 0.88


def test_ndcg_ideal_is_one_and_monotone():
    gains = np.array([3.0, 2.0, 1.0, 0.5])
    assert ndcg_at_k(gains, gains, 4) == 1.0           # already ideal
    worse = gains[::-1]                                 # reversed = worst order
    assert ndcg_at_k(worse, gains, 4) < 1.0
    assert dcg_at_k(gains, 4) > dcg_at_k(worse, 4)
