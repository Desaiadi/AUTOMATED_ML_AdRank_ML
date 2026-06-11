"""Pointwise evaluation metrics for probability models.

Includes the calibration metric the project headlines (Expected Calibration
Error, ECE) alongside AUC and log loss. Ranking metrics (NDCG@k, CTR@k) live in
``adrank.auction.ranking`` because they are defined per-auction, not pointwise.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> tuple[float, list[dict]]:
    """Expected Calibration Error with equal-width bins on [0, 1].

    ECE = sum_b (n_b / N) * | mean_pred_b - mean_actual_b |.

    Returns the scalar ECE and a per-bin breakdown suitable for a reliability
    diagram.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize: bin index in [1, n_bins]; clip the right edge into the last bin
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)

    n = len(y_true)
    ece = 0.0
    bins: list[dict] = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            bins.append({"bin": b, "count": 0, "mean_pred": None,
                         "mean_actual": None, "gap": None})
            continue
        mp = float(y_prob[mask].mean())
        ma = float(y_true[mask].mean())
        gap = abs(mp - ma)
        ece += (count / n) * gap
        bins.append({"bin": b, "count": count, "lo": float(edges[b]),
                     "hi": float(edges[b + 1]), "mean_pred": round(mp, 5),
                     "mean_actual": round(ma, 5), "gap": round(gap, 5)})
    return float(ece), bins


def maximum_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    _, bins = expected_calibration_error(y_true, y_prob, n_bins)
    return max((b["gap"] for b in bins if b["gap"] is not None), default=0.0)


def classification_report(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> dict:
    """AUC, log loss, ECE/MCE, base rate and mean prediction in one dict."""
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    ece, bins = expected_calibration_error(y_true, y_prob, n_bins)
    return {
        "n": int(len(y_true)),
        "auc": round(float(roc_auc_score(y_true, y_prob)), 5),
        "logloss": round(float(log_loss(y_true, y_prob)), 5),
        "ece": round(ece, 5),
        "mce": round(maximum_calibration_error(y_true, y_prob, n_bins), 5),
        "base_rate": round(float(np.mean(y_true)), 5),
        "mean_pred": round(float(np.mean(y_prob)), 5),
        "_reliability_bins": bins,
    }
