"""Ranking metrics used to score auction policies (per-auction, then averaged).

* **NDCG@k** — how close a policy's ordering is to the value-optimal ordering,
  using graded gains (here, the *true expected value* bid x true_pCTR of each ad).
* **CTR@k** — realized clicks per shown slot in the top-k positions, i.e. the
  click yield a policy actually produces once position bias is applied.
"""
from __future__ import annotations

import numpy as np


def dcg_at_k(gains: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain with log2(rank+1) discount (gains pre-ordered)."""
    g = np.asarray(gains, dtype=float)[:k]
    if g.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, g.size + 2))
    return float(np.sum(g * discounts))


def ndcg_at_k(order_gains: np.ndarray, ideal_gains: np.ndarray, k: int) -> float:
    """NDCG@k: DCG of the policy ordering / DCG of the ideal (sorted) ordering.

    ``order_gains``  — true gains in the order the policy ranked the ads.
    ``ideal_gains``  — the same gains sorted descending (value-optimal order).
    """
    idcg = dcg_at_k(np.sort(ideal_gains)[::-1], k)
    if idcg <= 0:
        return 0.0
    return dcg_at_k(order_gains, k) / idcg


def mean_ndcg_at_k(per_auction_order_gains: list[np.ndarray], k: int) -> float:
    vals = [ndcg_at_k(g, g, k) for g in per_auction_order_gains]
    return float(np.mean(vals)) if vals else 0.0
