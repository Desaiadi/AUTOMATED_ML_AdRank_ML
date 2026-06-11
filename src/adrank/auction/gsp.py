"""GSP-style auction simulator.

We reconstruct per-keyword auctions from the scored TEST impressions, then run
four ranking policies on the SAME auctions:

* ``bid_only``  - rank by bid alone (the pre-ML baseline: ignores quality).
* ``adrank``    - rank by ``bid * pCTR_model`` (classic GSP "Ad Rank").
* ``ev``        - rank by ``bid * pCTR_model * pCVR_model`` (expected-value ranking;
                  the headline ML policy).
* ``ideal``     - rank by ``bid * pCTR_true * pCVR_true`` (value-optimal upper bound).

Each candidate carries:
  * ``pctr_true``  = sigmoid(relevance_logit)        - position-independent truth
  * ``pctr_model`` = p_ctr_pred / examine(orig_pos)  - position-normalised estimate
  * ``pcvr_true``  = p_conv_true                     - intrinsic conversion truth
  * ``pcvr_model`` = p_cvr_pred                      - model P(conv | click)

Top slots are filled by policy score; a click is drawn as
``Bernoulli(examine(slot) * pctr_true)`` using **common random numbers** across
policies (one uniform per candidate) so policy differences aren't masked by noise.
GSP charges each winner the minimum bid needed to keep its slot:
``price_i = score_{i+1} / quality_i`` floored at the reserve, paid only on a click.

NDCG@10 uses the conversion-weighted gain ``pctr_true * pcvr_true`` (true expected
value), so it rewards ranking by end-to-end value rather than raw clicks. That is
why the ``ev`` policy lifts NDCG@10 more than it lifts CTR@10 - it intentionally
deprioritises high-click / low-conversion ads. Headline = ``ev`` vs ``bid_only``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from ..config import Config
from ..utils.common import get_logger, save_json, timed
from .ranking import ndcg_at_k

LOGGER = get_logger("adrank.auction")

POLICIES = ("bid_only", "adrank", "ev", "ideal")
HEADLINE_ML = "ev"


def _build_candidate_pools(cfg: Config) -> tuple[pd.DataFrame, dict]:
    proc_dir = Path(cfg.paths["processed_dir"])
    df = pd.read_parquet(proc_dir / "scored.parquet")
    decay = float(cfg.dgp.position_decay)

    examine_orig = np.power(decay, df["position"].to_numpy() - 1)
    df = df.assign(
        pctr_true=expit(df["relevance_logit"].to_numpy()),
        pctr_model=np.clip(df["p_ctr_pred"].to_numpy() / examine_orig, 1e-4, 0.999),
        pcvr_true=df["p_conv_true"].to_numpy(),
        pcvr_model=df["p_cvr_pred"].to_numpy(),
    )
    pools = {kw: g.index.to_numpy() for kw, g in df.groupby("keyword_id")}
    min_pool = max(cfg.dgp.eligible_ads_max, cfg.auction.topk_metrics + 2)
    pools = {kw: idx for kw, idx in pools.items() if len(idx) >= min_pool}
    return df, pools


def _run_policy(scores, quality, bid, pctr_true, gain, u,
                decay, n_slots, k, reserve):
    """Rank by `scores`, slot the top n_slots, return realized stats + NDCG@k."""
    order = np.argsort(-scores)
    top = order[:n_slots]
    slots = np.arange(1, len(top) + 1)
    examine = np.power(decay, slots - 1)

    clicked = (u[top] < examine * pctr_true[top]).astype(np.int8)  # common RNs

    next_scores = np.empty(len(top))
    next_scores[:-1] = scores[order[1:n_slots]]
    next_scores[-1] = 0.0
    price = np.maximum(reserve, next_scores / np.maximum(quality[top], 1e-6))
    price = np.minimum(price, bid[top])
    revenue = float(np.sum(price * clicked))

    return int(clicked[:k].sum()), int(min(k, len(top))), revenue, ndcg_at_k(gain[top][:k], gain, k)


def simulate(cfg: Config, rng: np.random.Generator, n_auctions: int = 40000) -> dict:
    decay = float(cfg.dgp.position_decay)
    n_slots = int(cfg.dgp.n_ad_slots)
    k = int(cfg.auction.topk_metrics)
    reserve = float(cfg.auction.reserve_price)

    with timed("auction.build_pools", LOGGER):
        df, pools = _build_candidate_pools(cfg)
    kw_ids = np.array(list(pools.keys()))
    pool_sizes = np.array([len(pools[kw]) for kw in kw_ids], dtype=float)
    kw_p = pool_sizes / pool_sizes.sum()
    LOGGER.info("auction pools: %d eligible keywords (>=%d ads)", len(kw_ids),
                max(cfg.dgp.eligible_ads_max, k + 2))

    bid_all = df["bid"].to_numpy()
    pctr_true_all = df["pctr_true"].to_numpy()
    pctr_model_all = df["pctr_model"].to_numpy()
    pcvr_true_all = df["pcvr_true"].to_numpy()
    pcvr_model_all = df["pcvr_model"].to_numpy()

    agg = {p: {"clicks": 0, "slots": 0, "revenue": 0.0, "ndcg": 0.0}
           for p in POLICIES}

    with timed("auction.simulate", LOGGER):
        chosen_kw = rng.choice(len(kw_ids), size=n_auctions, p=kw_p)
        for a in range(n_auctions):
            pool = pools[kw_ids[chosen_kw[a]]]
            n_elig = min(int(rng.integers(cfg.dgp.eligible_ads_min,
                                          cfg.dgp.eligible_ads_max + 1)), len(pool))
            cand = pool[rng.choice(len(pool), size=n_elig, replace=False)]

            bid = bid_all[cand]
            pctr_true = pctr_true_all[cand]
            pctr_model = pctr_model_all[cand]
            pcvr_true = pcvr_true_all[cand]
            pcvr_model = pcvr_model_all[cand]
            gain = pctr_true * pcvr_true            # true expected value (relevance)
            u = rng.random(n_elig)

            policy_scores = {
                "bid_only": (bid, np.ones_like(bid)),
                "adrank": (bid * pctr_model, pctr_model),
                "ev": (bid * pctr_model * pcvr_model, pctr_model * pcvr_model),
                "ideal": (bid * pctr_true * pcvr_true, pctr_true * pcvr_true),
            }
            for pol, (scores, quality) in policy_scores.items():
                c, s, rev, nd = _run_policy(scores, quality, bid, pctr_true,
                                            gain, u, decay, n_slots, k, reserve)
                agg[pol]["clicks"] += c
                agg[pol]["slots"] += s
                agg[pol]["revenue"] += rev
                agg[pol]["ndcg"] += nd

    out = {}
    for p in POLICIES:
        out[p] = {
            "ctr_at_k": round(agg[p]["clicks"] / max(1, agg[p]["slots"]), 6),
            "ndcg_at_k": round(agg[p]["ndcg"] / n_auctions, 6),
            "revenue_per_auction": round(agg[p]["revenue"] / n_auctions, 6),
        }

    base, ml = out["bid_only"], out[HEADLINE_ML]
    lift = {
        "ndcg_at_k_lift_pct": round(
            100 * (ml["ndcg_at_k"] - base["ndcg_at_k"]) / base["ndcg_at_k"], 3),
        "ctr_at_k_lift_pct": round(
            100 * (ml["ctr_at_k"] - base["ctr_at_k"]) / base["ctr_at_k"], 3),
        "revenue_lift_pct": round(
            100 * (ml["revenue_per_auction"] - base["revenue_per_auction"])
            / base["revenue_per_auction"], 3),
    }
    report = {
        "k": k, "n_auctions": n_auctions, "n_keywords": int(len(kw_ids)),
        "headline_policy": HEADLINE_ML,
        "policies": out, "adrank_vs_bid_only": lift,
    }
    save_json(report, Path(cfg.paths["reports_dir"]) / "auction_metrics.json")
    LOGGER.info("AUCTION lift (%s vs bid_only): NDCG@%d=+%.2f%%  CTR@%d=+%.2f%%  rev=+%.2f%%",
                HEADLINE_ML, k, lift["ndcg_at_k_lift_pct"], k,
                lift["ctr_at_k_lift_pct"], lift["revenue_lift_pct"])
    return report
