"""Synthetic search-ads auction log generator.

Produces a realistic impression-level log:
    120 campaigns x 18K keywords x N users over D days, up to 100M impressions.

Design goals
------------
1. **Learnable but noisy labels.** Clicks follow the examination hypothesis
       P(click) = P(examine | position) * sigmoid(relevance_logit)
   where ``relevance_logit`` is a sum of *latent* per-entity propensities
   (keyword, ad/campaign, user, context) plus per-impression Gaussian noise.
   The latent propensities are only partially recoverable from features (via
   smoothed historical CTR aggregates), and the Gaussian noise is irreducible.
   That gap is what caps any model near the target AUC ~ 0.79 / logloss ~ 0.43.

2. **Calibrated base rates.** Intercepts are solved on a pilot sample so the
   marginal CTR and CVR match ``dgp.ctr_base_rate`` / ``dgp.cvr_base_rate``.

3. **Scales to 100M.** Generation is chunked; each chunk is written as a Parquet
   part file under ``data/raw/impressions/`` so peak memory stays bounded.

The dimension tables (campaigns, keywords, users) are written alongside so the
auction simulator and feature pipelines can join back to entity attributes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit  # numerically stable sigmoid

from ..config import Config
from ..utils.common import get_logger, timed

LOGGER = get_logger("adrank.data.generate")

VERTICALS = [
    "retail", "travel", "finance", "auto", "tech",
    "health", "home", "apparel", "food", "education",
]
SEGMENTS = ["bargain", "mainstream", "premium", "researcher"]
DEVICES = ["mobile", "desktop", "tablet"]
MATCH_TYPES = ["exact", "phrase", "broad"]
KEYWORD_CATEGORIES = [f"cat_{i:02d}" for i in range(24)]


# --------------------------------------------------------------------------- #
# Dimension tables
# --------------------------------------------------------------------------- #
@dataclass
class Dimensions:
    campaigns: pd.DataFrame
    keywords: pd.DataFrame
    users: pd.DataFrame
    # arrays for fast vectorised lookup during fact generation
    camp_quality: np.ndarray
    camp_conv_q: np.ndarray
    camp_bid: np.ndarray
    camp_vertical: np.ndarray
    kw_ctr_latent: np.ndarray
    kw_intent: np.ndarray
    kw_vertical: np.ndarray
    kw_compet: np.ndarray
    kw_ntokens: np.ndarray
    kw_category: np.ndarray
    kw_popularity: np.ndarray
    user_ctr_latent: np.ndarray
    user_conv_latent: np.ndarray
    user_segment: np.ndarray
    user_device_pref: np.ndarray


def _build_dimensions(cfg: Config, rng: np.random.Generator) -> Dimensions:
    n_camp = cfg.scale.n_campaigns
    n_kw = cfg.scale.n_keywords
    n_users = cfg.scale.n_users

    # --- campaigns ---
    camp_vertical = rng.integers(0, len(VERTICALS), size=n_camp)
    camp_quality = rng.normal(0.0, 1.0, size=n_camp)            # latent ad quality (z)
    camp_conv_q = rng.normal(0.0, 1.0, size=n_camp)             # latent landing/conv quality
    camp_bid = np.round(np.exp(rng.normal(0.4, 0.45, size=n_camp)), 3)  # ~ lognormal CPC bids
    campaigns = pd.DataFrame({
        "campaign_id": np.arange(n_camp),
        "advertiser_id": rng.integers(0, max(1, n_camp // 3), size=n_camp),
        "vertical": [VERTICALS[v] for v in camp_vertical],
        "quality_base": camp_quality.round(4),
        "conv_quality_base": camp_conv_q.round(4),
        "bid_base": camp_bid,
        "daily_budget": np.round(np.exp(rng.normal(6.0, 0.7, size=n_camp)), 2),
    })

    # --- keywords ---
    kw_vertical = rng.integers(0, len(VERTICALS), size=n_kw)
    kw_ctr_latent = rng.normal(0.0, 1.0, size=n_kw)            # latent keyword clickiness
    kw_intent = rng.normal(0.0, 1.0, size=n_kw)               # commercial intent -> conv
    kw_compet = np.clip(rng.normal(1.0, 0.3, size=n_kw), 0.3, None)  # bid multiplier
    kw_ntokens = rng.integers(1, 6, size=n_kw)
    kw_category = rng.integers(0, len(KEYWORD_CATEGORIES), size=n_kw)
    # Zipf-like popularity: a few head keywords dominate impressions.
    pop = 1.0 / np.power(np.arange(1, n_kw + 1), 0.9)
    rng.shuffle(pop)
    kw_popularity = pop / pop.sum()
    keywords = pd.DataFrame({
        "keyword_id": np.arange(n_kw),
        "keyword_category": [KEYWORD_CATEGORIES[c] for c in kw_category],
        "vertical": [VERTICALS[v] for v in kw_vertical],
        "n_tokens": kw_ntokens,
        "commercial_intent": kw_intent.round(4),
        "competitiveness": kw_compet.round(4),
        "ctr_latent": kw_ctr_latent.round(4),
        "popularity": kw_popularity,
    })

    # --- users ---
    user_segment = rng.integers(0, len(SEGMENTS), size=n_users)
    user_ctr_latent = rng.normal(0.0, 1.0, size=n_users)
    user_conv_latent = rng.normal(0.0, 1.0, size=n_users)
    user_device_pref = rng.integers(0, len(DEVICES), size=n_users)
    users = pd.DataFrame({
        "user_id": np.arange(n_users),
        "user_segment": [SEGMENTS[s] for s in user_segment],
        "ctr_latent": user_ctr_latent.round(4),
        "conv_latent": user_conv_latent.round(4),
        "device_pref": [DEVICES[d] for d in user_device_pref],
    })

    return Dimensions(
        campaigns=campaigns, keywords=keywords, users=users,
        camp_quality=camp_quality, camp_conv_q=camp_conv_q,
        camp_bid=camp_bid, camp_vertical=camp_vertical,
        kw_ctr_latent=kw_ctr_latent, kw_intent=kw_intent,
        kw_vertical=kw_vertical, kw_compet=kw_compet,
        kw_ntokens=kw_ntokens, kw_category=kw_category, kw_popularity=kw_popularity,
        user_ctr_latent=user_ctr_latent, user_conv_latent=user_conv_latent,
        user_segment=user_segment, user_device_pref=user_device_pref,
    )


# --------------------------------------------------------------------------- #
# Context effects (deterministic given dims)
# --------------------------------------------------------------------------- #
def _context_effects(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "hour": rng.normal(0, 0.35, size=24),          # diurnal clickiness
        "dow": rng.normal(0, 0.20, size=7),
        "device": np.array([0.15, -0.05, -0.10]),      # mobile clicks a bit more
        "segment": np.array([0.20, 0.0, -0.10, -0.25]),
        "match": np.array([0.30, 0.05, -0.20]),        # exact > phrase > broad
    }


def _hour_weights() -> np.ndarray:
    # Diurnal impression volume: low overnight, peaks midday and evening.
    h = np.arange(24)
    w = (
        0.4
        + 0.6 * np.exp(-((h - 13) ** 2) / 18.0)
        + 0.5 * np.exp(-((h - 20) ** 2) / 10.0)
        + 0.1
    )
    return w / w.sum()


# --------------------------------------------------------------------------- #
# Core: build one chunk of impressions (vectorised)
# --------------------------------------------------------------------------- #
def _sample_chunk(
    n: int,
    dims: Dimensions,
    ctx: dict[str, np.ndarray],
    cfg: Config,
    rng: np.random.Generator,
    start_id: int,
    ctr_intercept: float,
    cvr_intercept: float,
) -> pd.DataFrame:
    sc = cfg.scale
    dgp = cfg.dgp

    # --- sample context / time ---
    day = rng.integers(0, sc.n_days, size=n)
    dow = day % 7
    is_weekend = (dow >= 5).astype(np.int8)
    hour = rng.choice(24, size=n, p=_hour_weights())

    # --- sample keyword by popularity, user uniformly ---
    keyword_id = rng.choice(len(dims.kw_popularity), size=n, p=dims.kw_popularity)
    user_id = rng.integers(0, sc.n_users, size=n)

    # --- choose a campaign for this keyword, biased by vertical match ---
    # cheap relevance proxy: campaigns whose vertical == keyword vertical are
    # far more likely to be the shown ad.
    kw_vert = dims.kw_vertical[keyword_id]
    rand_camp = rng.integers(0, sc.n_campaigns, size=n)
    same_vert_mask = rng.random(n) < 0.78
    # for "same vertical" impressions, redraw campaign until vertical matches via
    # a precomputed per-vertical campaign index
    campaign_id = rand_camp.copy()
    by_vert = [np.where(dims.camp_vertical == v)[0] for v in range(len(VERTICALS))]
    need = np.where(same_vert_mask)[0]
    for v in range(len(VERTICALS)):
        sel = need[kw_vert[need] == v]
        pool = by_vert[v]
        if len(sel) and len(pool):
            campaign_id[sel] = pool[rng.integers(0, len(pool), size=len(sel))]
    vertical_match = (dims.camp_vertical[campaign_id] == kw_vert).astype(np.float64)

    # one creative per campaign for simplicity, id namespaced by campaign
    ad_id = campaign_id  # 1 ad/campaign in this DGP; kept as its own column

    # --- device: usually the user's preferred device ---
    device_idx = dims.user_device_pref[user_id].copy()
    flip = rng.random(n) < 0.25
    device_idx[flip] = rng.integers(0, len(DEVICES), size=flip.sum())

    match_idx = rng.choice(len(MATCH_TYPES), size=n, p=[0.45, 0.35, 0.20])

    # --- position: skew to the top of the page (geometric-ish over slots) ---
    n_slots = dgp.n_ad_slots
    slot_p = np.power(0.62, np.arange(n_slots))
    slot_p = slot_p / slot_p.sum()
    position = rng.choice(np.arange(1, n_slots + 1), size=n, p=slot_p)

    n_eligible = rng.integers(dgp.eligible_ads_min, dgp.eligible_ads_max + 1, size=n)

    seg_idx = dims.user_segment[user_id]

    # --- relevance logit (systematic, pre-position) ---
    s = dgp.ctr_signal_scale
    relevance_logit = (
        ctr_intercept
        + s * 0.85 * dims.kw_ctr_latent[keyword_id]
        + s * 0.70 * dims.camp_quality[campaign_id]
        + s * 0.55 * dims.user_ctr_latent[user_id]
        + s * 0.90 * vertical_match
        + s * ctx["device"][device_idx]
        + s * ctx["match"][match_idx]
        + s * ctx["segment"][seg_idx]
        + s * ctx["hour"][hour]
        + s * ctx["dow"][dow]
        # an interaction the GBDT can exploit but LR can only approximate:
        + s * 0.45 * dims.kw_ctr_latent[keyword_id] * (device_idx == 0)
        # slow day-to-day demand drift, NOT exposed as a feature (only dow/hour
        # are). The model cannot track it, so recent-day predictions drift out
        # of calibration -- which is exactly what the calibration step repairs.
        + ctx["day_drift"][day]
    )
    relevance_logit = relevance_logit + rng.normal(0, dgp.ctr_noise_sd, size=n)

    p_examine = np.power(dgp.position_decay, position - 1)
    p_click = p_examine * expit(relevance_logit)
    clicked = (rng.random(n) < p_click).astype(np.int8)

    # --- conversion (only meaningful where clicked) ---
    sv = dgp.cvr_signal_scale
    conv_logit = (
        cvr_intercept
        + sv * 0.95 * dims.kw_intent[keyword_id]
        + sv * 0.80 * dims.camp_conv_q[campaign_id]
        + sv * 0.60 * dims.user_conv_latent[user_id]
        + sv * 0.30 * vertical_match
        + rng.normal(0, dgp.cvr_noise_sd, size=n)
    )
    p_conv = expit(conv_logit)
    converted = np.where(clicked == 1, (rng.random(n) < p_conv).astype(np.int8), 0).astype(np.int8)

    # --- economics ---
    # Advertisers bid more on clicky / converting inventory, so bids are coupled
    # to latent quality (mean-zero so the overall bid scale is preserved). This
    # makes the bid-only baseline a *reasonable* ranker -- without it, quality-
    # aware ranking would look unrealistically good.
    coupling = float(dgp.get("bid_quality_coupling", 0.0))
    quality_z = (
        0.5 * dims.camp_quality[campaign_id]
        + 0.4 * dims.kw_ctr_latent[keyword_id]
        + 0.3 * dims.camp_conv_q[campaign_id]
        + 0.3 * dims.kw_intent[keyword_id]
    )
    bid = (
        dims.camp_bid[campaign_id]
        * dims.kw_compet[keyword_id]
        * np.exp(coupling * quality_z + rng.normal(0, 0.12, size=n))
    ).round(4)
    quality_true = dims.camp_quality[campaign_id] + 0.5 * vertical_match

    impression_id = np.arange(start_id, start_id + n, dtype=np.int64)
    # synthetic timestamp: base epoch + day + hour (seconds)
    ts = (day.astype(np.int64) * 86400 + hour.astype(np.int64) * 3600
          + rng.integers(0, 3600, size=n))

    df = pd.DataFrame({
        "impression_id": impression_id,
        "ts": ts,
        "day": day.astype(np.int16),
        "hour": hour.astype(np.int8),
        "dow": dow.astype(np.int8),
        "is_weekend": is_weekend,
        "user_id": user_id.astype(np.int64),
        "user_segment": pd.Categorical.from_codes(seg_idx, SEGMENTS),
        "device": pd.Categorical.from_codes(device_idx, DEVICES),
        "keyword_id": keyword_id.astype(np.int32),
        "keyword_category": pd.Categorical.from_codes(
            dims.kw_category[keyword_id], KEYWORD_CATEGORIES),
        "keyword_n_tokens": dims.kw_ntokens[keyword_id].astype(np.int8),
        "match_type": pd.Categorical.from_codes(match_idx, MATCH_TYPES),
        "campaign_id": campaign_id.astype(np.int16),
        "advertiser_id": (campaign_id % max(1, sc.n_campaigns // 3)).astype(np.int16),
        "vertical": pd.Categorical.from_codes(dims.camp_vertical[campaign_id], VERTICALS),
        "ad_id": ad_id.astype(np.int16),
        "position": position.astype(np.int8),
        "n_eligible": n_eligible.astype(np.int8),
        "bid": bid,
        "quality_true": quality_true.round(4),
        "relevance_logit": relevance_logit.round(4),
        "p_click_true": p_click.round(5),
        # intrinsic conversion propensity for every row (used by the auction's
        # value ranking + NDCG gain). The CVR oracle eval still filters to clicks.
        "p_conv_true": p_conv.round(5),
        "clicked": clicked,
        "converted": converted,
    })
    return df


# --------------------------------------------------------------------------- #
# Intercept calibration so marginal rates match the configured base rates
# --------------------------------------------------------------------------- #
def _calibrate_intercepts(
    dims: Dimensions, ctx: dict, cfg: Config, rng: np.random.Generator
) -> tuple[float, float]:
    """Solve CTR/CVR intercepts on a pilot sample to hit target base rates."""
    pilot_n = min(200_000, max(50_000, cfg.scale.impressions // 20))
    pilot = _sample_chunk(
        pilot_n, dims, ctx, cfg, rng, start_id=0,
        ctr_intercept=0.0, cvr_intercept=0.0,
    )
    # Recover the pre-intercept logits from the pilot (intercept was 0 here).
    base_rel = pilot["relevance_logit"].to_numpy()
    pexam = np.power(cfg.dgp.position_decay, pilot["position"].to_numpy() - 1)

    def ctr_gap(b: float) -> float:
        return float(np.mean(pexam * expit(base_rel + b)) - cfg.dgp.ctr_base_rate)

    ctr_b = brentq(ctr_gap, -8.0, 8.0, xtol=1e-4)

    # CVR intercept targets P(conv | click); recompute conv logits' base
    clicked = pilot["clicked"].to_numpy() == 1
    if clicked.sum() < 100:  # pragma: no cover - tiny pilots
        return ctr_b, 0.0
    # Reconstruct conv base (intercept was 0). We approximate using stored p_conv_true
    # which already folds the zero intercept; invert to logit then shift.
    pconv = pilot.loc[clicked, "p_conv_true"].to_numpy()
    pconv = np.clip(pconv, 1e-6, 1 - 1e-6)
    base_conv_logit = np.log(pconv / (1 - pconv))

    def cvr_gap(b: float) -> float:
        return float(np.mean(expit(base_conv_logit + b)) - cfg.dgp.cvr_base_rate)

    cvr_b = brentq(cvr_gap, -8.0, 8.0, xtol=1e-4)
    LOGGER.info("calibrated intercepts: ctr_b=%.4f  cvr_b=%.4f", ctr_b, cvr_b)
    return ctr_b, cvr_b


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate(cfg: Config, rng: np.random.Generator, chunk_size: int = 1_000_000) -> dict:
    """Generate the full synthetic log and write Parquet parts + dim tables.

    Returns a small summary dict (row counts, realized rates, paths).
    """
    raw_dir = Path(cfg.paths["raw_dir"])
    imp_dir = raw_dir / "impressions"
    imp_dir.mkdir(parents=True, exist_ok=True)
    # clear any previous run's parts
    for old in imp_dir.glob("part-*.parquet"):
        old.unlink()

    with timed("data.build_dimensions", LOGGER):
        dims = _build_dimensions(cfg, rng)
        ctx = _context_effects(rng)
        # slow demand drift over the horizon (seasonal wave + mild trend),
        # centered so it does not move the marginal base rate.
        n_days = int(cfg.scale.n_days)
        d = np.arange(n_days)
        amp = float(cfg.dgp.get("ctr_day_drift", 0.0))
        drift = amp * (np.sin(2 * np.pi * d / max(1, n_days) * 1.5)
                       + 0.4 * (d / max(1, n_days) - 0.5))
        ctx["day_drift"] = drift - drift.mean()
        dims.campaigns.to_parquet(raw_dir / "campaigns.parquet", index=False)
        dims.keywords.drop(columns=["popularity"]).to_parquet(
            raw_dir / "keywords.parquet", index=False)
        dims.users.to_parquet(raw_dir / "users.parquet", index=False)

    with timed("data.calibrate_intercepts", LOGGER):
        ctr_b, cvr_b = _calibrate_intercepts(dims, ctx, cfg, rng)

    total = int(cfg.scale.impressions)
    written, clicks, convs = 0, 0, 0
    part = 0
    with timed("data.generate_impressions", LOGGER):
        while written < total:
            n = min(chunk_size, total - written)
            df = _sample_chunk(
                n, dims, ctx, cfg, rng, start_id=written,
                ctr_intercept=ctr_b, cvr_intercept=cvr_b,
            )
            df.to_parquet(imp_dir / f"part-{part:05d}.parquet", index=False)
            written += n
            clicks += int(df["clicked"].sum())
            convs += int(df["converted"].sum())
            part += 1
            if part % 10 == 0 or written == total:
                LOGGER.info("  generated %s / %s impressions (%d parts)",
                            f"{written:,}", f"{total:,}", part)

    ctr = clicks / written
    cvr = convs / clicks if clicks else 0.0
    summary = {
        "impressions": written,
        "clicks": clicks,
        "conversions": convs,
        "ctr": round(ctr, 5),
        "cvr_given_click": round(cvr, 5),
        "n_campaigns": cfg.scale.n_campaigns,
        "n_keywords": cfg.scale.n_keywords,
        "n_users": cfg.scale.n_users,
        "n_days": cfg.scale.n_days,
        "n_parts": part,
        "ctr_intercept": round(ctr_b, 5),
        "cvr_intercept": round(cvr_b, 5),
        "impressions_dir": str(imp_dir),
    }
    LOGGER.info("DATA SUMMARY: CTR=%.4f  CVR|click=%.4f  rows=%s",
                ctr, cvr, f"{written:,}")
    return summary
