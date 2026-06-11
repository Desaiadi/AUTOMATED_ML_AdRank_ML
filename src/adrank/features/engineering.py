"""Feature engineering for CTR/CVR models (pandas reference implementation).

The production pipeline runs the equivalent logic in PySpark on Databricks
(see ``spark_features.py``) and the aggregate SQL lives in ``sql/``. This pandas
version is the single-node reference used for the local end-to-end run and tests.

Two ideas dominate the design:

1. **Leakage-safe historical aggregates.** The earliest ``features.history_fraction``
   of days is reserved purely to estimate per-entity click/convert propensities.
   Those Bayesian-smoothed rates are then joined onto the *later* modeling days.
   A model therefore only ever sees "what we knew before this impression".

2. **Exactly 145 features.** Features are assembled in named groups; the builder
   asserts the final count equals ``features.target_feature_count`` (145),
   topping up with curated second-order interactions if needed so the contract
   is exact and reproducible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..utils.common import get_logger, timed

LOGGER = get_logger("adrank.features")

# Columns the model is allowed to learn from come from REQUEST_TIME only; oracle
# columns (p_click_true, relevance_logit, ...) are explicitly excluded.
_ONEHOT_COLS = {
    "device": ["mobile", "desktop", "tablet"],
    "user_segment": ["bargain", "mainstream", "premium", "researcher"],
    "match_type": ["exact", "phrase", "broad"],
    "vertical": ["retail", "travel", "finance", "auto", "tech",
                 "health", "home", "apparel", "food", "education"],
}

# Single-key historical CTR aggregates.
_CTR_KEYS = [
    "keyword_id", "campaign_id", "ad_id", "advertiser_id", "user_id",
    "vertical", "keyword_category", "device", "match_type", "user_segment",
    "position", "hour", "dow",
]
# Single-key historical CVR (conv|click) aggregates.
_CVR_KEYS = ["keyword_id", "campaign_id", "user_id", "vertical",
             "keyword_category", "advertiser_id"]
# Cross-key historical CTR aggregates.
_CROSS_KEYS = [
    ("keyword_id", "device"), ("keyword_id", "position"),
    ("campaign_id", "device"), ("campaign_id", "position"),
    ("user_id", "vertical"), ("vertical", "device"),
    ("vertical", "position"), ("keyword_category", "device"),
    ("user_segment", "device"), ("match_type", "position"),
    ("keyword_category", "position"), ("campaign_id", "hour"),
    ("user_segment", "vertical"), ("device", "hour"),
    ("advertiser_id", "device"),
]


def _load_log(cfg: Config) -> pd.DataFrame:
    imp_dir = Path(cfg.paths["raw_dir"]) / "impressions"
    parts = sorted(imp_dir.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No impression parts in {imp_dir}; run data generation first.")
    df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    # categoricals -> str for robust groupby/merge across the pipeline
    for c in ["user_segment", "device", "keyword_category", "match_type", "vertical"]:
        df[c] = df[c].astype(str)
    return df


def _smoothed_ctr(g: pd.DataFrame, num: str, den: str, alpha: float, prior: float) -> pd.Series:
    return (g[num] + alpha * prior) / (g[den] + alpha)


def _agg_single(history: pd.DataFrame, key: str, alpha: float,
                global_ctr: float) -> pd.DataFrame:
    g = history.groupby(key, observed=True).agg(
        _imp=("clicked", "size"), _clk=("clicked", "sum")).reset_index()
    out = pd.DataFrame({key: g[key]})
    out[f"ctr__{key}"] = _smoothed_ctr(g, "_clk", "_imp", alpha, global_ctr)
    out[f"logimp__{key}"] = np.log1p(g["_imp"])
    return out


def _agg_cvr(history: pd.DataFrame, key: str, alpha: float,
             global_cvr: float) -> pd.DataFrame:
    clicked = history[history["clicked"] == 1]
    g = clicked.groupby(key, observed=True).agg(
        _clk=("converted", "size"), _cnv=("converted", "sum")).reset_index()
    out = pd.DataFrame({key: g[key]})
    out[f"cvr__{key}"] = _smoothed_ctr(g, "_cnv", "_clk", alpha, global_cvr)
    return out


def _agg_cross(history: pd.DataFrame, k1: str, k2: str, alpha: float,
               global_ctr: float) -> pd.DataFrame:
    g = history.groupby([k1, k2], observed=True).agg(
        _imp=("clicked", "size"), _clk=("clicked", "sum")).reset_index()
    name = f"{k1}_X_{k2}"
    out = g[[k1, k2]].copy()
    out[f"ctr__{name}"] = _smoothed_ctr(g, "_clk", "_imp", alpha, global_ctr)
    out[f"logimp__{name}"] = np.log1p(g["_imp"])
    return out


def build_features(cfg: Config) -> dict:
    """Build the wide feature table and persist it to processed parquet."""
    alpha = float(cfg.features.smoothing_alpha)
    decay = float(cfg.dgp.position_decay)

    with timed("features.load_log", LOGGER):
        df = _load_log(cfg)

    # ---- time-based history / modeling split (no leakage) ----
    n_days = int(df["day"].max()) + 1
    hist_cutoff = int(round(n_days * float(cfg.features.history_fraction)))
    history = df[df["day"] < hist_cutoff]
    model_df = df[df["day"] >= hist_cutoff].copy()
    global_ctr = float(history["clicked"].mean())
    clicked_hist = history[history["clicked"] == 1]
    global_cvr = float(clicked_hist["converted"].mean()) if len(clicked_hist) else cfg.dgp.cvr_base_rate
    LOGGER.info("history days [0,%d) -> modeling days [%d,%d); global_ctr=%.4f global_cvr=%.4f",
                hist_cutoff, hist_cutoff, n_days, global_ctr, global_cvr)

    feat = pd.DataFrame(index=model_df.index)

    # ===== Group A: request-time numerics & transforms =====
    with timed("features.group_numeric", LOGGER):
        pos = model_df["position"].to_numpy()
        feat["position"] = pos
        feat["inv_position"] = 1.0 / pos
        feat["log_position"] = np.log1p(pos)
        feat["examine_prior"] = np.power(decay, pos - 1)
        feat["is_top1"] = (pos == 1).astype(np.int8)
        feat["is_top3"] = (pos <= 3).astype(np.int8)
        feat["n_eligible"] = model_df["n_eligible"].to_numpy()
        feat["log_n_eligible"] = np.log1p(model_df["n_eligible"].to_numpy())
        feat["keyword_n_tokens"] = model_df["keyword_n_tokens"].to_numpy()
        feat["bid"] = model_df["bid"].to_numpy()
        feat["log_bid"] = np.log1p(model_df["bid"].to_numpy())
        hour = model_df["hour"].to_numpy()
        feat["hour"] = hour
        feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        dow = model_df["dow"].to_numpy()
        feat["dow"] = dow
        feat["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        feat["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        feat["is_weekend"] = model_df["is_weekend"].to_numpy()

    # ===== Group B: one-hot categoricals =====
    with timed("features.group_onehot", LOGGER):
        for col, levels in _ONEHOT_COLS.items():
            vals = model_df[col].to_numpy()
            for lv in levels:
                feat[f"{col}__{lv}"] = (vals == lv).astype(np.int8)

    # ===== Group C: single-key historical CTR =====
    with timed("features.group_ctr_single", LOGGER):
        for key in _CTR_KEYS:
            agg = _agg_single(history, key, alpha, global_ctr)
            merged = model_df[[key]].merge(agg, on=key, how="left")
            feat[f"ctr__{key}"] = merged[f"ctr__{key}"].fillna(global_ctr).to_numpy()
            feat[f"logimp__{key}"] = merged[f"logimp__{key}"].fillna(0.0).to_numpy()

    # ===== Group D: single-key historical CVR =====
    with timed("features.group_cvr_single", LOGGER):
        for key in _CVR_KEYS:
            agg = _agg_cvr(history, key, alpha, global_cvr)
            merged = model_df[[key]].merge(agg, on=key, how="left")
            feat[f"cvr__{key}"] = merged[f"cvr__{key}"].fillna(global_cvr).to_numpy()

    # ===== Group E: cross-key historical CTR =====
    with timed("features.group_ctr_cross", LOGGER):
        for k1, k2 in _CROSS_KEYS:
            agg = _agg_cross(history, k1, k2, alpha, global_ctr)
            name = f"{k1}_X_{k2}"
            merged = model_df[[k1, k2]].merge(agg, on=[k1, k2], how="left")
            feat[f"ctr__{name}"] = merged[f"ctr__{name}"].fillna(global_ctr).to_numpy()
            feat[f"logimp__{name}"] = merged[f"logimp__{name}"].fillna(0.0).to_numpy()

    # ===== Group F: derived interaction / economic signals =====
    feat = feat.copy()  # de-fragment before a burst of column inserts
    with timed("features.group_derived", LOGGER):
        feat["ad_rank"] = feat["bid"] * np.exp(feat["ctr__campaign_id"])     # bid x quality
        feat["bid_x_kw_ctr"] = feat["log_bid"] * feat["ctr__keyword_id"]
        feat["pos_x_kw_ctr"] = feat["examine_prior"] * feat["ctr__keyword_id"]
        feat["kw_ctr_dev"] = feat["ctr__keyword_id"] - global_ctr
        feat["camp_ctr_dev"] = feat["ctr__campaign_id"] - global_ctr
        feat["user_ctr_dev"] = feat["ctr__user_id"] - global_ctr
        feat["kw_camp_ctr"] = feat["ctr__keyword_id"] * feat["ctr__campaign_id"]
        feat["expected_value"] = feat["ad_rank"] * feat["cvr__keyword_id"]   # bid*pCTR*pCVR proxy
        # vertical match between keyword & shown campaign (relevance proxy)
        feat["vertical_match"] = (
            model_df["vertical"].to_numpy() ==
            model_df["keyword_id"].map(
                history.groupby("keyword_id", observed=True)["vertical"].agg(
                    lambda s: s.mode().iat[0] if len(s) else "retail")
            ).fillna("retail").to_numpy()
        ).astype(np.int8)

    # ---- enforce exactly target_feature_count features ----
    target = int(cfg.features.target_feature_count)
    feat = _reconcile_feature_count(feat, target)
    feature_cols = list(feat.columns)
    assert len(feature_cols) == target, (len(feature_cols), target)

    # attach labels + keys needed downstream (auction/backtest). `position`,
    # `hour`, `bid` are intentionally omitted here — they already exist as
    # feature columns with identical values, so we avoid duplicate names.
    keep_meta = ["impression_id", "ts", "day", "user_id", "keyword_id",
                 "campaign_id", "ad_id", "quality_true", "relevance_logit",
                 "p_click_true", "p_conv_true", "clicked", "converted"]
    out = pd.concat([feat.reset_index(drop=True),
                     model_df[keep_meta].reset_index(drop=True)], axis=1).copy()

    proc_dir = Path(cfg.paths["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    out_path = proc_dir / "features.parquet"
    with timed("features.write", LOGGER):
        out.to_parquet(out_path, index=False)
    # persist the canonical feature list for the model + scoring stages
    (proc_dir / "feature_list.txt").write_text("\n".join(feature_cols))

    LOGGER.info("FEATURES: %d columns, %s modeling rows -> %s",
                len(feature_cols), f"{len(out):,}", out_path)
    return {
        "n_features": len(feature_cols),
        "n_rows": int(len(out)),
        "history_days": hist_cutoff,
        "modeling_days": n_days - hist_cutoff,
        "global_ctr": round(global_ctr, 5),
        "global_cvr": round(global_cvr, 5),
        "features_path": str(out_path),
    }


def _reconcile_feature_count(feat: pd.DataFrame, target: int) -> pd.DataFrame:
    """Top up (with curated interactions) or trim to hit exactly `target` cols."""
    n = feat.shape[1]
    if n == target:
        return feat
    if n > target:
        LOGGER.warning("trimming %d -> %d features (dropping interaction tail)", n, target)
        return feat.iloc[:, :target]

    # Need more: add deterministic pairwise products of the strongest signals.
    core = [c for c in [
        "ctr__keyword_id", "ctr__campaign_id", "ctr__user_id", "ctr__position",
        "ctr__vertical", "examine_prior", "ad_rank", "cvr__keyword_id",
        "ctr__device", "ctr__match_type", "ctr__keyword_id_X_device"
        if "ctr__keyword_id_X_device" in feat.columns else "ctr__keyword_id",
    ] if c in feat.columns]
    pairs = [(a, b) for i, a in enumerate(core) for b in core[i + 1:]]
    idx = 0
    while feat.shape[1] < target and idx < len(pairs):
        a, b = pairs[idx]
        feat[f"intx__{a}__{b}"] = feat[a].to_numpy() * feat[b].to_numpy()
        idx += 1
    if feat.shape[1] < target:  # pragma: no cover - safety pad
        for j in range(target - feat.shape[1]):
            feat[f"pad__{j}"] = 0.0
    LOGGER.info("padded features to %d via %d engineered interactions", target, idx)
    return feat
