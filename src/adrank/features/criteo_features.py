"""Leakage-safe feature engineering for the Criteo CTR dataset.

Same philosophy as ``adrank.features.engineering`` (the synthetic-data builder),
adapted to Criteo's schema:

* **Integer features I1..I13** -> raw, ``log1p`` (counts are heavy-tailed), and a
  missing-value indicator.
* **Categorical features C1..C26** -> leakage-safe **historical smoothed CTR**
  (target encoding) computed on an early history window, plus ``log(count)``.
  This is the standard winning approach on Criteo; one-hot on 32-bit hashed
  categoricals (millions of levels) is infeasible.
* A handful of **cross features** between the highest-signal categoricals.

The output matches what ``adrank.models`` expects (a wide table with the feature
columns + ``clicked`` + ``day``), so the *same* training / calibration / backtest
code runs on it unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.criteo import CAT_COLS, INT_COLS
from ..utils.common import get_logger, timed

LOGGER = get_logger("adrank.criteo.features")

# crosses between a few categoricals (indices chosen to be lower-cardinality)
CROSS_PAIRS = [("C1", "C2"), ("C5", "C6"), ("C8", "C9"), ("C14", "C17")]


def _smoothed(num, den, alpha, prior):
    return (num + alpha * prior) / (den + alpha)


def build_criteo_features(df: pd.DataFrame, history_fraction: float = 0.40,
                          alpha: float = 20.0) -> tuple[pd.DataFrame, list[str]]:
    """Return (wide feature frame with `clicked`+`day`, feature column list)."""
    n_days = int(df["day"].max()) + 1
    hist_cutoff = int(round(n_days * history_fraction))
    history = df[df["day"] < hist_cutoff]
    model_df = df[df["day"] >= hist_cutoff].copy()
    global_ctr = float(history["clicked"].mean())
    LOGGER.info("Criteo split: history days [0,%d), modeling [%d,%d); global_ctr=%.4f",
                hist_cutoff, hist_cutoff, n_days, global_ctr)

    feat = {}

    # ----- integer features -----
    with timed("criteo.int_features", LOGGER):
        for c in INT_COLS:
            v = model_df[c].to_numpy(dtype=float)
            feat[f"{c}_raw"] = np.nan_to_num(v, nan=0.0)
            feat[f"{c}_log"] = np.log1p(np.clip(np.nan_to_num(v, nan=0.0), 0, None))
            feat[f"{c}_isna"] = np.isnan(v).astype(np.int8)

    # ----- categorical target encodings (leakage-safe) -----
    with timed("criteo.cat_target_encoding", LOGGER):
        for c in CAT_COLS:
            agg = (history.groupby(c, observed=True)
                   .agg(_imp=("clicked", "size"), _clk=("clicked", "sum"))
                   .reset_index())
            agg[f"ctr__{c}"] = _smoothed(agg["_clk"], agg["_imp"], alpha, global_ctr)
            agg[f"logcnt__{c}"] = np.log1p(agg["_imp"])
            merged = model_df[[c]].merge(
                agg[[c, f"ctr__{c}", f"logcnt__{c}"]], on=c, how="left")
            feat[f"ctr__{c}"] = merged[f"ctr__{c}"].fillna(global_ctr).to_numpy()
            feat[f"logcnt__{c}"] = merged[f"logcnt__{c}"].fillna(0.0).to_numpy()

    # ----- cross features -----
    with timed("criteo.cross_features", LOGGER):
        for a, b in CROSS_PAIRS:
            key = f"{a}_x_{b}"
            tmp_hist = history.assign(_k=history[a].astype(str) + "|" + history[b].astype(str))
            agg = (tmp_hist.groupby("_k", observed=True)
                   .agg(_imp=("clicked", "size"), _clk=("clicked", "sum")).reset_index())
            agg[f"ctr__{key}"] = _smoothed(agg["_clk"], agg["_imp"], alpha, global_ctr)
            mk = (model_df[a].astype(str) + "|" + model_df[b].astype(str)).rename("_k")
            merged = pd.DataFrame({"_k": mk}).merge(agg[["_k", f"ctr__{key}"]],
                                                    on="_k", how="left")
            feat[f"ctr__{key}"] = merged[f"ctr__{key}"].fillna(global_ctr).to_numpy()

    out = pd.DataFrame(feat)
    feature_cols = list(out.columns)
    out["clicked"] = model_df["clicked"].to_numpy()
    out["day"] = model_df["day"].to_numpy()
    LOGGER.info("Criteo features: %d columns, %s modeling rows",
                len(feature_cols), f"{len(out):,}")
    return out, feature_cols
