"""Criteo dataset adapter.

The Criteo Display Advertising / 1TB Click Logs share one schema:

    col 0      : label (clicked, 0/1)
    cols 1-13  : integer features  I1..I13  (mostly counts; may be missing)
    cols 14-39 : categorical features C1..C26 (32-bit hashed hex strings; may be missing)

tab-separated, no header. This module reads that format into a DataFrame so the
existing CTR pipeline (features -> LR/GBDT -> calibration -> backtest) can run on
real data unchanged.

There is **no conversion label and no auction structure** in Display/1TB, so only
the CTR half of the project applies here (see docs/CRITEO.md). For local testing
without the multi-GB download, ``make_sample`` writes a small file in the *exact*
Criteo wire format with embedded, learnable signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.common import get_logger

LOGGER = get_logger("adrank.criteo")

INT_COLS = [f"I{i}" for i in range(1, 14)]      # 13 integer features
CAT_COLS = [f"C{i}" for i in range(1, 27)]      # 26 categorical features
COLUMNS = ["clicked"] + INT_COLS + CAT_COLS


def load_criteo(path: str | Path, nrows: int | None = None,
                n_day_buckets: int = 10) -> pd.DataFrame:
    """Read a Criteo TSV into a typed DataFrame and assign pseudo-`day` buckets.

    Display data has no explicit timestamp but is roughly chronological, so we
    bucket by row order into `n_day_buckets` 'days' to drive the leakage-safe
    history split and the walk-forward backtest. (For 1TB, pass the real day.)
    """
    df = pd.read_csv(path, sep="\t", header=None, names=COLUMNS, nrows=nrows,
                     dtype={c: "string" for c in CAT_COLS})
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")  # missing -> NaN
    df["clicked"] = df["clicked"].astype("int8")
    # categoricals: empty string -> explicit "__NA__" token
    for c in CAT_COLS:
        df[c] = df[c].fillna("__NA__").replace("", "__NA__")
    n = len(df)
    df["day"] = (np.arange(n) * n_day_buckets // max(1, n)).astype("int16")
    LOGGER.info("loaded Criteo: %s rows, CTR=%.4f, %d day-buckets",
                f"{n:,}", float(df["clicked"].mean()), n_day_buckets)
    return df


def make_sample(path: str | Path, n: int = 200_000, seed: int = 7) -> str:
    """Write a small file in the exact Criteo wire format, with learnable signal.

    Used to validate the adapter end-to-end without downloading real Criteo.
    Roughly mimics Criteo's ~0.25 base CTR and a mix of informative / noisy
    integer and categorical columns.
    """
    rng = np.random.default_rng(seed)
    # categorical cardinalities (a few high-card like real Criteo, most modest)
    cards = rng.integers(8, 600, size=len(CAT_COLS))
    cards[0], cards[3], cards[9] = 4000, 9000, 6000     # a few high-card columns
    # latent CTR effect per (column, value); only ~half the columns carry signal
    cat_effects, cat_weight = [], np.zeros(len(CAT_COLS))
    for j, card in enumerate(cards):
        cat_effects.append(rng.normal(0, 1.0, size=int(card)))
        cat_weight[j] = rng.choice([0.0, 0.0, 0.45, 0.8])  # many columns are noise
    int_weight = rng.choice([0.0, 0.15, 0.35], size=len(INT_COLS))

    logit = np.full(n, 0.0)
    cat_vals = np.empty((n, len(CAT_COLS)), dtype=object)
    for j, card in enumerate(cards):
        v = rng.integers(0, int(card), size=n)
        logit += cat_weight[j] * cat_effects[j][v]
        # encode like Criteo: 8-char hex hash of the value
        cat_vals[:, j] = [f"{(int(x) * 2654435761) & 0xffffffff:08x}" for x in v]

    int_vals = np.empty((n, len(INT_COLS)))
    for j in range(len(INT_COLS)):
        base = rng.poisson(3, size=n).astype(float)
        int_vals[:, j] = base
        logit += int_weight[j] * (np.log1p(base) - 1.0)
    # heavy irreducible noise so a model lands near real-Criteo levels (~0.79 AUC)
    logit += rng.normal(0, 1.6, size=n)

    # calibrate intercept to ~0.25 base CTR (Criteo-like)
    from scipy.special import expit
    from scipy.optimize import brentq
    b = brentq(lambda b: float(np.mean(expit(logit + b))) - 0.25, -8, 8)
    p = expit(logit + b)
    y = (rng.random(n) < p).astype(int)

    # introduce realistic missingness
    miss_int = rng.random(int_vals.shape) < 0.12
    int_str = np.where(miss_int, "", int_vals.astype(int).astype(str))
    miss_cat = rng.random(cat_vals.shape) < 0.05
    cat_vals = np.where(miss_cat, "", cat_vals)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        np.column_stack([y.astype(str), int_str, cat_vals]), columns=COLUMNS)
    frame.to_csv(out, sep="\t", header=False, index=False)
    LOGGER.info("wrote Criteo-format sample: %s rows, CTR=%.4f -> %s",
                f"{n:,}", float(y.mean()), out)
    return str(out)
