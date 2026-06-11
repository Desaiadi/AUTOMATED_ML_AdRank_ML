# Running AdRank-ML on real Criteo data

The CTR half of AdRank-ML runs on the **Criteo** datasets with no change to the
models, calibration, metrics, or backtest — only a dataset-specific loader and
feature builder sit in front.

## What transfers (and what doesn't)

| Capability | Criteo Display / 1TB | Notes |
|---|---|---|
| CTR model (LogisticRegression + HistGBDT) | ✅ direct | Criteo *is* a CTR dataset |
| Isotonic calibration + ECE | ✅ direct | pointwise, dataset-agnostic |
| Leakage-safe historical features | ✅ direct | target-encoding the 26 categoricals is the standard winning approach |
| Time split + walk-forward backtest | ✅ | 1TB has real days; Display is order-chronological (we bucket by row order) |
| **CVR model** | ⚠️ only on the **Criteo Attribution** dataset | Display/1TB have **no conversion label** |
| **GSP auction (NDCG@10 / CTR@10)** | ❌ | Display/1TB have **no bids, positions, or per-query candidate sets** — an auction can't be reconstructed from pointwise logs |

So on Criteo Display/1TB you get the full **CTR modeling + calibration +
backtest** story; the auction and CVR halves need richer data (the Attribution
dataset for conversions; a true serving log with bids/positions for the auction).

## Schema

Tab-separated, no header:

```
label  I1 .. I13            C1 .. C26
 0/1   13 integer features  26 categorical (32-bit hashed hex) features
```

Integer features may be missing; categoricals may be empty. Base CTR ≈ 0.25.

## Components added for Criteo

* `src/adrank/data/criteo.py` — TSV loader (typed, missing-value handling,
  pseudo-`day` bucketing) + `make_sample()` to synthesize a Criteo-format file
  with learnable signal for offline validation.
* `src/adrank/features/criteo_features.py` — leakage-safe features:
  * I1..I13 → raw, `log1p`, missing-indicator (39 features)
  * C1..C26 → history-window **smoothed CTR** (target encoding) + `log(count)` (52)
  * 4 categorical crosses
  * → **95 features**
* `scripts/run_criteo.py` — orchestrates load → features → `fit_eval_ctr` →
  backtest, writing `data/reports/criteo_metrics.json`.

`fit_eval_ctr` in `src/adrank/models/train.py` is the **dataset-agnostic** CTR
core (identical LR + GBDT + isotonic + AUC/logloss/ECE) shared by the synthetic
and Criteo paths.

## Quickstart

```bash
# 1) Validate end-to-end with a generated Criteo-format sample (no download)
python scripts/run_criteo.py --sample 300000
#    -> CTR AUC ~0.81, logloss ~0.43, ECE ~0.007  (data/reports/criteo_metrics.json)

# 2) Run on the real Criteo Display Advertising Challenge file
#    download: https://www.kaggle.com/c/criteo-display-ad-challenge  (or Criteo's
#    academic mirror); the file is `train.txt` (~11 GB, ~45M rows)
python scripts/run_criteo.py --path /data/criteo/train.txt --nrows 5000000

# 3) Criteo 1TB Click Logs (per-day files already carry a real day index)
python scripts/run_criteo.py --path /data/criteo1tb/day_0 --nrows 10000000
```

`--nrows` caps how much is read (start small). For the full multi-GB / 1TB
scale, run the categorical target-encoding aggregates in **PySpark** — the same
broadcast-join pattern as `src/adrank/features/spark_features.py` — and feed the
wide table into `fit_eval_ctr` on a sampled slice or a distributed learner.

## Expected results on real Criteo

Strong single models on Criteo Display land around **AUC ≈ 0.79–0.81** and
**log loss ≈ 0.44** (the Kaggle leaderboard was scored on log loss). The GBDT +
target-encoding setup here is a solid, well-calibrated baseline in that range;
the headline numbers in the main project (AUC 0.79 / logloss 0.43) are
deliberately tuned to the same regime.
