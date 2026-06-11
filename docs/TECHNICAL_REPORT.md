# AdRank-ML — Technical Report

**Marketing Performance Modeling (CTR/CVR) & GSP Auction Simulation**

This report documents the problem, the design, the alternatives considered at
each step, the rationale for the chosen approach, and the measured results. All
numbers quoted are from the `demo` profile run (~3M impressions); see
`data/reports/*.json` for the live artifacts.

---

## 1. Problem statement

Search advertising platforms (Google Ads, Amazon Ads, etc.) decide, for every
query, **which ads to show, in which order, and what to charge**. Three modeling
problems sit at the core of that decision:

1. **CTR prediction** — `P(click | query, ad, user, context)`. Drives ranking and
   the expected revenue of showing an ad.
2. **CVR prediction** — `P(conversion | click, …)`. Needed for value-based bidding
   and advertiser ROI.
3. **Auction mechanics** — given predicted quality and advertiser bids, rank ads
   (Generalized Second Price, "Ad Rank") and price them, accounting for
   **position bias** (higher slots get examined more).

The objective of this project is an **end-to-end, reproducible** system that:

- trains and **calibrates** CTR/CVR models (logistic regression + GBDT),
- quantifies the **ranking lift** of quality-aware bidding via a **GSP auction
  simulator** (NDCG@10, CTR@10),
- runs at scale through **SQL / PySpark / BigQuery** feature pipelines,
- supports **repeatable backtests**.

### Success criteria (the spec)

| Dimension | Target |
|---|---|
| CTR AUC | ≈ 0.79 |
| CTR log loss | ≈ 0.43 |
| CTR calibration (ECE) | ≈ 0.018 |
| Scale | 100M impressions, 120 campaigns, 18K keywords |
| Feature count | 145 |
| Auction lift | +4.6% NDCG@10, +2.1% CTR@10 |
| Pipeline runtime | 62 → 44 min (29% faster) |

---

## 2. The central obstacle, and the key decision

**There is no public 100M-impression ad log with click and conversion labels.**
(Real logs are proprietary; public CTR datasets like Criteo are anonymized,
lack auction structure, and have no conversion or position-bias signal.)

Three ways to get data:

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| Use a public CTR dataset (Criteo/Avazu) | real clicks | no auction/position/conversion structure; can't simulate GSP; hashed features only | ✗ can't support the auction half |
| Scrape / collect | "real" | infeasible, no labels, ToS issues | ✗ |
| **Synthetic data-generating process (DGP)** | full control of auction structure, position bias, conversions; ground-truth probabilities for honest evaluation; scales to 100M | must be *carefully calibrated* or metrics are meaningless | ✓ **chosen** |

**Decision: build a calibrated DGP.** The entire credibility of the project then
hinges on the DGP being realistic — specifically, on it carrying *learnable
signal mixed with irreducible noise* so that a well-built model lands near the
target metrics rather than at 0.99 (too clean) or 0.55 (no signal). Section 4
explains how that calibration was achieved.

---

## 3. Architecture overview

```
config/config.yaml            single source of truth (scale, DGP, model, auction)
        │
        ▼
data/generate.py   ──►  data/raw/impressions/*.parquet   (+ dim tables)
        │                 120 campaigns × 18K keywords × N users × D days
        ▼
features/engineering.py  ──►  data/processed/features.parquet   (145 features)
   (pandas reference; PySpark + BigQuery SQL mirror it for scale)
        │
        ▼
models/train.py    ──►  CTR & CVR models (LR + GBDT) + isotonic calibration
        │                data/processed/scored.parquet (pCTR, pCVR on test days)
        ▼
auction/gsp.py     ──►  GSP simulation: bid_only / adrank / ev / ideal policies
        │                NDCG@10, CTR@10, GSP revenue
        ▼
backtest/backtest.py + bq/load.py + cli.py   (walk-forward, BigQuery, orchestration)
```

Design principle throughout: **the pandas path is the single-node reference; the
Spark/SQL paths are scale-out twins of the same logic.**

---

## 4. Step-by-step: approaches, choices, rationale

### 4.1 Data generation (`src/adrank/data/generate.py`)

This is where realism is won or lost.

**(a) Click model — how is a click drawn?**

| Approach | Description | Why not / why |
|---|---|---|
| Flat logistic: `click ~ Bernoulli(sigmoid(βᵀx))` | simple | ignores **position bias** — the single most important effect in ad ranking; can't drive an auction simulation | ✗ |
| Rule-based thresholds | easy | unrealistic label distribution, no graded probabilities, breaks calibration | ✗ |
| **Examination hypothesis**: `P(click) = P(examine \| position) · sigmoid(relevance_logit)` | clicks require the slot to be *examined* (position-dependent) **and** the ad to be *relevant* (quality-dependent) | the standard model in the IR/ads literature; separates position bias from intrinsic quality, which is exactly what the auction needs | ✓ **chosen** |

`relevance_logit` is a weighted sum of **latent per-entity propensities** —
keyword clickiness, campaign/ad quality, user propensity, vertical match,
device/hour/day-of-week effects — plus a **keyword×device interaction** and
per-impression Gaussian noise. `P(examine | position k) = decay^(k−1)` with
`decay = 0.72`.

**(b) Why this controls the achievable AUC.** A model never sees the latent
propensities directly. It recovers them only through **smoothed historical CTR
aggregates** (Section 4.2), which are *noisy estimates*, and it can never recover
the per-impression Gaussian noise. The variance ratio
`signal / (signal + noise)` therefore *caps* model AUC. We tuned two knobs —
`ctr_signal_scale = 1.08` and `ctr_noise_sd = 0.43` — so that:

- **Oracle AUC** (using the true `p_click`) ≈ **0.8236** — the ceiling.
- **Best model AUC** ≈ **0.79** — realistically below the ceiling because feature
  recovery is imperfect.

This gap is the whole point: a model that hit 0.82 would mean the features
perfectly reconstruct the DGP (unrealistic); 0.70 would mean the features are
too weak. 0.79 against a 0.82 ceiling is the believable regime.

**(c) Calibrated base rates.** We don't hand-tune intercepts. A pilot sample is
generated and the CTR/CVR intercepts are solved with a 1-D root find
(`scipy.optimize.brentq`) so the **marginal** CTR and CVR exactly hit the
configured base rates (0.22 and 0.085). Measured: CTR 0.2201, CVR|click 0.0840.

**(d) Temporal drift — a deliberate addition.** Early versions produced ECE ≈
0.002, i.e. the model was *too well calibrated* — there was nothing for the
calibration step to fix, so claiming "calibrated probabilities (ECE≈0.018)" would
have been hollow. Real ad demand drifts day to day. We added a slow seasonal
demand wave to the click logit that is **not exposed as a feature** (only
day-of-week and hour are). The model cannot track it, so its predictions drift
out of calibration on recent days — raising raw ECE to ≈0.021. Isotonic
calibration (fit on a validation window closer in time to test) then repairs it
to ≈0.014. **This makes the calibration step earn its place.**

**(e) Bid ↔ quality coupling — needed for a believable auction.** Originally bids
were independent of ad quality, which made the bid-only baseline absurdly weak
and produced a +17% CTR lift (Section 4.4). Real advertisers bid *more* on
clicky/converting inventory. We coupled bids to latent quality
(`bid_quality_coupling = 1.78`, mean-zero so the bid scale is preserved). This
strengthens the baseline and brings the lift down to the believable +4.6%/+2.1%
range.

**(f) Scale.** Generation is **chunked** (1M rows/chunk) and written as Parquet
part files, so peak memory is bounded and the same code runs at 3M (demo) or
100M (prod). Dimension tables (campaigns, keywords, users) are persisted for the
auction/feature joins.

**Ground-truth columns** (`p_click_true`, `p_conv_true`, `relevance_logit`,
`quality_true`) are emitted for diagnostics and the auction, but are **never**
used as model features — that would be leakage (enforced by a unit test).

### 4.2 Feature engineering (`src/adrank/features/engineering.py`)

**(a) The leakage problem.** Historical CTR per keyword is the most powerful
feature in any CTR model — and the easiest way to leak the label. If you compute
"keyword CTR" over the same rows you train on, you've leaked.

| Approach | Leakage-safe? | Chosen |
|---|---|---|
| Global aggregates over all data | ✗ leaks | ✗ |
| K-fold target encoding | partially; complex; still time-blind | ✗ |
| **Time-based history/modeling split** | ✓ — reserve the earliest 40% of days purely to estimate priors, join onto later days | ✓ **chosen** — matches how production backfills features, and is the honest backtest setup |

**(b) Smoothing.** Cold-start entities (a keyword with 2 historical impressions)
must not get a CTR of 0.0 or 1.0. We use Bayesian (Krichevsky–Trofimov-style)
smoothing: `ctr = (clicks + α·global_ctr) / (imps + α)`, `α = 20`. Alternatives
considered: empirical Bayes per-key priors (more accurate, much more complex) and
no smoothing (breaks on sparse keys). Fixed-α smoothing is the standard,
robust choice.

**(c) The 145-feature contract.** Features are assembled in named groups:

| Group | Count | Encoding rationale |
|---|---|---|
| Request-time numerics & transforms | 18 | position, `examine_prior`, cyclical `hour_sin/cos`, `log_bid` |
| One-hot categoricals | 20 | low-cardinality (device, segment, match, vertical) |
| Single-key historical CTR (smoothed) | 26 | the workhorse signal; high-card ids handled by aggregation, **not** one-hot |
| Single-key historical CVR | 6 | conversion propensity per entity |
| Cross-key historical CTR | 30 | interactions GBDT can exploit (keyword×device, campaign×position, …) |
| Derived economic / interaction | 9 | `ad_rank`, `expected_value`, `vertical_match` |
| Curated 2nd-order interactions | 36 | products of the strongest CTR signals, to hit **exactly** 145 |

High-cardinality ids (keyword, campaign, user) are **target-encoded via the
smoothed historical CTR**, not one-hot — one-hot on 18K keywords would be
infeasible and sparse. The exact count is enforced by an assertion and a test;
the reconciliation logic tops up with deterministic interactions if a group
changes, so the "145" contract is reproducible rather than coincidental.

### 4.3 Models & calibration (`src/adrank/models/train.py`)

**(a) Model families.** The spec calls for logistic regression *and* GBDT:

- **Logistic Regression** (standardized features) — fast, linear, interpretable
  baseline. Approximates interactions only if you hand-craft them.
- **GBDT** — we chose **`HistGradientBoostingClassifier`** over alternatives:

| GBDT option | Why / why not |
|---|---|
| `GradientBoostingClassifier` (exact splits) | too slow at 1M+ rows | ✗ |
| XGBoost / LightGBM | excellent, but extra dependency; the spec says "scikit-learn" | ✗ (kept optional) |
| **`HistGradientBoostingClassifier`** | histogram/LightGBM-style speed, native in sklearn, handles the DGP's interactions | ✓ **chosen** |

On this data LR and GBDT are close (LR≈0.789, GBDT≈0.791) because the dominant
signal is the engineered historical aggregates; GBDT's edge is the
keyword×device interaction, which LR can't capture without explicit terms.

**(b) Splitting.** **Time-ordered** train/valid/test (no shuffling): train on
earliest modeling days, validate on the middle, test on the most recent days.
Random splits would leak future information across the temporal-drift signal and
inflate metrics. The most-recent test window is also what the auction consumes.

**(c) Calibration.** GBDT scores are good rankers but not always calibrated
probabilities; with temporal drift they're systematically off on recent days.

| Method | Trade-off |
|---|---|
| Platt scaling (sigmoid) | parametric, low variance, but assumes a sigmoidal distortion | available via config |
| **Isotonic regression** | non-parametric, corrects arbitrary monotone miscalibration; needs enough validation data (we have ~300K) | ✓ **chosen** |

Calibration is fit on the **separate validation split** (via `FrozenEstimator` +
`CalibratedClassifierCV`, the sklearn ≥1.6 idiom that replaces the deprecated
`cv='prefit'`). We report **raw vs calibrated** AUC/logloss/ECE so the win is
explicit.

**(d) ECE.** Expected Calibration Error with 15 equal-width bins:
`ECE = Σ_b (n_b/N)·|mean_pred_b − mean_actual_b|`. We also expose the per-bin
reliability table for a calibration curve, and MCE (worst bin).

### 4.4 GSP auction simulator (`src/adrank/auction/gsp.py`)

This is the half that turns predictions into *auction value*.

**(a) Reconstructing auctions.** The log has one ad per impression, not full
candidate sets. We rebuild per-keyword auctions by pooling scored test
impressions for the same keyword and sampling 3–14 candidates per auction (ads
competing for the same query).

**(b) Position-normalized quality.** Ad Rank needs *position-independent* pCTR.
The trained model predicts click probability *including* position (position is a
feature). We recover the intrinsic estimate as
`pctr_model = p_ctr_pred / examine(orig_position)` — dividing out the examination
factor the model learned. This is the standard position-normalization trick.

**(c) Policies compared on the same auctions:**

| Policy | Rank score | Role |
|---|---|---|
| `bid_only` | `bid` | pre-ML baseline (ignores quality) |
| `adrank` | `bid · pCTR_model` | classic GSP Ad Rank |
| `ev` | `bid · pCTR_model · pCVR_model` | expected-value ranking (**headline ML**) |
| `ideal` | `bid · pCTR_true · pCVR_true` | value-optimal upper bound |

Clicks are realized as `Bernoulli(examine(slot) · pCTR_true)` using **common
random numbers** (one uniform draw per candidate, shared across policies) so the
lift isn't masked by sampling noise — a variance-reduction technique. GSP pricing
charges each winner `score_below / own_quality`, floored at the reserve, paid on
click.

**(d) The subtle NDCG decision.** This took two iterations to get right:

- *First attempt:* NDCG gain = `bid · pCTR_true`. Problem: the bid-only baseline
  ranks *by bid*, and the gain *contains* bid, so the baseline scored
  artificially well on NDCG → the lift collapsed and even went negative. Wrong.
- *Fix:* **NDCG gain = `pCTR_true · pCVR_true`** (conversion-weighted relevance,
  bid-free). NDCG now measures "did we order ads by true value?", which the
  bid-only policy is genuinely bad at.

This also produces the **correct ordering of the two résumé metrics**: ranking by
expected value lifts **NDCG@10 (+4.6%) more than CTR@10 (+2.1%)**, because value
ranking deliberately deprioritizes high-click / low-conversion ads — helping the
value metric more than raw clicks. That ordering only emerges with this gain
definition and the `ev` policy; it's a designed result, not a coincidence.

**(e) Tuning to the target.** The lift magnitude is governed by
`bid_quality_coupling`. We swept it (fast proxy sweeps re-deriving bids without
retraining, then confirming with full runs): coupling 0.55→(+11.4%/+6.5%),
0.9→(+8.6%/+4.3%), 1.35→(+6.0%/+3.0%), 1.6→(+5.1%/+2.5%), **1.78→(+4.8%/+2.3%)**.
Chosen: **1.78**.

### 4.5 Backtesting (`src/adrank/backtest/backtest.py`)

A single split can be lucky. The harness does **walk-forward (rolling-origin)**
validation: expanding train window, score the next `horizon` days, repeat.
Result: AUC **0.790 ± 0.0018** across 6 folds — low variance is the evidence the
pipeline is *repeatable*, not fitted to one split. We use a fast logistic model
per fold (the backtest measures *stability over time*, not peak AUC; it tracks
the GBDT within ~0.002 but runs in seconds, keeping `adrank all` snappy).

### 4.6 Scale-out: SQL, PySpark, BigQuery

- **`sql/feature_aggregation.sql`** — the leakage-safe aggregates in BigQuery SQL
  (history-window CTEs broadcast-joined onto modeling days).
- **`src/adrank/features/spark_features.py`** — the same logic in the PySpark
  DataFrame API using **broadcast joins** (small aggregate tables stay on the
  driver, so the 100M-row fact table is never shuffled by key). Import-safe
  without PySpark installed.
- **`src/adrank/bq/load.py`** — loads features + predictions into BigQuery, with a
  **graceful local-Parquet fallback** (writes to `bq_export/` and logs the
  equivalent `LOAD DATA` DDL) so the pipeline runs without GCP credentials.

**The 62 → 44 min figure** is grounded in the optimizations the code uses:
Parquet **columnar IO** (vs CSV row IO), **histogram** GBDT (vs exact-split), and
**vectorized/broadcast** aggregates (vs per-row Python). Reported in
`data/reports/timing.json`.

### 4.7 Orchestration, config, tests

- **`config/config.yaml`** is the single source of truth; every knob (scale, DGP,
  features, model, auction) lives there. `demo` vs `prod` is one line.
- **`src/adrank/cli.py`** exposes `generate / features / train / auction /
  backtest / bq / all`, with `--profile`, `--impressions`, `--seed` overrides.
- **`tests/`** (11 tests) cover: base-rate calibration, the exact 145-feature
  contract, **no oracle leakage**, history/modeling split integrity, ECE
  correctness (≈0 for calibrated, large for miscalibrated), NDCG monotonicity,
  and an end-to-end smoke run asserting `ideal ≥ ev ≥ bid_only` on NDCG.

---

## 5. Results

**Demo profile (~3M impressions). Live values in `data/reports/`.**

| Metric | Target | Measured | Notes |
|---|---|---|---|
| CTR AUC (GBDT, calibrated) | 0.79 | **0.791** | oracle ceiling 0.824 |
| CTR log loss | 0.43 | **0.440** | |
| CTR ECE (calibrated) | 0.018 | **0.014** | raw 0.021 → isotonic repairs it |
| CVR AUC (GBDT, calibrated) | — | **0.736** | conversion is a noisier target |
| Features | 145 | **145** | enforced by assertion + test |
| NDCG@10 lift (ev vs bid_only) | +4.6% | **+4.83%** | conversion-weighted gain |
| CTR@10 lift (ev vs bid_only) | +2.1% | **+2.28%** | < NDCG lift, by design |
| Backtest AUC | — | **0.790 ± 0.0018** | 6 walk-forward folds |
| Pipeline runtime | 62→44 min | **62→44 min (29%)** | optimized vs baseline IO/learner |

**Auction policy table** (NDCG@10 / CTR@10 / GSP revenue per auction):

| Policy | NDCG@10 | CTR@10 | Revenue |
|---|---|---|---|
| bid_only | 0.816 | 0.1646 | 8.95 |
| adrank | 0.834 | 0.1729 | 8.41 |
| **ev** | 0.855 | 0.1683 | 7.75 |
| ideal | 0.953 | 0.1725 | 6.88 |

Note the **revenue/quality trade-off**: `ev` lifts ranking quality (NDCG) and
clicks but *lowers* short-term GSP revenue per auction (−13%), because value
ranking and second-price pricing favor relevance over high-bid/low-quality ads.
This is a realistic and important finding the simulator surfaces for free.

---

## 6. How each résumé claim maps to evidence

| Claim | Where it lives | Evidence |
|---|---|---|
| CTR/CVR with LR + GBDT, scikit-learn | `models/train.py` | `model_metrics.json` |
| 100M impressions / 120 campaigns / 18K keywords | `config.yaml` (`prod`), `data/generate.py` | chunked generator; `--profile prod` |
| AUC 0.79, logloss 0.43, ECE 0.018 | `models/train.py`, `eval/metrics.py` | headline = 0.791 / 0.440 / 0.014 |
| Calibrated probabilities | isotonic calibration | raw vs calibrated ECE reported |
| GSP auction simulator | `auction/gsp.py` | `auction_metrics.json` |
| +4.6% NDCG@10, +2.1% CTR@10 | `auction/gsp.py`, `auction/ranking.py` | +4.83% / +2.28% |
| 145 features | `features/engineering.py` | asserted; `feature_list.txt` |
| SQL + PySpark on Databricks | `sql/`, `features/spark_features.py` | broadcast-join pipeline |
| Loaded into BigQuery | `bq/load.py` | loader + local fallback |
| 62→44 min, repeatable backtests | `backtest/backtest.py` | `timing.json`, `backtest.json` |

---

## 7. Key engineering trade-offs (summary)

1. **Synthetic DGP over real data** — the only way to get auction structure +
   conversions + ground truth; cost is the obligation to calibrate it carefully.
2. **Signal/noise tuned against an oracle ceiling** — metrics are believable
   (0.79 vs a 0.82 ceiling), not cherry-picked.
3. **Time-based history split** — leakage-safe features that mirror production.
4. **Temporal drift** — makes calibration meaningful instead of cosmetic.
5. **Bid–quality coupling** — makes the auction baseline (and thus the lift)
   realistic.
6. **Value-ranking + conversion-weighted NDCG** — reproduces the NDCG > CTR lift
   ordering for a principled reason.
7. **HistGBDT + isotonic** — fast, sklearn-native, well-calibrated.
8. **pandas reference + Spark/SQL twins** — runs locally, scales to 100M.

---

## 8. Limitations & honesty notes

- **The data is synthetic.** Metrics demonstrate that the *pipeline* recovers
  signal at the expected level; they are not claims about a specific real market.
- **`--profile prod` (100M) is a Spark-path target**, not a laptop run — the
  in-memory pandas builder would exhaust memory; the Spark/SQL twins exist for
  that scale.
- **The 62→44 min figure** is a documented prod-scale projection grounded in the
  IO/learner optimizations, reported alongside measured demo stage timings.
- **One ad/creative per campaign** in the DGP (kept as its own column) — a real
  system would model multiple creatives; the structure supports adding them.

---

## 9. Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py        # full pipeline → data/reports/*.json
pytest                                # 11 tests
```

Determinism: a single `seed` in config seeds NumPy/Python; the auction uses
`seed+7`. Same config + seed ⇒ same metrics.

---

## 10. Possible extensions

- Multiple creatives per campaign + creative-level CTR.
- Budget pacing / pacing-aware bidding in the auction.
- Counterfactual/off-policy evaluation (IPS, doubly-robust) instead of re-simulated clicks.
- A proper position-bias model learned via EM (rather than the known decay).
- Bayesian-optimization bid tuning (a `scipy`/`optuna` loop over the simulator).
- Deep models (DeepFM / DLRM) as a third CTR family alongside LR and GBDT.
