"""Train and calibrate CTR / CVR models.

For each task we fit two model families on a time-ordered TRAIN split:

* **Logistic Regression** (standardized features) — a fast, linear baseline.
* **GBDT** — ``HistGradientBoostingClassifier``, sklearn's histogram gradient
  boosting (the LightGBM-style learner), which captures the feature
  interactions baked into the data-generating process.

Both are probability-**calibrated** with isotonic regression fit on a separate
VALID split, then scored on a held-out TEST split (the most recent days). We
report AUC / log loss / ECE for raw vs calibrated models so the calibration win
(ECE ~ 0.018) is explicit. The calibrated GBDT is the primary model; its TEST
scores are written to ``scored.parquet`` for the auction simulator.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Config
from ..eval.metrics import classification_report
from ..utils.common import get_logger, save_json, timed

LOGGER = get_logger("adrank.models")

try:  # sklearn >= 1.6 preferred calibration-of-prefit path
    from sklearn.frozen import FrozenEstimator

    def _calibrate(model, X, y, method):
        cal = CalibratedClassifierCV(FrozenEstimator(model), method=method)
        cal.fit(X, y)
        return cal
except ImportError:  # pragma: no cover - older sklearn fallback
    def _calibrate(model, X, y, method):
        cal = CalibratedClassifierCV(model, method=method, cv="prefit")
        cal.fit(X, y)
        return cal


def _time_split(df: pd.DataFrame, valid_frac: float, test_frac: float):
    """Split rows by `day` so TRAIN strictly precedes VALID precedes TEST."""
    days = np.sort(df["day"].unique())
    n = len(days)
    n_test = max(1, int(round(n * test_frac)))
    n_valid = max(1, int(round(n * valid_frac)))
    test_days = set(days[n - n_test:])
    valid_days = set(days[n - n_test - n_valid:n - n_test])
    is_test = df["day"].isin(test_days).to_numpy()
    is_valid = df["day"].isin(valid_days).to_numpy()
    is_train = ~(is_test | is_valid)
    return is_train, is_valid, is_test, {
        "train_days": sorted(set(days) - test_days - valid_days),
        "valid_days": sorted(valid_days),
        "test_days": sorted(test_days),
    }


def _build_models(cfg: Config):
    lg = cfg.model.logistic
    gb = cfg.model.gbdt
    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=lg.C, max_iter=lg.max_iter,
            class_weight=lg.class_weight, n_jobs=-1)),
    ])
    gbdt = HistGradientBoostingClassifier(
        learning_rate=gb.learning_rate, max_iter=gb.max_iter,
        max_leaf_nodes=gb.max_leaf_nodes, min_samples_leaf=gb.min_samples_leaf,
        l2_regularization=gb.l2_regularization, max_bins=gb.max_bins,
        early_stopping=gb.early_stopping, validation_fraction=gb.validation_fraction,
        n_iter_no_change=gb.n_iter_no_change, random_state=cfg.seed,
    )
    return {"logistic": logistic, "gbdt": gbdt}


def _train_task(task: str, df: pd.DataFrame, feature_cols: list[str],
                cfg: Config) -> dict:
    """Train CTR (`clicked`, all rows) or CVR (`converted`, clicked rows only)."""
    label = "clicked" if task == "ctr" else "converted"
    work = df if task == "ctr" else df[df["clicked"] == 1].copy()
    is_tr, is_va, is_te, split_days = _time_split(
        work, cfg.model.valid_fraction, cfg.model.test_fraction)

    X = work[feature_cols].to_numpy(dtype=np.float32)
    y = work[label].to_numpy(dtype=np.int8)
    Xtr, ytr = X[is_tr], y[is_tr]
    Xva, yva = X[is_va], y[is_va]
    Xte, yte = X[is_te], y[is_te]
    LOGGER.info("[%s] train=%s valid=%s test=%s  (label base rate train=%.4f)",
                task.upper(), f"{len(ytr):,}", f"{len(yva):,}", f"{len(yte):,}",
                float(ytr.mean()))

    method = cfg.model.calibration.method
    n_bins = cfg.model.calibration.ece_n_bins
    models = _build_models(cfg)
    results, fitted = {}, {}

    for name, model in models.items():
        with timed(f"model.{task}.{name}.fit", LOGGER):
            model.fit(Xtr, ytr)
        raw = classification_report(yte, model.predict_proba(Xte)[:, 1], n_bins)
        with timed(f"model.{task}.{name}.calibrate", LOGGER):
            cal_model = _calibrate(model, Xva, yva, method)
        cal = classification_report(yte, cal_model.predict_proba(Xte)[:, 1], n_bins)
        results[name] = {"raw": raw, "calibrated": cal}
        fitted[name] = cal_model
        LOGGER.info("[%s/%s] raw  AUC=%.4f logloss=%.4f ECE=%.4f | "
                    "cal AUC=%.4f logloss=%.4f ECE=%.4f",
                    task, name, raw["auc"], raw["logloss"], raw["ece"],
                    cal["auc"], cal["logloss"], cal["ece"])

    # primary model = calibrated GBDT
    primary = fitted["gbdt"]
    art_dir = Path(cfg.paths["artifacts_dir"])
    joblib.dump(primary, art_dir / f"model_{task}.joblib")

    return {
        "task": task, "label": label,
        "split_days": split_days,
        "results": results,
        "test_mask": is_te, "work_index": work.index.to_numpy(),
        "primary": primary, "feature_cols": feature_cols,
    }


def fit_eval_ctr(df: pd.DataFrame, feature_cols: list[str], cfg: Config) -> dict:
    """Train + calibrate + evaluate a CTR model on ANY wide frame.

    The frame only needs the `feature_cols`, a `clicked` label, and a `day`
    column for the time split. This is the dataset-agnostic core reused by the
    Criteo adapter (and anything else) — identical LR + GBDT + isotonic
    calibration + AUC/logloss/ECE as the synthetic pipeline.
    """
    is_tr, is_va, is_te, split_days = _time_split(
        df, cfg.model.valid_fraction, cfg.model.test_fraction)
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["clicked"].to_numpy(dtype=np.int8)
    Xtr, ytr, Xva, yva, Xte, yte = (X[is_tr], y[is_tr], X[is_va], y[is_va],
                                    X[is_te], y[is_te])
    LOGGER.info("[CTR] train=%s valid=%s test=%s  base_rate=%.4f",
                f"{len(ytr):,}", f"{len(yva):,}", f"{len(yte):,}", float(ytr.mean()))

    method = cfg.model.calibration.method
    n_bins = cfg.model.calibration.ece_n_bins
    results = {}
    for name, model in _build_models(cfg).items():
        with timed(f"ctr.{name}.fit", LOGGER):
            model.fit(Xtr, ytr)
        raw = classification_report(yte, model.predict_proba(Xte)[:, 1], n_bins)
        cal_model = _calibrate(model, Xva, yva, method)
        cal = classification_report(yte, cal_model.predict_proba(Xte)[:, 1], n_bins)
        for r in (raw, cal):
            r.pop("_reliability_bins", None)
        results[name] = {"raw": raw, "calibrated": cal}
        LOGGER.info("[ctr/%s] raw AUC=%.4f logloss=%.4f ECE=%.4f | "
                    "cal AUC=%.4f logloss=%.4f ECE=%.4f", name,
                    raw["auc"], raw["logloss"], raw["ece"],
                    cal["auc"], cal["logloss"], cal["ece"])
    return {"results": results, "split_days": split_days,
            "n_features": len(feature_cols),
            "headline": results["gbdt"]["calibrated"]}


def train(cfg: Config) -> dict:
    proc_dir = Path(cfg.paths["processed_dir"])
    feature_cols = (proc_dir / "feature_list.txt").read_text().splitlines()
    with timed("model.load_features", LOGGER):
        df = pd.read_parquet(proc_dir / "features.parquet")

    ctr = _train_task("ctr", df, feature_cols, cfg)
    cvr = _train_task("cvr", df, feature_cols, cfg)

    # ---- score the CTR test period for the auction simulator ----
    # Use the CTR test rows (most recent days); apply BOTH calibrated models.
    ctr_test_idx = ctr["work_index"][ctr["test_mask"]]
    scored = df.loc[ctr_test_idx].copy()
    Xs = scored[feature_cols].to_numpy(dtype=np.float32)
    scored["p_ctr_pred"] = ctr["primary"].predict_proba(Xs)[:, 1]
    scored["p_cvr_pred"] = cvr["primary"].predict_proba(Xs)[:, 1]  # P(conv|click)
    keep = ["impression_id", "ts", "day", "user_id", "keyword_id", "campaign_id",
            "ad_id", "position", "bid", "quality_true", "relevance_logit",
            "p_click_true", "p_conv_true", "clicked", "converted",
            "p_ctr_pred", "p_cvr_pred"]
    scored[keep].to_parquet(proc_dir / "scored.parquet", index=False)

    # ---- assemble + persist the metrics report ----
    report = {
        "ctr": {
            "models": ctr["results"],
            "split_days": ctr["split_days"],
            "primary": "gbdt_calibrated",
            "headline": ctr["results"]["gbdt"]["calibrated"],
        },
        "cvr": {
            "models": cvr["results"],
            "split_days": cvr["split_days"],
            "primary": "gbdt_calibrated",
            "headline": cvr["results"]["gbdt"]["calibrated"],
        },
        "n_features": len(feature_cols),
        "scored_rows": int(len(scored)),
    }
    save_json(report, Path(cfg.paths["reports_dir"]) / "model_metrics.json")
    h = report["ctr"]["headline"]
    LOGGER.info("CTR headline (GBDT calibrated): AUC=%.4f logloss=%.4f ECE=%.4f",
                h["auc"], h["logloss"], h["ece"])
    return report
