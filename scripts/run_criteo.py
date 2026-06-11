#!/usr/bin/env python
"""Run the AdRank-ML CTR pipeline on the Criteo dataset.

    # validate end-to-end on a generated Criteo-format sample (no download):
    python scripts/run_criteo.py --sample 200000

    # run on a real Criteo file (Display train.txt or a 1TB day file):
    python scripts/run_criteo.py --path /data/criteo/train.txt --nrows 5000000

Uses the SAME models, calibration, metrics, and walk-forward backtest as the
synthetic pipeline — only the data loader + feature builder are Criteo-specific.
Only the CTR half applies (Criteo Display/1TB have no conversions or auctions).
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from adrank.config import load_config            # noqa: E402
from adrank.data import criteo                   # noqa: E402
from adrank.features.criteo_features import build_criteo_features  # noqa: E402
from adrank.models.train import fit_eval_ctr     # noqa: E402
from adrank.backtest.backtest import run_backtest  # noqa: E402
from adrank.utils.common import get_logger, save_json  # noqa: E402

LOGGER = get_logger("adrank.criteo.run")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", help="path to a real Criteo TSV file")
    ap.add_argument("--sample", type=int, default=None,
                    help="generate a Criteo-format sample of this many rows instead")
    ap.add_argument("--nrows", type=int, default=None, help="limit rows read")
    ap.add_argument("--day-buckets", type=int, default=10)
    args = ap.parse_args(argv)

    cfg = load_config()
    reports = Path(cfg.paths["reports_dir"])

    if args.sample:
        path = Path(cfg.paths["raw_dir"]) / "criteo_sample.txt"
        criteo.make_sample(path, n=args.sample, seed=cfg.seed)
    elif args.path:
        path = args.path
    else:
        ap.error("pass either --path <real file> or --sample <n>")

    df = criteo.load_criteo(path, nrows=args.nrows, n_day_buckets=args.day_buckets)
    feats, cols = build_criteo_features(
        df, history_fraction=cfg.features.history_fraction,
        alpha=cfg.features.smoothing_alpha)

    result = fit_eval_ctr(feats, cols, cfg)

    # reuse the walk-forward backtest by writing the standard processed artifacts
    proc = Path(cfg.paths["processed_dir"])
    feats.to_parquet(proc / "features.parquet", index=False)
    (proc / "feature_list.txt").write_text("\n".join(cols))
    bt = run_backtest(cfg, horizon=2, min_train_days=3, task="ctr")

    report = {"dataset": "criteo", "source": str(path),
              "n_features": result["n_features"],
              "ctr": result["results"], "headline": result["headline"],
              "backtest": {k: bt[k] for k in
                           ["n_folds", "auc_mean", "auc_std", "logloss_mean"]}}
    save_json(report, reports / "criteo_metrics.json")
    h = result["headline"]
    LOGGER.info("=" * 60)
    LOGGER.info("CRITEO CTR (GBDT, calibrated): AUC=%.4f logloss=%.4f ECE=%.4f",
                h["auc"], h["logloss"], h["ece"])
    LOGGER.info("Backtest AUC=%.4f +/- %.4f over %d folds",
                bt["auc_mean"], bt["auc_std"], bt["n_folds"])
    LOGGER.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
