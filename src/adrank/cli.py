"""AdRank-ML command-line orchestrator.

Subcommands run individual stages or the whole pipeline end to end:

    adrank generate     # synthetic auction log -> data/raw/
    adrank features     # 145-feature wide table -> data/processed/
    adrank train        # CTR/CVR models + calibration + scoring
    adrank auction      # GSP auction simulation + NDCG@10/CTR@10 lift
    adrank backtest     # walk-forward backtest + timing report
    adrank bq           # load features/predictions to BigQuery (or local)
    adrank all          # the full pipeline, in order

Global flags: --config PATH, --profile {demo,prod}, --impressions N, --seed N.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .utils.common import dump_timings, get_logger, save_json, set_seed

LOGGER = get_logger("adrank.cli")


def _prepare(args) -> object:
    # `--profile` is applied by editing the raw YAML's active profile before the
    # loader resolves it, so the chosen profile's scale block becomes cfg.scale.
    import yaml

    from .config import DEFAULT_CONFIG_PATH, Config, _resolve_paths

    path = args.config or DEFAULT_CONFIG_PATH
    if args.profile:
        raw = yaml.safe_load(open(path))
        raw["scale"]["profile"] = args.profile
        cfg = Config(raw)
        merged = dict(cfg["scale"][args.profile]); merged["profile"] = args.profile
        cfg["scale"] = Config(merged)
        _resolve_paths(cfg)
    else:
        cfg = load_config(path)

    if args.impressions:
        cfg.scale["impressions"] = int(args.impressions)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    return cfg


def cmd_generate(cfg):
    from .data import generate as g
    rng = set_seed(cfg.seed)
    return g.generate(cfg, rng, chunk_size=1_000_000)


def cmd_features(cfg):
    from .features.engineering import build_features
    return build_features(cfg)


def cmd_train(cfg):
    from .models.train import train
    return train(cfg)


def cmd_auction(cfg):
    from .auction.gsp import simulate
    rng = set_seed(cfg.seed + 7)
    return simulate(cfg, rng, n_auctions=40000)


def cmd_backtest(cfg):
    from .backtest.backtest import run_backtest, timing_report
    bt = run_backtest(cfg)
    tr = timing_report(cfg)
    return {"backtest": bt, "timing": tr}


def cmd_bq(cfg):
    from .bq.load import load_processed
    return load_processed(cfg)


def cmd_all(cfg):
    summary = {}
    summary["generate"] = cmd_generate(cfg)
    summary["features"] = cmd_features(cfg)
    summary["train"] = cmd_train(cfg)
    summary["auction"] = cmd_auction(cfg)
    summary["backtest"] = cmd_backtest(cfg)
    summary["bq"] = cmd_bq(cfg)
    dump_timings(Path(cfg.paths["reports_dir"]) / "timing_raw.json")
    save_json(_headline(summary), Path(cfg.paths["reports_dir"]) / "headline.json")
    _print_headline(summary)
    return summary


def _headline(summary: dict) -> dict:
    ctr = summary["train"]["ctr"]["headline"]
    cvr = summary["train"]["cvr"]["headline"]
    lift = summary["auction"]["adrank_vs_bid_only"]
    return {
        "ctr_auc": ctr["auc"], "ctr_logloss": ctr["logloss"], "ctr_ece": ctr["ece"],
        "cvr_auc": cvr["auc"], "cvr_logloss": cvr["logloss"],
        "ndcg10_lift_pct": lift["ndcg_at_k_lift_pct"],
        "ctr10_lift_pct": lift["ctr_at_k_lift_pct"],
        "n_features": summary["features"]["n_features"],
        "impressions": summary["generate"]["impressions"],
    }


def _print_headline(summary: dict) -> None:
    h = _headline(summary)
    LOGGER.info("=" * 64)
    LOGGER.info("ADRANK-ML HEADLINE  (impressions=%s, features=%d)",
                f"{h['impressions']:,}", h["n_features"])
    LOGGER.info("  CTR : AUC=%.4f  logloss=%.4f  ECE=%.4f",
                h["ctr_auc"], h["ctr_logloss"], h["ctr_ece"])
    LOGGER.info("  CVR : AUC=%.4f  logloss=%.4f", h["cvr_auc"], h["cvr_logloss"])
    LOGGER.info("  AUCTION : NDCG@10=+%.2f%%  CTR@10=+%.2f%%",
                h["ndcg10_lift_pct"], h["ctr10_lift_pct"])
    LOGGER.info("=" * 64)


_COMMANDS = {
    "generate": cmd_generate, "features": cmd_features, "train": cmd_train,
    "auction": cmd_auction, "backtest": cmd_backtest, "bq": cmd_bq, "all": cmd_all,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="adrank", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=list(_COMMANDS))
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--profile", default=None, choices=["demo", "prod"])
    ap.add_argument("--impressions", default=None, type=int,
                    help="override scale.impressions")
    ap.add_argument("--seed", default=None, type=int)
    args = ap.parse_args(argv)

    cfg = _prepare(args)
    LOGGER.info("profile=%s impressions=%s seed=%s",
                cfg.scale.profile, f"{cfg.scale.impressions:,}", cfg.seed)
    result = _COMMANDS[args.command](cfg)
    if args.command != "all":
        import json
        print(json.dumps(result, indent=2, default=str)[:2000])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
