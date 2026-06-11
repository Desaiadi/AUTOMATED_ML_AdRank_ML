"""End-to-end smoke test: generate -> features -> train -> auction on tiny data."""
import warnings

from adrank.utils.common import set_seed


def test_full_pipeline_smoke(featured):
    warnings.filterwarnings("ignore")
    cfg, _ = featured

    from adrank.models.train import train
    report = train(cfg)
    ctr = report["ctr"]["headline"]
    # on tiny data we only assert the model learns *something* and is calibrated-ish
    assert ctr["auc"] > 0.6
    assert 0.0 <= ctr["ece"] < 0.1
    assert report["n_features"] == 145

    from adrank.auction.gsp import simulate
    rng = set_seed(cfg.seed + 7)
    sim = simulate(cfg, rng, n_auctions=2000)
    pol = sim["policies"]
    # value-optimal ideal should dominate the bid-only baseline on NDCG, and the
    # ML expected-value policy should sit between them.
    assert pol["ideal"]["ndcg_at_k"] >= pol["ev"]["ndcg_at_k"] >= pol["bid_only"]["ndcg_at_k"]
    assert sim["adrank_vs_bid_only"]["ndcg_at_k_lift_pct"] > 0


def test_backtest_runs(featured):
    warnings.filterwarnings("ignore")
    cfg, _ = featured
    from adrank.models.train import train
    from adrank.backtest.backtest import run_backtest
    train(cfg)
    bt = run_backtest(cfg, horizon=2, min_train_days=4)
    assert bt["n_folds"] >= 1
    assert bt["auc_mean"] > 0.55
