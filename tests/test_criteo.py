"""The Criteo adapter feeds the same CTR core (fit_eval_ctr) as the synthetic path."""
import warnings

from adrank.config import load_config
from adrank.data import criteo
from adrank.features.criteo_features import build_criteo_features
from adrank.models.train import fit_eval_ctr


def test_criteo_sample_format(tmp_path):
    path = criteo.make_sample(tmp_path / "c.txt", n=5000, seed=1)
    df = criteo.load_criteo(path, n_day_buckets=8)
    # exact Criteo schema: label + 13 int + 26 cat (+ derived day)
    assert set(criteo.INT_COLS) | set(criteo.CAT_COLS) <= set(df.columns)
    assert len(criteo.INT_COLS) == 13 and len(criteo.CAT_COLS) == 26
    assert df["clicked"].isin([0, 1]).all()
    assert 0.15 < df["clicked"].mean() < 0.35   # Criteo-like base rate


def test_criteo_features_and_ctr(tmp_path):
    warnings.filterwarnings("ignore")
    cfg = load_config()
    path = criteo.make_sample(tmp_path / "c.txt", n=60_000, seed=2)
    df = criteo.load_criteo(path, n_day_buckets=10)
    feats, cols = build_criteo_features(df, history_fraction=0.4, alpha=20.0)
    # 13*3 integer + 26*2 categorical + 4 crosses = 95 features
    assert len(cols) == 95
    assert "clicked" in feats and "day" in feats
    # no oracle/raw leakage: features are only engineered transforms
    assert not any(c in cols for c in ("clicked", "day"))

    res = fit_eval_ctr(feats, cols, cfg)
    assert res["headline"]["auc"] > 0.7          # learns real signal
    assert res["headline"]["ece"] < 0.1
