import numpy as np
import pandas as pd

from adrank.data.schema import ORACLE_COLUMNS


def test_generate_hits_base_rates(generated):
    cfg, summary = generated
    # marginal CTR/CVR should land near the configured base rates
    assert abs(summary["ctr"] - cfg.dgp.ctr_base_rate) < 0.02
    assert abs(summary["cvr_given_click"] - cfg.dgp.cvr_base_rate) < 0.02
    assert summary["n_campaigns"] == 120


def test_feature_count_is_exactly_target(featured):
    cfg, summary = featured
    assert summary["n_features"] == cfg.features.target_feature_count == 145


def test_feature_list_matches_columns(featured):
    cfg, _ = featured
    import pathlib
    cols = pathlib.Path(cfg.paths["processed_dir"], "feature_list.txt").read_text().splitlines()
    assert len(cols) == 145
    df = pd.read_parquet(pathlib.Path(cfg.paths["processed_dir"], "features.parquet"))
    assert set(cols) <= set(df.columns)
    # features must contain no NaN (cold-start filled with global priors)
    assert not df[cols].isna().any().any()


def test_no_oracle_columns_leak_into_features(featured):
    cfg, _ = featured
    import pathlib
    cols = set(pathlib.Path(cfg.paths["processed_dir"], "feature_list.txt").read_text().splitlines())
    # ground-truth/oracle columns must never be used as model features
    assert cols.isdisjoint(set(ORACLE_COLUMNS))
    assert "clicked" not in cols and "converted" not in cols


def test_history_modeling_split_no_overlap(featured):
    cfg, summary = featured
    # modeling rows all come from days >= history cutoff
    import pathlib
    df = pd.read_parquet(pathlib.Path(cfg.paths["processed_dir"], "features.parquet"))
    assert df["day"].min() >= summary["history_days"]
