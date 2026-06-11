"""Shared pytest fixtures: a tiny, fast end-to-end config in a temp directory."""
from __future__ import annotations

import pytest

from adrank.config import load_config
from adrank.utils.common import set_seed


@pytest.fixture
def tiny_cfg(tmp_path):
    """A miniature config pointing all paths at a temp dir for fast tests."""
    cfg = load_config()
    cfg.scale["n_users"] = 4000
    cfg.scale["n_days"] = 12
    cfg.scale["impressions"] = 120_000
    cfg.scale["n_keywords"] = 1500
    for key in ("raw_dir", "processed_dir", "artifacts_dir", "reports_dir"):
        d = tmp_path / key
        d.mkdir(parents=True, exist_ok=True)
        cfg.paths[key] = str(d)
    return cfg


@pytest.fixture
def generated(tiny_cfg):
    from adrank.data import generate as g
    rng = set_seed(tiny_cfg.seed)
    summary = g.generate(tiny_cfg, rng, chunk_size=60_000)
    return tiny_cfg, summary


@pytest.fixture
def featured(generated):
    from adrank.features.engineering import build_features
    cfg, _ = generated
    summary = build_features(cfg)
    return cfg, summary
