"""Schema definitions for the synthetic auction log.

The impression fact table mimics a real search-ads serving log. Columns fall into
four groups:

* **keys / dims**   — identifiers and categorical context known at request time.
* **economics**     — bid and (latent) quality used by the auction simulator.
* **oracle**        — ground-truth probabilities and latent scores. These are the
                      data-generating process's internal state. They are emitted
                      for diagnostics (e.g. oracle AUC, auction click sampling) but
                      are NEVER used as model features — doing so would be leakage.
* **labels**        — observed outcomes (``clicked``, ``converted``).
"""
from __future__ import annotations

# Columns available to feature engineering (known at impression/request time).
REQUEST_TIME_COLUMNS = [
    "impression_id",
    "ts",
    "day",
    "hour",
    "dow",
    "is_weekend",
    "user_id",
    "user_segment",
    "device",
    "keyword_id",
    "keyword_category",
    "keyword_n_tokens",
    "match_type",
    "campaign_id",
    "advertiser_id",
    "vertical",
    "ad_id",
    "position",
    "n_eligible",
    "bid",
]

# Latent / ground-truth columns. Kept out of the feature matrix on purpose.
ORACLE_COLUMNS = [
    "quality_true",      # latent ad quality used as Ad Rank quality in the auction
    "relevance_logit",   # systematic part of the click logit (pre-position)
    "p_click_true",
    "p_conv_true",
]

LABEL_COLUMNS = ["clicked", "converted"]

ALL_COLUMNS = REQUEST_TIME_COLUMNS + ORACLE_COLUMNS + LABEL_COLUMNS

CATEGORICAL_COLUMNS = [
    "user_segment",
    "device",
    "keyword_category",
    "match_type",
    "vertical",
]

# High-cardinality id columns — handled via target/hash encoding, not one-hot.
HIGH_CARD_COLUMNS = ["campaign_id", "keyword_id", "ad_id", "advertiser_id", "user_id"]
