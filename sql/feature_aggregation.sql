-- =============================================================================
-- feature_aggregation.sql  (BigQuery dialect)
--
-- Builds the leakage-safe historical CTR/CVR aggregate features that dominate
-- the feature set. The earliest `${HIST_CUTOFF}` days are the HISTORY window
-- used to estimate per-entity propensities; those are joined onto the later
-- MODELING window. This is the SQL equivalent of adrank.features.engineering
-- (single-key CTR shown here; cross-key aggregates follow the same pattern).
--
-- Bayesian smoothing:  ctr = (clicks + alpha*global_ctr) / (imps + alpha)
-- with alpha = ${ALPHA}.  global_ctr is computed over the history window.
-- =============================================================================

DECLARE hist_cutoff INT64 DEFAULT ${HIST_CUTOFF};
DECLARE alpha       FLOAT64 DEFAULT ${ALPHA};

CREATE TEMP TABLE globals AS
SELECT
  AVG(clicked)                                            AS global_ctr,
  SAFE_DIVIDE(SUM(converted), SUM(clicked))              AS global_cvr
FROM `${PROJECT}.adrank.impressions_raw`
WHERE day < hist_cutoff;

-- ---- single-key historical CTR (one CTE per key; keyword shown) ----
CREATE TEMP TABLE ctr_keyword AS
SELECT
  keyword_id,
  (SUM(clicked) + alpha * (SELECT global_ctr FROM globals))
    / (COUNT(*) + alpha)                                  AS ctr__keyword_id,
  LOG(1 + COUNT(*))                                       AS logimp__keyword_id
FROM `${PROJECT}.adrank.impressions_raw`
WHERE day < hist_cutoff
GROUP BY keyword_id;

CREATE TEMP TABLE ctr_campaign AS
SELECT
  campaign_id,
  (SUM(clicked) + alpha * (SELECT global_ctr FROM globals))
    / (COUNT(*) + alpha)                                  AS ctr__campaign_id,
  LOG(1 + COUNT(*))                                       AS logimp__campaign_id
FROM `${PROJECT}.adrank.impressions_raw`
WHERE day < hist_cutoff
GROUP BY campaign_id;

-- ---- single-key historical CVR (conv | click) ----
CREATE TEMP TABLE cvr_keyword AS
SELECT
  keyword_id,
  (SUM(converted) + alpha * (SELECT global_cvr FROM globals))
    / (SUM(clicked) + alpha)                              AS cvr__keyword_id
FROM `${PROJECT}.adrank.impressions_raw`
WHERE day < hist_cutoff AND clicked = 1
GROUP BY keyword_id;

-- ---- cross-key historical CTR (keyword x device shown) ----
CREATE TEMP TABLE ctr_keyword_x_device AS
SELECT
  keyword_id, device,
  (SUM(clicked) + alpha * (SELECT global_ctr FROM globals))
    / (COUNT(*) + alpha)                                  AS ctr__keyword_id_X_device
FROM `${PROJECT}.adrank.impressions_raw`
WHERE day < hist_cutoff
GROUP BY keyword_id, device;

-- ---- assemble the wide MODELING table (history features joined to later days) ----
CREATE OR REPLACE TABLE `${PROJECT}.adrank.features_wide` AS
SELECT
  i.impression_id, i.ts, i.day, i.hour, i.dow, i.is_weekend, i.position,
  i.n_eligible, i.keyword_n_tokens, i.bid, LOG(1 + i.bid) AS log_bid,
  POW(${POSITION_DECAY}, i.position - 1)                  AS examine_prior,
  -- historical aggregates (NULLs for cold-start entities default to the global)
  COALESCE(ck.ctr__keyword_id,  g.global_ctr)            AS ctr__keyword_id,
  COALESCE(ck.logimp__keyword_id, 0.0)                  AS logimp__keyword_id,
  COALESCE(cc.ctr__campaign_id, g.global_ctr)           AS ctr__campaign_id,
  COALESCE(cc.logimp__campaign_id, 0.0)                AS logimp__campaign_id,
  COALESCE(vk.cvr__keyword_id,  g.global_cvr)           AS cvr__keyword_id,
  COALESCE(xd.ctr__keyword_id_X_device, g.global_ctr)   AS ctr__keyword_id_X_device,
  -- labels
  i.clicked, i.converted
FROM `${PROJECT}.adrank.impressions_raw` i
CROSS JOIN globals g
LEFT JOIN ctr_keyword          ck ON i.keyword_id = ck.keyword_id
LEFT JOIN ctr_campaign         cc ON i.campaign_id = cc.campaign_id
LEFT JOIN cvr_keyword          vk ON i.keyword_id = vk.keyword_id
LEFT JOIN ctr_keyword_x_device xd ON i.keyword_id = xd.keyword_id AND i.device = xd.device
WHERE i.day >= hist_cutoff;
