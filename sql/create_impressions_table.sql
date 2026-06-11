-- =============================================================================
-- create_impressions_table.sql
-- Raw impression log DDL (BigQuery dialect). Mirrors the synthetic generator's
-- schema (adrank.data.schema). Partitioned by event day and clustered by the
-- highest-cardinality join keys so the feature-aggregation queries prune well.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS `${PROJECT}.adrank`
  OPTIONS (location = 'US');

CREATE TABLE IF NOT EXISTS `${PROJECT}.adrank.impressions_raw`
(
  impression_id     INT64    NOT NULL,
  ts                INT64,                 -- seconds since horizon start
  day               INT64    NOT NULL,
  hour              INT64,
  dow               INT64,
  is_weekend        INT64,
  user_id           INT64,
  user_segment      STRING,
  device            STRING,
  keyword_id        INT64,
  keyword_category  STRING,
  keyword_n_tokens  INT64,
  match_type        STRING,
  campaign_id       INT64,
  advertiser_id     INT64,
  vertical          STRING,
  ad_id             INT64,
  position          INT64,
  n_eligible        INT64,
  bid               FLOAT64,
  clicked           INT64,                 -- label
  converted         INT64                  -- label
)
PARTITION BY RANGE_BUCKET(day, GENERATE_ARRAY(0, 400, 1))
CLUSTER BY keyword_id, campaign_id, user_id;
