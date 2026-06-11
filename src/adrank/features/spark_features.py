"""PySpark feature pipeline (Databricks-style) — the production scale-out path.

The pandas builder in ``engineering.py`` is the single-node reference; at 100M
impressions the same leakage-safe aggregates run in Spark on Databricks and the
wide table is written to Parquet / loaded into BigQuery. This module expresses
that logic in the Spark DataFrame API.

It is import-safe without PySpark: ``pyspark`` is imported lazily inside
``build_features_spark`` so the rest of the package (and the local pipeline)
works whether or not Spark is installed. Run it with:

    spark-submit --master local[*] -m adrank.features.spark_features \
        --raw  data/raw/impressions \
        --out  data/processed/features_spark.parquet
"""
from __future__ import annotations

import argparse


# Same key sets as the pandas reference, kept in sync intentionally.
CTR_KEYS = ["keyword_id", "campaign_id", "ad_id", "advertiser_id", "user_id",
            "vertical", "keyword_category", "device", "match_type",
            "user_segment", "position", "hour", "dow"]
CVR_KEYS = ["keyword_id", "campaign_id", "user_id", "vertical",
            "keyword_category", "advertiser_id"]
CROSS_KEYS = [("keyword_id", "device"), ("keyword_id", "position"),
              ("campaign_id", "device"), ("campaign_id", "position"),
              ("user_id", "vertical"), ("vertical", "device"),
              ("vertical", "position"), ("keyword_category", "device"),
              ("user_segment", "device"), ("match_type", "position"),
              ("keyword_category", "position"), ("campaign_id", "hour"),
              ("user_segment", "vertical"), ("device", "hour"),
              ("advertiser_id", "device")]


def build_features_spark(spark, raw_path: str, out_path: str,
                         history_fraction: float = 0.40, alpha: float = 20.0,
                         position_decay: float = 0.72):
    """Build the wide feature table in Spark and write it to Parquet.

    Mirrors adrank.features.engineering: an early HISTORY window estimates
    Bayesian-smoothed per-entity CTR/CVR; those are broadcast-joined onto the
    later MODELING window. Broadcast joins keep the small aggregate tables on
    the driver side so the 100M-row fact table is never shuffled by key.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.functions import broadcast

    df = spark.read.parquet(raw_path)
    max_day = df.agg(F.max("day")).first()[0]
    hist_cutoff = int(round((max_day + 1) * history_fraction))
    history = df.filter(F.col("day") < hist_cutoff)
    model_df = df.filter(F.col("day") >= hist_cutoff)

    g = history.agg(
        F.avg("clicked").alias("global_ctr"),
        (F.sum("converted") / F.sum("clicked")).alias("global_cvr")).first()
    global_ctr, global_cvr = float(g["global_ctr"]), float(g["global_cvr"])

    out = model_df

    # request-time transforms
    out = (out
           .withColumn("examine_prior", F.pow(F.lit(position_decay), F.col("position") - 1))
           .withColumn("log_bid", F.log1p("bid"))
           .withColumn("inv_position", 1.0 / F.col("position"))
           .withColumn("hour_sin", F.sin(2 * 3.141592653589793 * F.col("hour") / 24))
           .withColumn("hour_cos", F.cos(2 * 3.141592653589793 * F.col("hour") / 24)))

    def smoothed(num, den, prior):
        return (num + F.lit(alpha) * F.lit(prior)) / (den + F.lit(alpha))

    # single-key CTR
    for key in CTR_KEYS:
        agg = (history.groupBy(key)
               .agg(F.count(F.lit(1)).alias("_imp"), F.sum("clicked").alias("_clk"))
               .withColumn(f"ctr__{key}", smoothed(F.col("_clk"), F.col("_imp"), global_ctr))
               .withColumn(f"logimp__{key}", F.log1p("_imp"))
               .select(key, f"ctr__{key}", f"logimp__{key}"))
        out = out.join(broadcast(agg), on=key, how="left")

    # single-key CVR
    clicked = history.filter(F.col("clicked") == 1)
    for key in CVR_KEYS:
        agg = (clicked.groupBy(key)
               .agg(F.count(F.lit(1)).alias("_clk"), F.sum("converted").alias("_cnv"))
               .withColumn(f"cvr__{key}", smoothed(F.col("_cnv"), F.col("_clk"), global_cvr))
               .select(key, f"cvr__{key}"))
        out = out.join(broadcast(agg), on=key, how="left")

    # cross-key CTR
    for k1, k2 in CROSS_KEYS:
        name = f"{k1}_X_{k2}"
        agg = (history.groupBy(k1, k2)
               .agg(F.count(F.lit(1)).alias("_imp"), F.sum("clicked").alias("_clk"))
               .withColumn(f"ctr__{name}", smoothed(F.col("_clk"), F.col("_imp"), global_ctr))
               .select(k1, k2, f"ctr__{name}"))
        out = out.join(broadcast(agg), on=[k1, k2], how="left")

    # cold-start entities -> global priors
    ctr_cols = [c for c in out.columns if c.startswith("ctr__")]
    cvr_cols = [c for c in out.columns if c.startswith("cvr__")]
    out = out.fillna(global_ctr, subset=ctr_cols).fillna(global_cvr, subset=cvr_cols)

    out.write.mode("overwrite").parquet(out_path)
    return out_path


def _main():  # pragma: no cover - Spark entrypoint
    from pyspark.sql import SparkSession

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--history-fraction", type=float, default=0.40)
    ap.add_argument("--alpha", type=float, default=20.0)
    args = ap.parse_args()

    spark = (SparkSession.builder.appName("adrank-features").getOrCreate())
    path = build_features_spark(spark, args.raw, args.out,
                                args.history_fraction, args.alpha)
    print(f"wrote {path}")
    spark.stop()


if __name__ == "__main__":  # pragma: no cover
    _main()
