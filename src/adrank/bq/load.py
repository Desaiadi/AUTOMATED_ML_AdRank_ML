"""Load processed tables into BigQuery (with a local fallback).

In production the wide feature table and scored predictions are loaded into
BigQuery for analyst access and repeatable backtests. This module uses
``google-cloud-bigquery`` when it's installed AND ``bigquery.enabled`` is true;
otherwise it degrades gracefully: it writes the same tables as Parquet under
``data/processed/bq_export/`` and logs the equivalent ``LOAD DATA`` DDL, so the
pipeline runs end-to-end without GCP credentials.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config
from ..utils.common import get_logger, timed

LOGGER = get_logger("adrank.bq")


def _emit_local(df: pd.DataFrame, table: str, export_dir: Path) -> str:
    export_dir.mkdir(parents=True, exist_ok=True)
    out = export_dir / f"{table}.parquet"
    df.to_parquet(out, index=False)
    LOGGER.info("[bq:local] wrote %s rows -> %s", f"{len(df):,}", out)
    LOGGER.info("[bq:local] equivalent: LOAD DATA INTO `%s.%s` FROM FILES "
                "(format='PARQUET', uris=['%s'])", "<dataset>", table, out)
    return str(out)


def load_table(cfg: Config, df: pd.DataFrame, table_key: str) -> dict:
    table = cfg.bigquery.tables[table_key]
    export_dir = Path(cfg.paths["processed_dir"]) / "bq_export"

    if not cfg.bigquery.get("enabled", False):
        path = _emit_local(df, table, export_dir)
        return {"backend": "local", "table": table, "path": path, "rows": int(len(df))}

    try:  # pragma: no cover - exercised only with GCP creds
        from google.cloud import bigquery

        client = bigquery.Client(project=cfg.bigquery.project)
        table_id = f"{cfg.bigquery.project}.{cfg.bigquery.dataset}.{table}"
        job = client.load_table_from_dataframe(
            df, table_id,
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"))
        job.result()
        LOGGER.info("[bq] loaded %s rows -> %s", f"{len(df):,}", table_id)
        return {"backend": "bigquery", "table": table_id, "rows": int(len(df))}
    except Exception as exc:  # pragma: no cover - fall back on any GCP error
        LOGGER.warning("[bq] BigQuery load failed (%s); falling back to local Parquet", exc)
        path = _emit_local(df, table, export_dir)
        return {"backend": "local_fallback", "table": table, "path": path, "rows": int(len(df))}


def load_processed(cfg: Config) -> dict:
    """Load the wide features and scored predictions to BigQuery / local."""
    proc_dir = Path(cfg.paths["processed_dir"])
    results = {}
    with timed("bq.load_features", LOGGER):
        feats = pd.read_parquet(proc_dir / "features.parquet")
        results["features"] = load_table(cfg, feats, "features")
    with timed("bq.load_predictions", LOGGER):
        scored = pd.read_parquet(proc_dir / "scored.parquet")
        results["predictions"] = load_table(cfg, scored, "predictions")
    return results
