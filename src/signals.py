"""Run the SQL signal pipeline in DuckDB."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, SQL = ROOT / "data" / "raw", ROOT / "sql"
T0, END_MONTH, SHRINK_K = 12, 22, 20


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for t in ["customers", "channel_monthly", "balance_monthly", "txn_monthly"]:
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{RAW / (t + '.parquet')}')")
    return con


def build_signals(con, t0: int = T0, end_month: int = END_MONTH, k: int = SHRINK_K) -> pd.DataFrame:
    p = dict(t0=t0, end_month=end_month, k=k)
    q = lambda f: (SQL / f).read_text().format(**p)
    con.execute("CREATE OR REPLACE TEMP TABLE channel_signal AS " + q("01_channel_signal.sql"))
    con.execute("CREATE OR REPLACE TEMP TABLE balance_trend  AS " + q("02_balance_trend.sql"))
    con.execute("CREATE OR REPLACE TEMP TABLE txn_velocity   AS " + q("03_txn_velocity.sql"))
    con.execute("CREATE OR REPLACE TEMP TABLE outcome        AS " + q("04_outcome.sql"))
    return con.execute(q("05_signals.sql")).df()
