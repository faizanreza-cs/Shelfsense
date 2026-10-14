"""Run Snowflake SQL on DuckDB (local and CI) or on Snowflake itself.

All SQL in this repo is written once, in Snowflake's dialect. Locally it is translated to DuckDB
with sqlglot, so the same queries are tested in CI; on Snowflake they run unchanged.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import duckdb
import pandas as pd
import sqlglot

from shelfsense.config import Settings, get_settings

# tables the API, the agent and SQL may read, with their columns (truth stays out on purpose)
SCHEMA: dict[str, tuple[str, ...]] = {
    "products": ("item_id", "title", "brand", "category", "flavor", "price", "margin_pct",
                 "in_stock"),
    "orders": ("order_id", "user_id", "item_id", "order_ts", "price"),
    "reviews": ("review_id", "user_id", "item_id", "rating", "title", "text", "review_ts",
                "verified_purchase"),
    "review_aspects": ("review_id", "item_id", "month", "aspect", "label"),
    "product_aspects": ("item_id", "aspect", "positive", "negative", "mentions", "net_sentiment"),
    "aspect_monthly": ("item_id", "aspect", "month", "positive", "negative", "mentions",
                       "net_sentiment"),
    "drift_monthly": ("month", "metric", "psi", "status", "rows"),
    "aspect_alerts": ("item_id", "aspect", "month", "mentions", "negative_share",
                      "baseline_negative_share", "baseline_mentions", "z", "title", "brand"),
    "recommendations": ("user_id", "rank", "item_id"),
}  # fmt: skip
TABLES = tuple(SCHEMA)


class Warehouse(Protocol):
    name: str

    def query(self, sql: str, timeout: float | None = None) -> pd.DataFrame: ...


def to_duckdb(sql: str) -> str:
    """Translate one Snowflake statement to DuckDB."""
    return sqlglot.transpile(sql, read="snowflake", write="duckdb")[0]


class DuckDBWarehouse:
    name = "duckdb"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"No database at {self.path}. Run `shelfsense build` first.")
        # Read-only, and SQL can't touch files or the network: model-written queries see only
        # the tables, whatever functions they try (read_csv, LATERAL tricks, ...).
        self._con = duckdb.connect(
            str(self.path), read_only=True, config={"enable_external_access": False}
        )
        self._lock = threading.Lock()

    def query(self, sql: str, timeout: float | None = None) -> pd.DataFrame:
        """Run one query; ``timeout`` seconds stops it (a cross join can run for hours)."""
        with self._lock:
            cur = self._con.cursor()  # one cursor per call: safe across API threads
        timer = threading.Timer(timeout, cur.interrupt) if timeout else None
        if timer:
            timer.start()
        try:
            return cur.execute(to_duckdb(sql)).df()
        except duckdb.InterruptException as e:
            raise TimeoutError(f"query stopped after {timeout:g} s") from e
        finally:
            if timer:
                timer.cancel()
            cur.close()


def _snowflake_params() -> dict[str, Any]:
    """Connection settings from SNOWFLAKE_* variables (account, user, password or key, ...)."""
    keys = {
        "account": "SNOWFLAKE_ACCOUNT",
        "user": "SNOWFLAKE_USER",
        "password": "SNOWFLAKE_PASSWORD",
        "authenticator": "SNOWFLAKE_AUTHENTICATOR",
        "private_key_file": "SNOWFLAKE_PRIVATE_KEY_FILE",
        "role": "SNOWFLAKE_ROLE",
        "warehouse": "SNOWFLAKE_WAREHOUSE",
    }
    params: dict[str, Any] = {k: os.environ[v] for k, v in keys.items() if os.environ.get(v)}
    params["database"] = os.environ.get("SNOWFLAKE_DATABASE", "SHELFSENSE")
    params["client_session_keep_alive"] = True  # the API keeps one connection open
    params["schema"] = os.environ.get("SNOWFLAKE_SCHEMA", "STORE")
    missing = {"account", "user"} - params.keys()
    if missing:
        raise ValueError(f"set SNOWFLAKE_{'/SNOWFLAKE_'.join(sorted(m.upper() for m in missing))}")
    return params


class SnowflakeWarehouse:
    """Runs the SQL unchanged on Snowflake (``pip install 'shelfsense[snowflake]'``).

    Use a read-only role for the agent (see ``snowflake/setup.sql``).
    """

    name = "snowflake"

    def __init__(self, connection: Any | None = None) -> None:
        if connection is None:
            import snowflake.connector

            connection = snowflake.connector.connect(**_snowflake_params())
        self._con = connection

    def query(self, sql: str, timeout: float | None = None) -> pd.DataFrame:
        cur = self._con.cursor()
        try:
            cur.execute(sql, timeout=int(timeout) if timeout else None)
            df = cur.fetch_pandas_all()
        finally:
            cur.close()
        df.columns = [str(c).lower() for c in df.columns]  # Snowflake returns upper case
        return df

    def write(self, table: str, df: pd.DataFrame) -> int:
        """Create or replace ``table`` from a DataFrame. Returns rows written."""
        from snowflake.connector.pandas_tools import write_pandas

        out = df.copy()
        out.columns = [c.upper() for c in out.columns]  # unquoted identifiers, like the SQL
        ok, _, rows, _ = write_pandas(
            self._con,
            out,
            table.upper(),
            auto_create_table=True,
            overwrite=True,
            quote_identifiers=False,
            use_logical_type=True,
        )
        if not ok:
            raise RuntimeError(f"write_pandas failed for {table}")
        return int(rows)


@lru_cache(maxsize=4)
def _duckdb(path: str) -> DuckDBWarehouse:
    return DuckDBWarehouse(path)


@lru_cache(maxsize=1)
def _snowflake() -> SnowflakeWarehouse:
    # one connection for the process (logging in takes seconds); cursors are per query
    return SnowflakeWarehouse()


def get_warehouse(settings: Settings | None = None) -> Warehouse:
    s = settings or get_settings()
    if s.backend == "snowflake":
        return _snowflake()
    if s.backend != "duckdb":
        raise ValueError(f"unknown backend {s.backend!r} (duckdb | snowflake)")
    return _duckdb(str(s.db_path))


def reset_cache() -> None:
    """Drop cached connections, e.g. after `shelfsense build` replaced the database file."""
    _duckdb.cache_clear()
    _snowflake.cache_clear()
