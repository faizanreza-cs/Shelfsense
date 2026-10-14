import pandas as pd
import pytest

from shelfsense import warehouse
from shelfsense.agent.sql_guard import UnsafeSQLError, check_result, scrub_error, validate
from shelfsense.agent.tools import execute
from shelfsense.queries import MIN_USERS, run_saved, saved_queries
from shelfsense.warehouse import SCHEMA, SnowflakeWarehouse, to_duckdb


@pytest.mark.parametrize("name", sorted(saved_queries()))
def test_saved_queries_run_and_respect_the_threshold(wh, name):
    df = run_saved(name, wh)
    assert len(df) > 0
    assert (df["users"] >= MIN_USERS).all()


def test_threshold_is_enforced_even_if_a_query_forgets():
    class Fake:
        name = "fake"

        def query(self, sql, timeout=None):
            return pd.DataFrame({"users": [60, 49], "x": [1, 2]})

    assert run_saved("monthly_sales", Fake())["users"].tolist() == [60]


def test_schema_matches_the_warehouse(wh):
    for table, cols in SCHEMA.items():
        assert tuple(wh.query(f"SELECT * FROM {table} LIMIT 1").columns) == cols, table


def test_snowflake_functions_translate_to_duckdb():
    sql = to_duckdb("SELECT IFF(a > 1, 1, 0), DATEADD(day, 7, d), COUNT_IF(a > 1) FROM t")
    assert "CASE WHEN" in sql and "INTERVAL" in sql


def test_engine_blocks_file_access(wh, tmp_path):
    secret = tmp_path / "secret.csv"
    secret.write_text("token\nTOPSECRET\n")
    with pytest.raises(Exception) as err:
        wh.query(f"SELECT * FROM read_csv('{secret.as_posix()}')")
    assert "TOPSECRET" not in str(err.value)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE products",
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM truth_review_labels",
        "SELECT 1; SELECT 2",
        "SELECT * FROM information_schema.tables",
    ],
)
def test_guard_blocks(sql):
    with pytest.raises(UnsafeSQLError):
        validate(sql)


# ways to get shopper ids out: wrapped, aliased, through CTEs, unions, stars, filters, errors
LEAKS = [
    "SELECT user_id, COUNT(*) FROM orders GROUP BY 1",
    "SELECT UPPER(MAX(user_id)) AS id FROM orders GROUP BY user_id",
    "SELECT LISTAGG(user_id, ',') FROM reviews",
    "WITH t AS (SELECT user_id AS u FROM orders) SELECT REVERSE(u) FROM t",
    "WITH t AS (SELECT user_id FROM recommendations) SELECT * FROM t",
    "SELECT * FROM reviews",
    "SELECT x FROM (SELECT user_id FROM orders) AS s(x)",
    "SELECT u FROM (SELECT 'x' AS u UNION ALL SELECT user_id FROM orders)",
    "SELECT COUNT(*) FROM orders WHERE user_id = 'C000123'",
    "SELECT COUNT(*) FROM orders WHERE user_id LIKE 'C0001%'",
    "SELECT TO_JSON(OBJECT_CONSTRUCT(*)) FROM orders",
    "SELECT o FROM orders o",
    "SELECT * FROM (SELECT user_id, price FROM orders) "
    "PIVOT (SUM(price) FOR user_id IN ('C000001'))",
]
ALLOWED = [
    "SELECT COUNT(DISTINCT user_id) AS shoppers FROM orders",
    "SELECT n, COUNT(*) AS shoppers FROM (SELECT COUNT(*) AS n FROM orders GROUP BY user_id) "
    "GROUP BY n ORDER BY n",
    "SELECT COUNT(DISTINCT o.user_id) AS reviewers FROM orders o JOIN reviews r "
    "ON r.user_id = o.user_id AND r.item_id = o.item_id",
    "SELECT p.brand, COUNT(DISTINCT r.user_id) AS reviewers FROM reviews r "
    "JOIN products p ON p.item_id = r.item_id GROUP BY p.brand",
    "SELECT item_id, COUNT(*) AS times FROM recommendations WHERE rank = 1 GROUP BY item_id",
]


@pytest.mark.parametrize("sql", LEAKS)
def test_shopper_ids_cant_leave(sql):
    with pytest.raises(UnsafeSQLError):
        validate(sql)


@pytest.mark.parametrize("sql", ALLOWED)
def test_shopper_ids_can_be_counted_grouped_and_joined(wh, sql):
    df = wh.query(validate(sql))
    check_result(df)
    assert len(df) > 0


def test_result_check_catches_ids_in_any_case():
    for value in ("C000123", "c000123", "ids: C000001,C000002"):
        with pytest.raises(UnsafeSQLError):
            check_result(pd.DataFrame({"x": [value]}))
    check_result(pd.DataFrame({"x": ["taste", "value"], "n": [1, 2]}))


def test_engine_errors_dont_echo_values():
    assert scrub_error("Could not convert string 'C000123' to INT32") == (
        "Could not convert string '...' to INT32"
    )
    out, is_error = execute("run_sql", {"sql": "SELECT CAST(title AS INT) FROM products"})
    assert is_error and "convert" in out.lower() and "Protein" not in out


def test_long_queries_are_stopped(wh):
    cross = "SELECT COUNT(*) FROM orders a JOIN orders b ON a.user_id <> b.user_id"
    with pytest.raises(TimeoutError):
        wh.query(cross, timeout=0.5)
    assert len(wh.query("SELECT 1 AS x")) == 1  # the connection still works


def test_snowflake_connects_once(monkeypatch):
    made = []

    class Fake:
        name = "snowflake"

        def __init__(self):
            made.append(self)

    monkeypatch.setenv("SHELFSENSE_BACKEND", "snowflake")
    monkeypatch.setattr(warehouse, "SnowflakeWarehouse", Fake)
    warehouse.reset_cache()
    try:
        assert warehouse.get_warehouse() is warehouse.get_warehouse() and len(made) == 1
    finally:
        warehouse.reset_cache()


def test_snowflake_backend_sends_sql_unchanged_and_lowercases_columns():
    seen = []

    class Cursor:
        def execute(self, sql, timeout=None):
            seen.append(sql)

        def fetch_pandas_all(self):
            return pd.DataFrame({"BRAND": ["Solstice"], "N": [1]})

        def close(self):
            pass

    class Conn:
        def cursor(self):
            return Cursor()

    sql = "SELECT brand, COUNT_IF(in_stock) AS n FROM products GROUP BY brand"
    df = SnowflakeWarehouse(connection=Conn()).query(sql)
    assert seen == [sql] and list(df.columns) == ["brand", "n"]
