"""Guardrails for model-written SQL (Snowflake dialect).

The agent may run ad-hoc SQL, but only:

* one read-only SELECT (CTEs and UNION allowed),
* against the allow-listed tables,
* with no table functions, LATERAL, or file / network / session functions,
* never returning shopper ids: answers come from aggregates, never individuals. ``user_id``
  may be counted (``COUNT(DISTINCT user_id)``), grouped or partitioned by, joined on
  (``a.user_id = b.user_id``) and passed through CTEs and subqueries unchanged, but never put
  in a function, compared with anything else, or returned. The check follows the column
  through aliases, CTEs, subqueries, unions and ``*`` (sqlglot resolves every column first).
  ``check_result`` also rejects any result whose values look like a shopper id.
* with a row cap.

What it doesn't do: enforce a minimum number of shoppers per output row on ad-hoc SQL (the
saved queries do), or stop someone who already knows an id from filtering on aggregates over
many queries. For that, use an engine-level control such as a Snowflake aggregation policy
(``snowflake/setup.sql``).

This is the first line of defence. The engine is the second: DuckDB runs read-only with
external access disabled, and on Snowflake the agent should use the read-only role in
``snowflake/setup.sql``.
"""

from __future__ import annotations

import re

import pandas as pd
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope

from shelfsense.warehouse import SCHEMA, TABLES

ALLOWED_TABLES = frozenset(TABLES)
ID_COLUMN = "user_id"
SHOPPER_TABLES = frozenset({"orders", "reviews", "recommendations"})
BLOCKED_FUNCTIONS = frozenset(
    {
        "read_csv", "read_csv_auto", "read_parquet", "parquet_scan", "read_json",
        "read_json_auto", "read_json_objects", "read_ndjson", "read_text", "read_blob", "glob",
        "sniff_csv", "query", "query_table", "getenv", "json_execute_serialized_sql",
        "result_scan", "infer_schema", "get_ddl", "system$whitelist", "flatten", "columns",
    }
)  # fmt: skip


USER_ID = re.compile(r"c\d{6}", re.I)  # format of the synthetic shopper ids
_COUNTS = (exp.Count, exp.ApproxDistinct)
_SCHEMA: dict[str, object] = {t: dict.fromkeys(cols, "VARCHAR") for t, cols in SCHEMA.items()}


class UnsafeSQLError(ValueError):
    pass


def _source(scope: Scope, alias: str) -> exp.Table | Scope | None:
    """What a column qualifier points at: a table, a CTE/subquery scope, or an outer scope's."""
    node: Scope | None = scope
    while node is not None:
        if alias in node.sources:
            return node.sources[alias]
        node = node.parent
    return None


def _check_shopper_ids(tree: exp.Expression) -> None:
    """Follow ``user_id`` through every scope; see the module docstring for what's allowed."""
    try:
        q = qualify(tree.copy(), schema=_SCHEMA, dialect="snowflake", quote_identifiers=False)
        scopes = traverse_scope(q)
    except SqlglotError as e:
        raise UnsafeSQLError(f"could not resolve the query's columns: {e}") from e
    by_expr = {id(sc.expression): sc for sc in scopes}
    carries: dict[int, list[bool]] = {}  # per scope: which outputs carry shopper ids

    def is_id(scope: Scope, col: exp.Column) -> bool:
        src = _source(scope, col.table)
        if isinstance(src, exp.Table):
            return src.name.lower() in SHOPPER_TABLES and col.name.lower() == ID_COLUMN
        if isinstance(src, Scope) and isinstance(src.expression, exp.Query):
            names = [n.lower() for n in src.expression.named_selects]
            flags = carries[id(src)]
            return any(f and n == col.name.lower() for n, f in zip(names, flags, strict=False))
        return False

    # pass 1, inner scopes first: find the id columns and which outputs pass them through
    found: list[exp.Column] = []
    for scope in scopes:
        query = scope.expression
        if isinstance(query, exp.SetOperation):  # an output carries ids if any branch's does
            left, right = (carries[id(sc)] for sc in scope.set_operation_scopes)
            carries[id(scope)] = [a or b for a, b in zip(left, right, strict=False)]
            continue
        flags = [False] * len(query.selects) if isinstance(query, exp.Select) else []
        for col in scope.columns:
            if not is_id(scope, col):
                continue
            found.append(col)
            parent = col.parent
            if (
                isinstance(query, exp.Select)
                and isinstance(parent, exp.Alias)
                and (parent.parent is query)
            ):
                flags[query.selects.index(parent)] = True
        carries[id(scope)] = flags
    id_nodes = {id(c) for c in found}

    # pass 2: every other use must be a count, a grouping key or a join on another user_id
    for col in found:
        node: exp.Expression = col
        parent = col.parent
        while isinstance(parent, exp.Paren):
            node, parent = parent, parent.parent
        through = isinstance(parent, exp.Alias) and isinstance(parent.parent, exp.Select)
        counted = isinstance(parent, _COUNTS) or (
            isinstance(parent, exp.Distinct) and isinstance(parent.parent, _COUNTS)
        )
        keyed = isinstance(parent, exp.Group | exp.Ordered | exp.Is) or (
            isinstance(parent, exp.Window) and node is not parent.this
        )
        joined = isinstance(parent, exp.EQ | exp.NEQ) and any(
            id(side) in id_nodes for side in (parent.left, parent.right) if side is not node
        )
        if isinstance(parent, exp.In) and node is parent.this:
            sub = parent.args.get("query")
            sub = sub.this if isinstance(sub, exp.Subquery) else sub
            inner = by_expr.get(id(sub))
            joined = inner is not None and any(carries[id(inner)])
        if not (through or counted or keyed or joined):
            raise UnsafeSQLError(
                "user_id may only be counted (COUNT(DISTINCT user_id)), grouped by or joined "
                "on another user_id; it can't be returned, transformed or compared with values"
            )
    # a star left after qualify sits inside a function (OBJECT_CONSTRUCT(*), ...): whole rows
    for star in q.find_all(exp.Star):
        parent = star.parent.parent if isinstance(star.parent, exp.Column) else star.parent
        if not isinstance(parent, _COUNTS):
            raise UnsafeSQLError("* is only allowed in COUNT(*) on shopper tables; list columns")
    if any(carries[id(scopes[-1])]):
        raise UnsafeSQLError(
            "queries may not return shopper ids (user_id); aggregate instead, e.g. "
            "COUNT(DISTINCT user_id)"
        )


def validate(sql: str, row_limit: int = 200) -> str:
    """Return a safe, row-capped version of ``sql`` (Snowflake dialect) or raise."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="snowflake") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQLError(f"could not parse SQL: {e}") from e
    if len(statements) != 1:
        raise UnsafeSQLError("exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, exp.Select | exp.Union | exp.Intersect | exp.Except):
        raise UnsafeSQLError(f"only SELECT queries are allowed (got {tree.key.upper()})")
    for node in tree.walk():
        if isinstance(node, exp.Insert | exp.Update | exp.Delete | exp.Drop | exp.Create
                      | exp.Alter | exp.Command | exp.Copy | exp.Merge):  # fmt: skip
            raise UnsafeSQLError("write and DDL statements are not allowed")
        if isinstance(node, exp.Lateral):
            raise UnsafeSQLError("LATERAL is not allowed")

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    used = set()
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if isinstance(table.this, exp.Func) or not name:
            raise UnsafeSQLError("table functions are not allowed")
        if table.args.get("db") or table.args.get("catalog"):
            raise UnsafeSQLError("schema-qualified table names are not allowed")
        if name not in ALLOWED_TABLES and name not in cte_names:
            raise UnsafeSQLError(f"table {table.name!r} is not allowed")
        used.add(name)

    for func in tree.find_all(exp.Func):
        fname = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()
        if fname in BLOCKED_FUNCTIONS:
            raise UnsafeSQLError(f"function {fname!r} is not allowed")

    if any(isinstance(n, exp.Pivot) for n in tree.walk()):
        raise UnsafeSQLError("PIVOT and UNPIVOT are not allowed; use CASE WHEN instead")
    if used & SHOPPER_TABLES:
        _check_shopper_ids(tree)

    limit = tree.args.get("limit")
    options = limit.args.get("limit_options") if limit is not None else None
    plain = isinstance(tree, exp.Select) and (
        limit is None
        or (
            isinstance(limit, exp.Limit)
            and isinstance(limit.expression, exp.Literal)
            and not limit.expression.is_string
            and not (options and options.args.get("percent"))
        )
    )
    if plain:
        n = int(limit.expression.name) if limit is not None else row_limit
        # capped in place, so a top-level ORDER BY keeps working on every engine
        return tree.limit(min(n, row_limit)).sql(dialect="snowflake")
    # PERCENT, TOP/FETCH variants, set operations: wrap with an outer cap.
    # shortcut: Snowflake may not keep an inner ORDER BY through the wrapper; order rows in the
    # caller if a set-operation query needs ordering.
    return exp.select("*").from_(tree.subquery("q")).limit(row_limit).sql(dialect="snowflake")


_QUOTED = re.compile(r"'[^']*'")


def scrub_error(message: str) -> str:
    """Mask quoted values in an engine error (e.g. a failed cast quotes the value it saw)."""
    return _QUOTED.sub("'...'", message)


def check_result(df: pd.DataFrame) -> None:
    """Reject results that contain shopper ids (a second check after ``validate``)."""
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_string_dtype(s) or s.dtype == object):
            continue
        if s.dropna().astype(str).str.contains(USER_ID).any():
            raise UnsafeSQLError(
                f"column {col!r} contains shopper ids; return aggregates (counts, sums) instead"
            )
