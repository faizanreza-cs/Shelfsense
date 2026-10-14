"""Tools the Insights Copilot can call. Every number in an answer comes from one of these.

Model metrics come from ``results.json`` (written by ``shelfsense build``). Recommendations use
the fitted recommender saved next to it. Product and brand insights, monitoring and SQL read the
warehouse (DuckDB or Snowflake).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import pandas as pd

from shelfsense import pipeline
from shelfsense.agent.sql_guard import UnsafeSQLError, check_result, scrub_error, validate
from shelfsense.config import get_settings
from shelfsense.queries import MIN_USERS, run_saved, saved_queries
from shelfsense.warehouse import get_warehouse

_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")  # item and shopper ids are embedded in SQL: strict


def _ok(rows: list[dict[str, Any]] | pd.DataFrame, note: str = "", **extra: Any) -> dict:
    if isinstance(rows, pd.DataFrame):
        rows = json.loads(rows.to_json(orient="records", date_format="iso", double_precision=4))
    return {"rows": rows, "row_count": len(rows), "note": note, **extra}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _products() -> pd.DataFrame:
    return get_warehouse().query(
        "SELECT item_id, title, brand, category, price FROM products ORDER BY item_id"
    )


def _resolve(item: str) -> dict[str, Any]:
    """A product from its id (P0015) or a unique part of its title."""
    p = _products()
    key = item.strip()
    hit = p[p["item_id"].str.upper() == key.upper()]
    if hit.empty:
        hit = p[p["title"].str.contains(key, case=False, regex=False)]
    if hit.empty:
        raise ValueError(f"no product matches {item!r}")
    if len(hit) > 1:
        names = "; ".join(f"{r.item_id} {r.title}" for r in hit.head(5).itertuples())
        raise ValueError(f"{len(hit)} products match {item!r} ({names}); use an item_id")
    return dict(hit.iloc[0])


def _with_products(ids: list[str], extra: dict[str, list[Any]] | None = None) -> pd.DataFrame:
    p = _products().set_index("item_id")
    out = p.loc[[i for i in ids if i in p.index]].reset_index()
    for col, values in (extra or {}).items():
        out[col] = values[: len(out)]
    return out


# --------------------------------------------------------------------------- tools


def t_list_saved_queries() -> dict:
    rows = [
        {"name": q.name, "title": q.title, "description": q.description}
        for q in saved_queries().values()
    ]
    return _ok(rows, note=f"Saved queries; rows keep >= {MIN_USERS} distinct shoppers.")


def t_run_saved_query(name: str) -> dict:
    q = saved_queries().get(name)
    if q is None:
        raise ValueError(f"unknown query {name!r}; choose from {', '.join(saved_queries())}")
    df = run_saved(name, get_warehouse())
    return _ok(df, note=f"{q.title}: {q.description}", query=name)


def t_run_sql(sql: str) -> dict:
    s = get_settings()
    safe = validate(sql, s.sql_row_limit)
    try:
        df = get_warehouse().query(safe, timeout=s.sql_timeout)
    except Exception as e:  # engine errors can quote data values; mask them for the model
        raise ValueError(f"query failed: {scrub_error(str(e))}") from None
    check_result(df)
    return _ok(df, note="Read-only query, row-capped.", sql=safe)


def t_recommender_metrics() -> dict:
    r = pipeline.load_results()["recommender"]
    weights = sorted(r["reranker_weights"].items(), key=lambda kv: -abs(kv[1]))[:4]
    note = (
        f"Offline test: models fitted on purchases before {r['cutoff']}, scored on what "
        f"{r['test_shoppers']:,} shoppers bought for the first time in the next "
        f"{r['horizon_days']} days. recall_at_10 = share of those products in the top 10; "
        "coverage = share of the catalogue ever recommended. 'hybrid' re-ranks the other models' "
        "candidates with a logistic regression; '+ rules' adds the business rules "
        f"{json.dumps(r['rules'])}. Strongest re-ranker inputs: "
        + ", ".join(f"{k} ({v:+.2f})" for k, v in weights)
        + "."
    )
    return _ok(r["models"], note=note)


def t_recommend_for_shopper(user_id: str, k: int = 10, apply_rules: bool = True) -> dict:
    if not _ID.match(user_id):
        raise ValueError("user_id must be an id like C000123")
    m = pipeline.load_model()
    rec, f = m["recommender"], m["fitted"]
    k = max(1, min(int(k), 20))
    if user_id not in f.users:
        top = list(rec.items[(-f.pop).argsort()[:k]])
        return _ok(_with_products(top), note=f"{user_id} has no purchases: best sellers instead.")
    rules = m["rules"] if apply_rules else None
    ids = rec.recommend(f, [user_id], "hybrid", k=k, rules=rules)[user_id]
    bought = int(f.X[f.users.get_loc(user_id)].sum())
    note = (
        f"Top {len(ids)} products {user_id} hasn't bought, from {bought} past products "
        "(hybrid recommender" + (f", rules {json.dumps(rules)})." if rules else ", no rules).")
    )
    return _ok(_with_products(ids, {"rank": list(range(1, len(ids) + 1))}), note=note)


def t_similar_products(item: str, k: int = 10) -> dict:
    prod = _resolve(item)
    m = pipeline.load_model()
    pairs = m["recommender"].similar(m["fitted"], prod["item_id"], max(1, min(int(k), 20)))
    ids = [i for i, _ in pairs]
    out = _with_products(ids, {"similarity": [s for _, s in pairs]})
    note = (
        f"Shoppers who bought {prod['title']} ({prod['item_id']}) also bought these: cosine "
        "similarity of the two products' buyer lists (1 = same buyers)."
    )
    return _ok(out, note=note)


def t_product_insights(item: str) -> dict:
    prod = _resolve(item)
    wh, iid = get_warehouse(), prod["item_id"]
    if not _ID.match(iid):
        raise ValueError(f"unexpected item id {iid!r}")
    aspects = wh.query(
        "SELECT aspect, mentions, positive, negative, net_sentiment FROM product_aspects "
        f"WHERE item_id = '{iid}' ORDER BY mentions DESC"
    )
    cat = wh.query(
        "SELECT a.aspect, ROUND((SUM(a.positive) - SUM(a.negative)) / SUM(a.mentions), 3) AS v "
        "FROM product_aspects a JOIN products p ON p.item_id = a.item_id "
        f"WHERE p.category = (SELECT category FROM products WHERE item_id = '{iid}') "
        "GROUP BY a.aspect"
    ).set_index("aspect")["v"]
    aspects["category_net_sentiment"] = aspects["aspect"].map(cat)
    alerts = wh.query(
        "SELECT aspect, month, mentions, negative_share, baseline_negative_share, z "
        f"FROM aspect_alerts WHERE item_id = '{iid}' ORDER BY month"
    )
    examples = wh.query(
        "SELECT r.review_id, a.aspect, a.label, r.rating, a.month, LEFT(r.text, 220) AS text "
        "FROM review_aspects a JOIN reviews r ON r.review_id = a.review_id "
        f"WHERE a.item_id = '{iid}' "
        "QUALIFY ROW_NUMBER() OVER (PARTITION BY a.aspect, a.label ORDER BY r.review_ts DESC) <= 2 "
        "ORDER BY a.aspect, a.label"
    )
    note = (
        f"{prod['title']} ({iid}, {prod['brand']}, {prod['category']}). Net sentiment = "
        "(positive - negative) / mentions, from the review aspect model; compare with the "
        "category's. Cite review ids when quoting examples."
    )
    if len(alerts):
        note += f" Monitoring raised {len(alerts)} alert(s) for this product: see 'alerts'."
    return _ok(
        aspects,
        note=note,
        product=prod,
        alerts=json.loads(alerts.to_json(orient="records")),
        examples=json.loads(examples.to_json(orient="records")),
    )


def t_brand_scorecard(brand: str | None = None) -> dict:
    df = run_saved("aspect_scorecard_by_brand", get_warehouse())
    store = df.groupby("aspect").apply(
        lambda g: round(float((g["net_sentiment"] * g["mentions"]).sum() / g["mentions"].sum()), 3),
        include_groups=False,
    )
    if brand:
        rows = df[df["brand"].str.lower() == brand.strip().lower()].copy()
        if rows.empty:
            raise ValueError(
                f"unknown brand {brand!r}; brands: {', '.join(sorted(df['brand'].unique()))}"
            )
        rows["store_net_sentiment"] = rows["aspect"].map(store)
        rows["gap_vs_store"] = (rows["net_sentiment"] - rows["store_net_sentiment"]).round(3)
        note = (
            f"{rows['brand'].iloc[0]} by aspect vs the store average (net sentiment = (positive "
            "- negative) / mentions). The most negative gap is the brand's weakest point."
        )
        return _ok(rows.sort_values("gap_vs_store"), note=note)
    df["weighted"] = df["net_sentiment"] * df["mentions"]
    g = df.groupby("brand")
    summary = pd.DataFrame({
        "mentions": g["mentions"].sum(),
        "net_sentiment": (g["weighted"].sum() / g["mentions"].sum()).round(3),
        "weakest_aspect": g.apply(lambda x: x.loc[x["net_sentiment"].idxmin(), "aspect"],
                                  include_groups=False),
    }).reset_index().sort_values("net_sentiment", ascending=False)  # fmt: skip
    note = "Net sentiment over all aspects, weighted by mentions, per brand, best first."
    return _ok(summary, note=note)


def t_monitoring_report(month: str | None = None) -> dict:
    wh = get_warehouse()
    months = wh.query("SELECT DISTINCT month FROM drift_monthly ORDER BY month")["month"].tolist()
    if not months:
        raise ValueError("no monitoring results; run `shelfsense build`")
    month = month or months[-1]
    if month not in months:
        raise ValueError(f"no monitoring for {month!r}; months {months[0]} to {months[-1]}")
    drift = wh.query(
        f"SELECT metric, psi, status, rows FROM drift_monthly WHERE month = '{month}' "
        "ORDER BY psi DESC"
    )
    alerts = wh.query(
        "SELECT item_id, title, aspect, month, mentions, negative_share, "
        "baseline_negative_share, z FROM aspect_alerts ORDER BY month, z DESC"
    )
    res = pipeline.load_results()["monitoring"]
    note = (
        f"Store-level drift for {month}: PSI of each distribution against the first "
        f"{res['reference_months']} months (< 0.1 stable, 0.1-0.25 watch, >= 0.25 drift). "
        f"Product alerts: a product-aspect whose share of negative mentions jumped against its "
        f"history ({len(alerts)} alert-months in total, all months)."
    )
    extra: dict[str, Any] = {"month": month, "alerts": json.loads(alerts.to_json(orient="records"))}
    if res.get("truth_check"):
        t = res["truth_check"]
        extra["truth_check"] = t
        note += (
            f" Simulator check: {t['reformulated_item']} was reformulated in "
            f"{t['true_change_month']}; first alert {t['first_alert_month']}, "
            f"{t['other_alerts']} other alerts."
        )
    return _ok(drift, note=note, **extra)


def t_aspect_model_metrics() -> dict:
    a = pipeline.load_results()["aspects"]
    rows = []
    for name, key in (("labeller", "labeller_f1"), ("student model", "student_f1")):
        if key in a:
            rows.append({"model": name, **a[key]})
    note = (
        f"Labeller '{a['labeller']}' labelled {a['labelled_reviews']:,} reviews; a TF-IDF + "
        f"logistic regression student learned from them and scored all {a['scored_reviews']:,}. "
        "F1 = macro-F1 over none/positive/negative per aspect, against the simulator's true labels"
        " (student scored on reviews it never saw)."
    )
    extra: dict[str, Any] = {}
    if a.get("perfect_labels_curve"):
        extra["perfect_labels_curve"] = a["perfect_labels_curve"]
        best = a["perfect_labels_curve"][-1]
        note += (
            f" With perfect labels the same student reaches {best['f1']:.3f} at "
            f"{best['labels']:,} labels: label quality, not the model, is the limit."
        )
    if a.get("quality_correlation"):
        extra["quality_correlation"] = a["quality_correlation"]
    return _ok(rows, note=note, **extra)


# --------------------------------------------------------------------------- registry

TOOLS: dict[str, tuple[Callable[..., dict], dict[str, Any]]] = {
    "list_saved_queries": (t_list_saved_queries, {
        "description": "List the saved queries (name, title, description).",
        "input_schema": {"type": "object", "properties": {}},
    }),
    "run_saved_query": (t_run_saved_query, {
        "description": "Run a saved query: aspect_scorecard_by_brand (net sentiment per brand and aspect), category_aspect_trend (monthly sentiment by category and aspect), bought_together (product pairs with lift), repeat_purchase_by_category (share of shoppers who re-buy), monthly_sales (orders, revenue, shoppers by month and category). Rows cover at least 50 shoppers.",
        "input_schema": {"type": "object", "properties": {
            "name": {"type": "string", "enum": sorted(saved_queries())}},
            "required": ["name"]},
    }),
    "run_sql": (t_run_sql, {
        "description": "Run one read-only Snowflake SQL SELECT when no other tool fits. Tables: products(item_id, title, brand, category, flavor, price, margin_pct, in_stock), orders(order_id, user_id, item_id, order_ts, price), reviews(review_id, user_id, item_id, rating, title, text, review_ts, verified_purchase), review_aspects(review_id, item_id, month, aspect, label), product_aspects(item_id, aspect, positive, negative, mentions, net_sentiment), aspect_monthly(item_id, aspect, month, positive, negative, mentions, net_sentiment), drift_monthly(month, metric, psi, status, rows), aspect_alerts(item_id, aspect, month, mentions, negative_share, baseline_negative_share, z, title, brand), recommendations(user_id, rank, item_id). Aspects: taste, digestion, effectiveness, value, packaging; labels: positive, negative. Never return user_id: count or group by it instead. Results are capped at 200 rows.",
        "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}},
                         "required": ["sql"]},
    }),
    "recommender_metrics": (t_recommender_metrics, {
        "description": "Offline evaluation of the product recommenders: recall@10, NDCG@10, catalogue coverage and average margin for popularity, item-item, SVD, content, the learned hybrid and the hybrid with business rules.",
        "input_schema": {"type": "object", "properties": {}},
    }),
    "recommend_for_shopper": (t_recommend_for_shopper, {
        "description": "Next products to recommend to one shopper (by user_id, e.g. C000123), from the hybrid recommender, with business rules (in stock, max 3 per brand, margin boost) unless apply_rules is false.",
        "input_schema": {"type": "object", "properties": {
            "user_id": {"type": "string"},
            "k": {"type": "integer", "minimum": 1, "maximum": 20},
            "apply_rules": {"type": "boolean"}},
            "required": ["user_id"]},
    }),
    "similar_products": (t_similar_products, {
        "description": "Products most often bought by the same shoppers as a given product (item_id like P0015, or a unique part of its title).",
        "input_schema": {"type": "object", "properties": {
            "item": {"type": "string"},
            "k": {"type": "integer", "minimum": 1, "maximum": 20}},
            "required": ["item"]},
    }),
    "product_insights": (t_product_insights, {
        "description": "What reviewers say about one product (item_id like P0015, or a unique part of its title): net sentiment per aspect vs its category, monitoring alerts, and example reviews with ids to quote.",
        "input_schema": {"type": "object", "properties": {"item": {"type": "string"}},
                         "required": ["item"]},
    }),
    "brand_scorecard": (t_brand_scorecard, {
        "description": "Review sentiment by brand. With a brand: its net sentiment per aspect vs the store average. Without: one row per brand with overall sentiment and weakest aspect.",
        "input_schema": {"type": "object", "properties": {"brand": {"type": "string"}}},
    }),
    "monitoring_report": (t_monitoring_report, {
        "description": "Model and data monitoring: store-level drift (PSI) of orders, ratings and predicted review aspects for a month (default: latest), plus product-level alerts when a product's negative reviews on an aspect jump.",
        "input_schema": {"type": "object", "properties": {
            "month": {"type": "string", "description": "YYYY-MM; default latest"}}},
    }),
    "aspect_model_metrics": (t_aspect_model_metrics, {
        "description": "Quality of the review aspect model: macro-F1 of the labeller and of the student model per aspect, the learning curve with perfect labels, and how well product scores track product quality.",
        "input_schema": {"type": "object", "properties": {}},
    }),
}  # fmt: skip


def tool_definitions() -> list[dict[str, Any]]:
    return [{"name": name, **spec} for name, (_, spec) in TOOLS.items()]


def execute(name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Run a tool; returns (json_string, is_error). Errors go back to the model, not up."""
    if name not in TOOLS:
        return json.dumps({"error": f"unknown tool {name!r}"}), True
    fn, _ = TOOLS[name]
    if "_invalid_arguments" in args:  # the model's arguments weren't valid JSON
        return json.dumps({"error": "the tool arguments were not valid JSON; call it again"}), True
    try:
        return json.dumps(fn(**args), default=str), False
    except (UnsafeSQLError, ValueError, TypeError, KeyError, FileNotFoundError) as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), True
    except Exception as e:  # warehouse errors and the like: tell the model, keep the run alive
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), True
