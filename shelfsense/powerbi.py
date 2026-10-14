"""Power BI kit: export a star schema as CSV files for Power BI Desktop.

``shelfsense export-powerbi`` writes ``powerbi/data/*.csv``. Load them with Get data > Text/CSV,
create the relationships listed in ``powerbi/README.md``, paste the measures from
``powerbi/measures.dax`` and apply ``powerbi/theme.json``. Shopper-level rows never leave the
warehouse: orders and recommendations are exported as monthly or per-product aggregates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shelfsense import pipeline
from shelfsense.warehouse import get_warehouse


def tables() -> dict[str, pd.DataFrame]:
    wh = get_warehouse()
    res = pipeline.load_results()
    orders = wh.query(
        "SELECT SUBSTR(CAST(order_ts AS VARCHAR), 1, 7) AS month, item_id, COUNT(*) AS orders, "
        "ROUND(SUM(price), 2) AS revenue, COUNT(DISTINCT user_id) AS shoppers "
        "FROM orders GROUP BY 1, 2"
    )
    months = sorted(set(orders["month"]))
    start = pd.to_datetime(pd.Series(months) + "-01")
    dim_month = pd.DataFrame(
        {
            "month": months,
            "month_start": start.dt.date,
            "quarter": start.dt.year.astype(str) + "-Q" + start.dt.quarter.astype(str),
            "year": start.dt.year,
        }
    )
    recommended = wh.query(
        "SELECT item_id, COUNT(*) AS times_recommended, "
        "CAST(COUNT_IF(rank = 1) AS INT) AS times_first FROM recommendations GROUP BY item_id"
    )
    rec = pd.DataFrame(res["recommender"]["models"])
    a = res["aspects"]
    f1 = [
        {"model": name, "aspect": k, "f1": v}
        for name, key in (("labeller", "labeller_f1"), ("student", "student_f1"))
        for k, v in a.get(key, {}).items()
    ]
    return {
        "dim_product": wh.query("SELECT * FROM products"),
        "dim_month": dim_month,
        "fact_orders_monthly": orders,
        "fact_product_aspects": wh.query("SELECT * FROM product_aspects"),
        "fact_aspect_monthly": wh.query("SELECT * FROM aspect_monthly"),
        "fact_drift_monthly": wh.query("SELECT * FROM drift_monthly"),
        "fact_aspect_alerts": wh.query(
            "SELECT month, item_id, aspect, mentions, negative_share, baseline_negative_share, z "
            "FROM aspect_alerts"
        ),
        "fact_recommended_items": recommended,
        "fact_recommender_eval": rec,
        "fact_aspect_model": pd.DataFrame(f1, columns=["model", "aspect", "f1"]),
    }


def export(out: str | Path = "powerbi/data") -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in tables().items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    return paths
