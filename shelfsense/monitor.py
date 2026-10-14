"""Monitoring: store-level drift (PSI) and product-level aspect alerts.

* **Drift (PSI)**: each month's distribution of order categories, order prices, star ratings
  and predicted aspect labels, compared with a reference period (the first months). Rule of
  thumb: PSI < 0.1 stable, 0.1-0.25 watch, >= 0.25 drift.
* **Aspect alerts**: for every product and aspect, this month's share of negative mentions
  against all earlier months (two-proportion z-test). An alert needs enough mentions, a large
  rise and a high z, because about 20,000 product-aspect-months are tested.

Store-level PSI can't see one product going wrong among 240; the product alerts can.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shelfsense.synth import ASPECTS, LABELS

PSI_WATCH, PSI_DRIFT = 0.1, 0.25


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    """Population stability index between two count vectors over the same bins."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.shape != a.shape or e.sum() <= 0 or a.sum() <= 0:
        raise ValueError("need two non-empty count vectors of the same length")
    e = np.clip(e / e.sum(), eps, None)
    a = np.clip(a / a.sum(), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    return "drift" if value >= PSI_DRIFT else "watch" if value >= PSI_WATCH else "stable"


def _month(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.strftime("%Y-%m")


def drift_table(
    orders: pd.DataFrame, products: pd.DataFrame, reviews: pd.DataFrame,
    labels: pd.DataFrame, reference_months: int = 6,
) -> pd.DataFrame:  # fmt: skip
    """PSI per month and metric against the first ``reference_months`` months."""
    o = orders.merge(products[["item_id", "category"]], on="item_id")
    o["month"] = _month(o["order_ts"])
    r = reviews[["review_id", "rating", "review_ts"]].merge(labels, on="review_id")
    r["month"] = _month(r["review_ts"])
    months = sorted(o["month"].unique())
    ref = set(months[:reference_months])

    cats = sorted(products["category"].unique())
    price_edges = np.quantile(o.loc[o["month"].isin(ref), "price"], np.linspace(0, 1, 11)[1:-1])

    def counts(metric: str, frame: pd.DataFrame) -> np.ndarray:
        if metric == "order_category_mix":
            return frame["category"].value_counts().reindex(cats, fill_value=0).to_numpy()
        if metric == "order_price":
            return np.bincount(np.searchsorted(price_edges, frame["price"]), minlength=10)
        if metric == "review_rating":
            return frame["rating"].value_counts().reindex(range(1, 6), fill_value=0).to_numpy()
        aspect = metric.removeprefix("predicted_")
        return frame[aspect].value_counts().reindex(list(LABELS), fill_value=0).to_numpy()

    metrics = ["order_category_mix", "order_price", "review_rating",
               *(f"predicted_{a}" for a in ASPECTS)]  # fmt: skip
    rows = []
    for metric in metrics:
        frame = o if metric.startswith("order") else r
        base = counts(metric, frame[frame["month"].isin(ref)])
        for m in months[reference_months:]:
            cur = frame[frame["month"] == m]
            if cur.empty:
                continue
            v = psi(base, counts(metric, cur))
            rows.append({"month": m, "metric": metric, "psi": round(v, 4), "status": status(v),
                         "rows": len(cur)})  # fmt: skip
    return pd.DataFrame(rows)


def aspect_alerts(
    monthly: pd.DataFrame, min_mentions: int = 10, min_baseline: int = 30,
    z_crit: float = 4.0, min_rise: float = 0.25,
) -> pd.DataFrame:  # fmt: skip
    """Product-aspect-months whose negative share rose sharply against all earlier months.

    ``monthly`` is ``nlp.aspect_monthly`` output (item_id, aspect, month, positive, negative,
    mentions).
    """
    rows = []
    for (item, aspect), g in monthly.sort_values("month").groupby(["item_id", "aspect"]):
        neg = g["negative"].to_numpy(dtype=float)
        n = g["mentions"].to_numpy(dtype=float)
        base_neg, base_n = np.cumsum(neg) - neg, np.cumsum(n) - n  # everything before the month
        ok = (n >= min_mentions) & (base_n >= min_baseline)
        if not ok.any():
            continue
        p1 = np.divide(neg, n, out=np.zeros_like(n), where=n > 0)
        p0 = np.divide(base_neg, base_n, out=np.zeros_like(n), where=base_n > 0)
        pooled = np.divide(neg + base_neg, n + base_n, out=np.zeros_like(n), where=n + base_n > 0)
        se = np.sqrt(pooled * (1 - pooled) * (1 / np.maximum(n, 1) + 1 / np.maximum(base_n, 1)))
        z = np.divide(p1 - p0, se, out=np.zeros_like(n), where=se > 0)
        hit = ok & (z >= z_crit) & (p1 - p0 >= min_rise)
        for i in np.flatnonzero(hit):
            rows.append({
                "item_id": item, "aspect": aspect, "month": g["month"].iloc[i],
                "mentions": int(n[i]), "negative_share": round(float(p1[i]), 3),
                "baseline_negative_share": round(float(p0[i]), 3),
                "baseline_mentions": int(base_n[i]), "z": round(float(z[i]), 2),
            })  # fmt: skip
    cols = ["item_id", "aspect", "month", "mentions", "negative_share",
            "baseline_negative_share", "baseline_mentions", "z"]  # fmt: skip
    return pd.DataFrame(rows, columns=cols).sort_values(["month", "z"], ascending=[True, False])
