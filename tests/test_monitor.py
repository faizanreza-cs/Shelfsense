import numpy as np
import pandas as pd
import pytest

from shelfsense import monitor


def test_psi():
    assert monitor.psi(np.array([10, 20, 30]), np.array([10, 20, 30])) == pytest.approx(0)
    # (0.5-0.25)*ln(0.5/0.25) + (0.5-0.75)*ln(0.5/0.75)
    expected = 0.25 * np.log(2) - 0.25 * np.log(2 / 3)
    assert monitor.psi(np.array([25, 75]), np.array([50, 50])) == pytest.approx(expected)
    with pytest.raises(ValueError):
        monitor.psi(np.array([1, 2]), np.array([1, 2, 3]))
    assert monitor.status(0.3) == "drift" and monitor.status(0.05) == "stable"


def _monthly(neg_last: int) -> pd.DataFrame:
    months = [f"2025-{m:02d}" for m in range(1, 13)]
    neg = [3] * 11 + [neg_last]
    return pd.DataFrame({"item_id": "P1", "aspect": "taste", "month": months,
                         "negative": neg, "positive": [27] * 11 + [30 - neg_last],
                         "mentions": 30})  # fmt: skip


def test_alert_fires_on_a_jump_and_not_on_noise():
    alerts = monitor.aspect_alerts(_monthly(neg_last=22))
    assert list(alerts["month"]) == ["2025-12"] and alerts["z"].iloc[0] > 4
    assert monitor.aspect_alerts(_monthly(neg_last=5)).empty


def test_drift_skips_months_too_thin_to_measure():
    months = pd.date_range("2025-01-01", periods=9, freq="MS")
    rows = [(m + pd.Timedelta(days=d % 27), "A" if d % 3 else "B") for m in months[:8]
            for d in range(300)]  # fmt: skip
    rows += [(months[8], "B")] * 5  # a final month with 5 orders, all of category B
    o = pd.DataFrame(rows, columns=["order_ts", "item_id"])
    o["price"] = 10.0
    products = pd.DataFrame({"item_id": ["A", "B"], "category": ["A", "B"]})
    reviews = pd.DataFrame({"review_id": ["r1"], "rating": [5], "review_ts": [months[0]]})
    labels = pd.DataFrame({"review_id": ["r1"], **dict.fromkeys(monitor.ASPECTS, "none")})
    drift = monitor.drift_table(o, products, reviews, labels, reference_months=6)
    assert set(drift["month"]) == {"2025-07", "2025-08"}  # September's 5 rows are skipped
    assert (drift["status"] == "stable").all()


def test_monitoring_finds_the_reformulation_and_nothing_else(built):
    m = built["monitoring"]
    t = m["truth_check"]
    assert t["first_alert_month"] == t["true_change_month"] and t["other_alerts"] == 0
    assert m["drift_flags"] == 0  # the store itself is stable
