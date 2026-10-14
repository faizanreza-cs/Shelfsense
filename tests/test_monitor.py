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


def test_monitoring_finds_the_reformulation_and_nothing_else(built):
    m = built["monitoring"]
    t = m["truth_check"]
    assert t["first_alert_month"] == t["true_change_month"] and t["other_alerts"] == 0
    assert m["drift_flags"] == 0  # the store itself is stable
