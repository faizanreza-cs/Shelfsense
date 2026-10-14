import numpy as np
import pandas as pd
import pytest

from shelfsense import recsys


def _toy():
    products = pd.DataFrame(
        {
            "item_id": ["A", "B", "C", "D"],
            "title": ["Whey Vanilla", "Whey Chocolate", "Vitamin D", "Magnesium"],
            "brand": ["X", "X", "Y", "Z"],
            "category": ["Protein", "Protein", "Vitamins", "Minerals"],
            "price": [30.0, 32.0, 10.0, 15.0],
            "margin_pct": [0.2, 0.5, 0.3, 0.4],
            "in_stock": [True, True, True, False],
        }
    )
    ts = pd.Timestamp("2025-01-01")
    inter = pd.DataFrame(
        {"user_id": ["u1", "u1", "u2", "u2", "u3"], "item_id": ["A", "B", "A", "C", "A"],
         "ts": [ts] * 5}
    )  # fmt: skip
    return products, inter


def test_item_item_similarity_by_hand():
    products, inter = _toy()
    rec = recsys.Recommender(products, n_factors=2)
    f = rec.fit(inter, None, pd.Timestamp("2025-02-01"))
    # buyers: A {u1,u2,u3}, B {u1}, C {u2}: cosine(A, B) = 1 / sqrt(3 * 1)
    a, b = rec.items.get_loc("A"), rec.items.get_loc("B")
    assert f.sim[a, b] == pytest.approx(1 / np.sqrt(3))
    assert f.sim[a, a] == 0.0  # an item never recommends itself
    assert rec.similar(f, "B", 1)[0][0] == "A"


def test_recommendations_skip_owned_items_and_follow_rules():
    products, inter = _toy()
    rec = recsys.Recommender(products, n_factors=2)
    f = rec.fit(inter, None, pd.Timestamp("2025-02-01"))
    out = rec.recommend(f, ["u3"], "popularity", k=3)["u3"]
    assert "A" not in out and len(out) == 3
    ruled = rec.recommend(f, ["u3"], "popularity", k=3,
                          rules={"in_stock_only": True, "max_per_brand": 1})["u3"]  # fmt: skip
    assert "D" not in ruled  # out of stock
    brands = products.set_index("item_id").loc[ruled, "brand"]
    assert brands.value_counts().max() == 1


def test_unknown_shoppers_are_an_error_not_someone_elses_list():
    products, inter = _toy()
    rec = recsys.Recommender(products, n_factors=2)
    f = rec.fit(inter, None, pd.Timestamp("2025-02-01"))
    with pytest.raises(KeyError, match="no purchase history"):
        rec.recommend(f, ["u1", "nobody"], "popularity")


def test_evaluation_ends_at_the_last_complete_month():
    from shelfsense.pipeline import data_end

    full = pd.Series(pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h"))
    assert data_end(full) == full.max()
    # a dump whose last two months are nearly empty: evaluate up to the end of October
    tail = pd.Series(pd.to_datetime(["2024-11-03", "2024-12-01", "2024-12-02"]))
    sparse = pd.concat([full[full < "2024-11-01"], tail], ignore_index=True)
    assert data_end(sparse) == full[full < "2024-11-01"].max()


def test_evaluate_by_hand():
    recs = {"u": ["a", "b", "c"]}
    out = recsys.evaluate(recs, {"u": {"b", "z"}}, n_items=10)
    assert out["recall_at_10"] == 0.5
    assert out["ndcg_at_10"] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)), abs=1e-4)
    assert out["coverage"] == 0.3


def test_new_items_ignore_what_shoppers_already_own():
    products, inter = _toy()
    rec = recsys.Recommender(products, n_factors=2)
    f = rec.fit(inter, None, pd.Timestamp("2025-01-02"))
    later = pd.concat([inter, pd.DataFrame({"user_id": ["u1", "u1"], "item_id": ["A", "C"],
                                            "ts": [pd.Timestamp("2025-01-10")] * 2})])  # fmt: skip
    assert recsys.new_items(later, f, pd.Timestamp("2025-01-02"), 30) == {"u1": {"C"}}


def test_hybrid_beats_the_baselines_and_rules_raise_margin(built):
    m = {r["model"]: r for r in built["recommender"]["models"]}
    assert m["hybrid"]["recall_at_10"] >= m["item_item"]["recall_at_10"] - 0.005
    assert m["hybrid"]["recall_at_10"] > m["popularity"]["recall_at_10"]
    assert m["hybrid + rules"]["avg_margin_pct"] > m["hybrid"]["avg_margin_pct"]
    assert set(built["recommender"]["reranker_weights"]) == set(recsys.FEATURES)
