import pandas as pd

from shelfsense import synth
from shelfsense.warehouse import TABLES

SMALL = synth.SynthConfig(n_users=2000, products_per_category=10)


def test_generation_is_deterministic():
    a, b = synth.generate(SMALL), synth.generate(SMALL)
    for name in ("products", "orders", "reviews", "truth_review_labels"):
        pd.testing.assert_frame_equal(a.tables[name], b.tables[name])


def test_reviews_carry_their_true_labels():
    d = synth.generate(SMALL)
    reviews = d.tables["reviews"].merge(d.tables["truth_review_labels"], on="review_id")
    cardboard = reviews[reviews["text"].str.contains("Tastes like cardboard.", regex=False)]
    assert len(cardboard) and (cardboard["taste"] == "negative").all()
    better = reviews[reviews["text"].str.contains("never felt better", regex=False)]
    assert len(better) and (better["digestion"] == "positive").all()
    assert set(reviews["rating"]) <= {1, 2, 3, 4, 5}


def test_the_reformulated_product_is_the_top_protein(built):
    t = built["truth"]["reformulation"]
    assert t["reformulated_item"].startswith("P") and t["taste_after"] < t["taste_before"]


def test_ground_truth_stays_out_of_the_warehouse():
    assert not any(t.startswith("truth") for t in TABLES)
