"""The real-data path, tested offline: the synthetic store written in the dataset's format."""

import json

import pytest

from shelfsense import amazon, pipeline, synth
from shelfsense.config import Settings


def test_price_and_category_parsing():
    assert amazon._price(12.5) == 12.5 and amazon._price("$1,299.00") == 1299.0
    assert amazon._price(None) is None and amazon._price("—") is None and amazon._price(0) is None
    assert amazon._is_supplement(["Health & Household", "Vitamins, Minerals & Supplements", "Zinc"])
    assert not amazon._is_supplement(["Health & Household", "Household Supplies"])


def test_sources_fall_back_to_the_next_one(tmp_path, monkeypatch):
    folder = tmp_path / "meta_categories"
    folder.mkdir()
    (folder / "meta_X.jsonl").write_text(json.dumps({"parent_asin": "B1"}) + "\n")  # no .gz
    monkeypatch.setenv("SHELFSENSE_AMAZON_BASE", str(tmp_path))
    assert [json.loads(x)["parent_asin"] for x in amazon._lines("meta", "X")] == ["B1"]
    with pytest.raises(RuntimeError, match="no source"):
        list(amazon._lines("reviews", "X"))


def test_fetch_runs_the_whole_pipeline_on_dataset_files(tmp_path, monkeypatch):
    data = synth.generate(synth.SynthConfig(n_users=12_000))
    amazon.to_amazon_format(data, tmp_path / "raw", "Health_and_Household")
    monkeypatch.setenv("SHELFSENSE_AMAZON_BASE", str(tmp_path / "raw"))
    s = Settings(db_path=tmp_path / "db" / "shelfsense.duckdb")
    res = amazon.fetch_and_build(since="2024-01-01", min_reviews=2, settings=s)
    assert res["data"]["source"].startswith("amazon-reviews-2023")
    assert 200 <= res["data"]["products"] <= 240
    assert "truth_check" not in res["monitoring"] and "student_f1" not in res["aspects"]
    rc = res["aspects"]["rating_check"]
    assert rc["mean_stars_with_a_negative_aspect"] < rc["mean_stars_positive_only"]
    import duckdb

    con = duckdb.connect(str(s.db_path), read_only=True)
    items = {i for (i,) in con.execute("SELECT item_id FROM products").fetchall()}
    assert "B000OTHER" not in items  # the paper towels aren't a supplement
    users = [u for (u,) in con.execute("SELECT DISTINCT user_id FROM reviews LIMIT 5").fetchall()]
    assert all(u.startswith("C") and not u.startswith("CAG") for u in users)  # ids replaced
    con.close()
    assert pipeline.load_results(s)["data"] == res["data"]
