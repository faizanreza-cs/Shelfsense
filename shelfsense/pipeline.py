"""End to end: data -> recommender -> review aspects -> monitoring -> warehouse + ``results.json``.

``analyse`` takes plain tables (products, orders, reviews), so it runs the same way on the
synthetic store and on Amazon Reviews 2023 (``shelfsense fetch``). When the simulator's truth
tables are there, every step is also scored against them.
"""

from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shelfsense import monitor, nlp, recsys, synth
from shelfsense.config import Settings, get_settings, model_path, results_path
from shelfsense.synth import ASPECTS
from shelfsense.warehouse import reset_cache

HORIZON = 90  # days of purchases the recommender is evaluated on
DEFAULT_RULES = {"in_stock_only": True, "max_per_brand": 3, "margin_boost": 0.3}
LEARNING_CURVE = (250, 500, 1000, 2000)


def _rules_for(products: pd.DataFrame) -> dict[str, Any]:
    """Business rules the catalogue supports (real catalogues may lack stock or margin)."""
    rules: dict[str, Any] = {"max_per_brand": DEFAULT_RULES["max_per_brand"]}
    if "in_stock" in products:
        rules["in_stock_only"] = True
    if "margin_pct" in products:
        rules["margin_boost"] = DEFAULT_RULES["margin_boost"]
    return rules


def run_recommender(
    products: pd.DataFrame, orders: pd.DataFrame, reviews: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Offline evaluation on the last ``HORIZON`` days, then refit on everything for serving."""
    inter = recsys.interactions(orders)
    cutoff = (inter["ts"].max() - pd.Timedelta(days=HORIZON)).normalize()
    rec = recsys.Recommender(products)
    rec.train_reranker(inter, reviews, cutoff, horizon=HORIZON)
    f = rec.fit(inter, reviews, cutoff)
    targets = recsys.new_items(inter, f, cutoff, HORIZON)
    users = sorted(targets)
    margin = products.set_index("item_id")["margin_pct"] if "margin_pct" in products else None
    rows = []
    for m in [*recsys.MODELS, "hybrid"]:
        r = rec.recommend(f, users, m)
        rows.append({"model": m, **recsys.evaluate(r, targets, len(products), margin)})
    rules = _rules_for(products)
    r = rec.recommend(f, users, "hybrid", rules=rules)
    rows.append({"model": "hybrid + rules", **recsys.evaluate(r, targets, len(products), margin)})
    result = {
        "cutoff": str(cutoff.date()),
        "horizon_days": HORIZON,
        "test_shoppers": len(users),
        "models": rows,
        "rules": rules,
        "reranker_weights": rec.reranker_weights(),
    }
    # serving: candidate models refitted on all purchases; the re-ranker keeps its weights
    f_all = rec.fit(inter, reviews, inter["ts"].max() + pd.Timedelta(days=1))
    all_users = [str(u) for u in f_all.users]
    recs = rec.recommend(f_all, all_users, "hybrid", rules=rules)
    table = pd.DataFrame(
        [(u, i + 1, item) for u, items in recs.items() for i, item in enumerate(items)],
        columns=["user_id", "rank", "item_id"],
    )
    return result, table, {"recommender": rec, "fitted": f_all, "rules": rules}


def _labels_for_sample(
    reviews: pd.DataFrame, labeller: str, n: int, seed: int, provider: object | None,
    labels: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str]:  # fmt: skip
    """The teacher's labels for a sample of reviews: given, from an LLM, or from keywords."""
    if labels is not None:
        return labels[labels["review_id"].isin(reviews["review_id"])], "file"
    sample = reviews.sample(min(n, len(reviews)), random_state=seed)
    if labeller == "llm":
        if provider is None:
            from shelfsense.agent.provider import get_provider

            provider = get_provider()
        return nlp.label_with_llm(sample, provider), "llm"
    if labeller != "keyword":
        raise ValueError(f"unknown labeller {labeller!r} (keyword | llm)")
    return nlp.label_with_keywords(sample), "keyword"


def run_aspects(
    reviews: pd.DataFrame, truth_labels: pd.DataFrame | None = None,
    quality: pd.DataFrame | None = None, *, labeller: str = "keyword", label_sample: int = 2000,
    provider: object | None = None, labels: pd.DataFrame | None = None, seed: int = 0,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:  # fmt: skip
    """Label a sample, train the student, score every review; plus product and monthly scores."""
    teacher, source = _labels_for_sample(reviews, labeller, label_sample, seed, provider, labels)
    if teacher.empty:
        raise ValueError("no labelled reviews to train on")
    train = reviews.merge(teacher[["review_id"]], on="review_id")
    model = nlp.AspectModel().fit(train["text"], train[["review_id"]].merge(teacher))
    pred = model.predict(reviews["text"])
    pred.insert(0, "review_id", reviews["review_id"].to_numpy())
    long = nlp.long_labels(pred, reviews)
    scores = nlp.product_aspects(long)
    monthly = nlp.aspect_monthly(long)
    result: dict[str, Any] = {
        "labeller": source,
        "labelled_reviews": len(teacher),
        "scored_reviews": len(reviews),
        "mention_rate": {a: round(float((pred[a] != "none").mean()), 3) for a in ASPECTS},
    }
    # a check that needs no labels (real data): negative aspects should come with lower stars
    per = pred.set_index("review_id")[list(ASPECTS)]
    neg = per.eq("negative").any(axis=1)
    pos_only = per.eq("positive").any(axis=1) & ~neg
    stars = reviews.set_index("review_id")["rating"]
    result["rating_check"] = {
        "mean_stars_with_a_negative_aspect": round(float(stars[neg[neg].index].mean()), 2),
        "mean_stars_positive_only": round(float(stars[pos_only[pos_only].index].mean()), 2),
        "share_with_a_negative_aspect": round(float(neg.mean()), 3),
    }
    if truth_labels is not None:
        # score on reviews the student never saw
        unseen = ~reviews["review_id"].isin(teacher["review_id"])
        test_ids = reviews.loc[unseen, ["review_id"]]
        result["labeller_f1"] = nlp.f1_by_aspect(teacher, truth_labels)
        result["student_f1"] = nlp.f1_by_aspect(test_ids.merge(pred), truth_labels)
        # what a perfect labeller would give: the same student trained on the true labels
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(reviews))
        test = reviews.iloc[order[:10_000]]
        sources = {"perfect_labels_curve": truth_labels}
        if source == "keyword":  # free to label any sample, so it gets a curve too
            pool = reviews.iloc[order[10_000 : 10_000 + max(LEARNING_CURVE)]]
            sources["keyword_labels_curve"] = nlp.label_with_keywords(pool)
        for key, lab in sources.items():
            curve = []
            for n in LEARNING_CURVE:
                part = reviews.iloc[order[10_000 : 10_000 + n]]
                m = nlp.AspectModel().fit(part["text"], part[["review_id"]].merge(lab))
                p = m.predict(test["text"])
                p.insert(0, "review_id", test["review_id"].to_numpy())
                curve.append({"labels": n, "f1": nlp.f1_by_aspect(p, truth_labels)["mean"]})
            result[key] = curve
    if quality is not None:
        result["quality_correlation"] = nlp.quality_correlation(scores, quality)
    long_out = long[["review_id", "item_id", "month", "aspect", "label"]]
    return result, {
        "review_aspects": long_out,
        "product_aspects": scores,
        "aspect_monthly": monthly,
    }


def run_monitoring(
    products: pd.DataFrame, orders: pd.DataFrame, reviews: pd.DataFrame,
    tables: dict[str, pd.DataFrame], truth: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:  # fmt: skip
    """Store-level drift and product-level aspect alerts from the scored reviews."""
    wide = tables["review_aspects"].pivot_table(
        index="review_id", columns="aspect", values="label", aggfunc="first"
    )
    wide = wide.reindex(columns=list(ASPECTS)).fillna("none").reset_index()
    wide = reviews[["review_id"]].merge(wide, on="review_id", how="left").fillna("none")
    drift = monitor.drift_table(orders, products, reviews, wide)
    alerts = monitor.aspect_alerts(tables["aspect_monthly"])
    alerts = alerts.merge(products[["item_id", "title", "brand"]], on="item_id", how="left")
    result: dict[str, Any] = {
        "reference_months": 6,
        "months_checked": int(drift["month"].nunique()) if len(drift) else 0,
        "max_psi": drift.groupby("metric")["psi"].max().round(4).to_dict() if len(drift) else {},
        "drift_flags": int((drift["status"] != "stable").sum()) if len(drift) else 0,
        "alerts": json.loads(alerts.to_json(orient="records")),
    }
    if truth and truth.get("reformulation"):
        t = truth["reformulation"]
        hit = alerts[(alerts["item_id"] == t["reformulated_item"]) & (alerts["aspect"] == "taste")]
        result["truth_check"] = {
            "reformulated_item": t["reformulated_item"],
            "true_change_month": t["reformulation_month"],
            "first_alert_month": str(hit["month"].min()) if len(hit) else None,
            "other_alerts": int(len(alerts) - len(hit)),
        }
    return result, {"drift_monthly": drift, "aspect_alerts": alerts}


def analyse(
    tables: dict[str, pd.DataFrame], truth: dict[str, Any] | None = None, *,
    labeller: str = "keyword", label_sample: int = 2000, provider: object | None = None,
    labels: pd.DataFrame | None = None, source: str = "synthetic",
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Any]]:  # fmt: skip
    """Run everything on products/orders/reviews. Returns (results, derived tables, model)."""
    products, orders, reviews = tables["products"], tables["orders"], tables["reviews"]
    rec_result, rec_table, model = run_recommender(products, orders, reviews)
    nlp_result, nlp_tables = run_aspects(
        reviews, tables.get("truth_review_labels"), tables.get("truth_aspect_quality"),
        labeller=labeller, label_sample=label_sample, provider=provider, labels=labels,
    )  # fmt: skip
    mon_result, mon_tables = run_monitoring(products, orders, reviews, nlp_tables, truth)
    ts = pd.to_datetime(orders["order_ts"])
    results = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": {
            "source": source,
            "products": len(products),
            "shoppers": int(orders["user_id"].nunique()),
            "orders": len(orders),
            "reviews": len(reviews),
            "first_order": str(ts.min().date()),
            "last_order": str(ts.max().date()),
        },
        "recommender": rec_result,
        "aspects": nlp_result,
        "monitoring": mon_result,
        "truth": truth,
    }
    derived = {"recommendations": rec_table, **nlp_tables, **mon_tables}
    return results, derived, model


def save(results: dict[str, Any], model: dict[str, Any], db_path: str | Path) -> None:
    """Write results.json and the fitted recommender next to the database (atomically)."""
    for path, blob in (
        (results_path(db_path), json.dumps(results, indent=1, default=str).encode()),
        (model_path(db_path), pickle.dumps(model)),
    ):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(blob)
        tmp.replace(path)
    _load.cache_clear()
    _load_model.cache_clear()
    reset_cache()


def build(
    settings: Settings | None = None, cfg: synth.SynthConfig | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Generate the synthetic store, run everything, write the database and results."""
    s = settings or get_settings()
    data = synth.generate(cfg)
    results, derived, model = analyse(data.tables, data.truth, **kwargs)
    data.tables.update(derived)
    synth.write(data, s.db_path)
    save(results, model, s.db_path)
    return results


@lru_cache(maxsize=2)
def _load(path: str, mtime: float) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text()))


def load_results(settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    path = results_path(s.db_path)
    if not path.exists():
        raise FileNotFoundError(f"No results at {path}. Run `shelfsense build` first.")
    return _load(str(path), path.stat().st_mtime)


@lru_cache(maxsize=2)
def _load_model(path: str, mtime: float) -> dict[str, Any]:
    # our own file, written by `shelfsense build` next to the database; never load one you
    # didn't build yourself (pickle runs code on load)
    with open(path, "rb") as fh:
        return dict(pickle.load(fh))


def load_model(settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    path = model_path(s.db_path)
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run `shelfsense build` first.")
    return _load_model(str(path), path.stat().st_mtime)
