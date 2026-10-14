"""Synthetic store data for an online health and nutrition shop, with ground truth.

* **Catalogue:** 240 supplements in 8 categories from 12 brands (all fictional). Every product
  has a hidden quality on five review aspects: taste, digestion, effectiveness, value and
  packaging.
* **Shoppers:** 30,000 shoppers buy over two years. What they buy depends on category tastes,
  complements (protein buyers often add electrolytes), brand loyalty, price sensitivity,
  product quality and re-buying.
* **Reviews:** about a third of first purchases get a review. Each sentence talks about an
  aspect, and its sentiment comes from the product's hidden quality, so every review has true
  aspect labels. About a quarter of the sentences are harder: idioms, negation and complaints
  without the obvious keywords ("tastes like cardboard", "my stomach has never felt better").
* **A reformulation:** one popular protein powder gets worse taste from month 20. Monitoring
  should notice.

Reviews are built from sentence templates, so they are cleaner than real reviews; the real-data
workflow runs the same pipeline on Amazon Reviews 2023.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ASPECTS: tuple[str, ...] = ("taste", "digestion", "effectiveness", "value", "packaging")
LABELS: tuple[str, ...] = ("none", "positive", "negative")

# category: (price range, chance a review mentions each aspect, product nouns, flavours)
CATEGORIES: dict[str, dict[str, Any]] = {
    "Protein Powder": {"price": (29, 59), "mention": (0.70, 0.35, 0.30, 0.35, 0.20),
                       "noun": "protein", "flavors": ["Vanilla", "Chocolate", "Cookies & Cream", "Unflavored"]},
    "Collagen": {"price": (22, 45), "mention": (0.45, 0.15, 0.45, 0.30, 0.20),
                 "noun": "collagen", "flavors": ["Unflavored", "Berry", "Vanilla"]},
    "Multivitamins": {"price": (12, 35), "mention": (0.10, 0.30, 0.45, 0.35, 0.20),
                      "noun": "multivitamin", "flavors": ["Tablets", "Gummies"]},
    "Vitamin D": {"price": (9, 22), "mention": (0.05, 0.10, 0.40, 0.40, 0.15),
                  "noun": "vitamin D", "flavors": ["Softgels", "Gummies"]},
    "Magnesium": {"price": (14, 30), "mention": (0.05, 0.40, 0.50, 0.30, 0.15),
                  "noun": "magnesium", "flavors": ["Glycinate", "Citrate"]},
    "Probiotics": {"price": (18, 45), "mention": (0.05, 0.60, 0.40, 0.30, 0.20),
                   "noun": "probiotic", "flavors": ["Capsules", "Shelf-stable"]},
    "Electrolytes": {"price": (15, 35), "mention": (0.65, 0.15, 0.35, 0.30, 0.20),
                     "noun": "electrolyte mix", "flavors": ["Lemon Lime", "Watermelon", "Raspberry"]},
    "Sleep Support": {"price": (12, 30), "mention": (0.20, 0.10, 0.60, 0.25, 0.15),
                      "noun": "sleep aid", "flavors": ["Capsules", "Gummies"]},
}  # fmt: skip
COMPLEMENTS = {
    "Protein Powder": ["Electrolytes", "Collagen"],
    "Collagen": ["Vitamin D", "Multivitamins"],
    "Multivitamins": ["Vitamin D", "Probiotics"],
    "Vitamin D": ["Magnesium", "Multivitamins"],
    "Magnesium": ["Sleep Support"],
    "Probiotics": ["Multivitamins"],
    "Electrolytes": ["Protein Powder"],
    "Sleep Support": ["Magnesium"],
}
BRANDS = [
    "Fernway Nutrition", "Northpeak", "Vitalis Lab", "Greenroot", "Oakline", "PureForm",
    "Summit Daily", "Bloomwell", "Kinetic Labs", "Harbor Health", "Solstice", "Truevine",
]  # fmt: skip

# Sentence templates: (aspect, polarity) -> full sentences, plus short clauses for mixed ones
TEMPLATES: dict[tuple[str, str], list[str]] = {
    ("taste", "positive"): [
        "Tastes great, not chalky at all.",
        "The {flavor} flavor is delicious.",
        "Mixes easily and tastes good.",
        "Honestly the best tasting {noun} I have tried.",
        "No weird aftertaste.",
        "Flavor is pleasant and not too sweet.",
        "Even my kids like the taste.",
        "Smooth texture and good flavor.",
    ],
    ("taste", "negative"): [
        "Tastes awful and very chalky.",
        "The {flavor} flavor is way too artificial.",
        "Strong aftertaste that I can't get past.",
        "Clumps in the shaker and tastes bitter.",
        "Too sweet, almost undrinkable.",
        "The taste is terrible, I had to throw it out.",
        "Gritty texture and bland flavor.",
        "It doesn't taste like it used to.",
    ],
    ("digestion", "positive"): [
        "Easy on my stomach.",
        "No bloating at all.",
        "Didn't upset my stomach like other brands.",
        "My digestion has been better since I started.",
        "Gentle on the stomach, even before breakfast.",
    ],
    ("digestion", "negative"): [
        "Gave me bloating and gas.",
        "Upset my stomach every time.",
        "Caused cramps and nausea.",
        "Made me feel sick to my stomach.",
        "Constant stomach issues after taking it.",
    ],
    ("effectiveness", "positive"): [
        "I noticed a real difference after two weeks.",
        "It really works.",
        "Noticeably more energy during the day.",
        "Saw results within a month.",
        "My recovery after workouts improved.",
        "Definitely works for me.",
    ],
    ("effectiveness", "negative"): [
        "Didn't notice any difference.",
        "Does nothing for me.",
        "No results after a month.",
        "Waste of time, it doesn't work.",
        "I felt no effect at all.",
    ],
    ("value", "positive"): [
        "Great value for the price.",
        "Cheaper than the store brand and lasts longer.",
        "Worth every penny.",
        "Good price for the quality.",
    ],
    ("value", "negative"): [
        "Way too expensive for what you get.",
        "Overpriced, the container is half empty.",
        "Not worth the money.",
        "The price went up and it's no longer worth it.",
    ],
    ("packaging", "positive"): [
        "Arrived well sealed and on time.",
        "The container is sturdy and the scoop is included.",
        "Nice packaging, easy to open.",
        "Bottle arrived intact with the seal on.",
    ],
    ("packaging", "negative"): [
        "The seal was broken when it arrived.",
        "Container arrived cracked and leaking.",
        "No scoop inside the tub.",
        "The lid is hard to close and the bag tore.",
    ],
}
# Harder phrasings: the meaning is clear to a reader but keyword rules tend to miss or flip it
HARD: dict[tuple[str, str], list[str]] = {
    ("taste", "positive"): [
        "I was worried about the flavor, but no complaints here.",
        "The {flavor} one is my go-to now.",
        "Never had a {noun} this smooth.",
        "Five stars for flavor alone.",
    ],
    ("taste", "negative"): [
        "Tastes like cardboard.",
        "I wanted to love the flavor, but I couldn't.",
        "My shaker smells of it for days.",
        "The {flavor} version tastes nothing like it should.",
    ],
    ("digestion", "positive"): [
        "Gone are the days of bloating after my shakes.",
        "My stomach has never felt better.",
        "Zero tummy trouble.",
    ],
    ("digestion", "negative"): [
        "Had to stay close to a bathroom all day.",
        "My stomach did not agree with it.",
        "Spent the afternoon regretting it, if you know what I mean.",
    ],
    ("effectiveness", "positive"): [
        "Does exactly what it says.",
        "I can't imagine going without it now.",
        "Sleeping through the night for the first time in years.",
    ],
    ("effectiveness", "negative"): [
        "Might as well be taking a sugar pill.",
        "I wanted it to work so badly.",
        "Three bottles in and still waiting for results.",
    ],
    ("value", "positive"): [
        "You won't find a better deal.",
        "Pricey up front, but it lasts forever.",
        "Half the cost of what I paid at the pharmacy.",
    ],
    ("value", "negative"): [
        "Paying a premium for a fancy label.",
        "Smaller tub, same price as before.",
        "You can get the same thing for half the price elsewhere.",
    ],
    ("packaging", "positive"): [
        "Not a single dent or leak.",
        "Came double bagged with no mess.",
        "The resealable pouch is a nice touch.",
    ],
    ("packaging", "negative"): [
        "Powder everywhere when I opened the box.",
        "The pouch zipper gave out after a week.",
        "Lid popped off in transit.",
    ],
}
HARD_SHARE = 0.25

CLAUSES: dict[tuple[str, str], str] = {
    ("taste", "positive"): "it tastes great", ("taste", "negative"): "it tastes chalky",
    ("digestion", "positive"): "it's easy on my stomach",
    ("digestion", "negative"): "it gives me bloating",
    ("effectiveness", "positive"): "it actually works",
    ("effectiveness", "negative"): "I saw no results",
    ("value", "positive"): "the price is fair", ("value", "negative"): "it's overpriced",
    ("packaging", "positive"): "the packaging is solid",
    ("packaging", "negative"): "the seal was broken",
}  # fmt: skip
FILLERS = [
    "I take it every morning with breakfast.", "Bought this on a friend's recommendation.",
    "Ordered the large size this time.", "Been using it for about three weeks now.",
    "My doctor suggested I try it.", "Second time ordering.",
]  # fmt: skip


@dataclass(frozen=True)
class SynthConfig:
    seed: int = 11
    start: str = "2024-10-01"
    months: int = 24
    products_per_category: int = 30
    n_users: int = 30_000
    mean_orders: float = 6.0
    review_rate: float = 0.35
    reformulation_month: int = 20  # 1-based month when one product's taste gets worse


@dataclass
class SynthData:
    tables: dict[str, pd.DataFrame]
    truth: dict[str, Any]


def _catalogue(cfg: SynthConfig, rng: np.random.Generator) -> pd.DataFrame:
    brand_quality = dict(zip(BRANDS, rng.normal(0, 0.35, len(BRANDS)), strict=True))
    brand_price = dict(zip(BRANDS, rng.uniform(-0.15, 0.25, len(BRANDS)), strict=True))
    rows: list[dict[str, Any]] = []
    for c, spec in CATEGORIES.items():
        lo, hi = spec["price"]
        for i in range(cfg.products_per_category):
            brand = BRANDS[(i + list(CATEGORIES).index(c) * 5) % len(BRANDS)]
            flavor = spec["flavors"][i % len(spec["flavors"])]
            q = np.clip(brand_quality[brand] + rng.normal(0, 0.5, len(ASPECTS)), -1, 1)
            price = float(np.clip(rng.uniform(lo, hi) * (1 + brand_price[brand]), 6, 80))
            rows.append(
                {
                    "item_id": f"P{len(rows) + 1:04d}",
                    "title": f"{brand} {flavor} {spec['noun'].title()} {i // 12 + 1}",
                    "brand": brand,
                    "category": c,
                    "flavor": flavor,
                    "price": round(price, 2),
                    "margin_pct": round(float(rng.uniform(0.18, 0.5)), 3),
                    "in_stock": bool(rng.random() > 0.05),
                    "popularity": float(rng.lognormal(0, 0.8)),
                    **{f"q_{a}": float(q[k]) for k, a in enumerate(ASPECTS)},
                }
            )
    return pd.DataFrame(rows)


def _quality(products: pd.DataFrame) -> np.ndarray:
    """Overall quality: aspects weighted by how much reviews of the category talk about them."""
    w = np.array([CATEGORIES[c]["mention"] for c in products["category"]])
    q = products[[f"q_{a}" for a in ASPECTS]].to_numpy()
    return np.asarray((w * q).sum(axis=1) / w.sum(axis=1))


def _orders(cfg: SynthConfig, products: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    cats = list(CATEGORIES)
    cat_of = products["category"].map({c: i for i, c in enumerate(cats)}).to_numpy()
    blocks = [np.flatnonzero(cat_of == i) for i in range(len(cats))]
    price = products["price"].to_numpy()
    price_z = (price - price.mean()) / price.std()
    base = np.log(products["popularity"].to_numpy()) + 1.3 * _quality(products)
    brand_idx = products["brand"].map({b: i for i, b in enumerate(BRANDS)}).to_numpy()
    days = cfg.months * 30
    n = cfg.n_users

    n_orders = rng.poisson(rng.lognormal(np.log(cfg.mean_orders) - 0.18, 0.6, n))
    prefs = rng.dirichlet(np.full(len(cats), 0.6), n)
    sens = rng.uniform(0, 1.2, n)
    favourite = np.where(rng.random(n) < 0.4, rng.integers(0, len(BRANDS), n), -1)
    # each shopper's choice probabilities within every category, computed once (users x items)
    util = base[None, :] - sens[:, None] * price_z[None, :]
    util += 1.5 * (brand_idx[None, :] == favourite[:, None])
    cum = np.zeros_like(util)
    for b in blocks:
        e = np.exp(util[:, b] - util[:, b].max(axis=1, keepdims=True))
        cum[:, b] = np.cumsum(e / e.sum(axis=1, keepdims=True), axis=1)
    pref_cum = np.cumsum(prefs, axis=1)
    comp = {cats.index(c): [cats.index(x) for x in v] for c, v in COMPLEMENTS.items()}

    users, items, times = [], [], []
    for u in np.flatnonzero(n_orders):
        k = int(n_orders[u])
        r = rng.random((k, 4))
        owned: list[int] = []
        prev = -1
        for j in range(k):
            if owned and r[j, 0] < 0.35:  # buy again
                item = owned[int(r[j, 1] * len(owned))]
            else:
                if prev >= 0 and r[j, 0] > 0.7:  # complement of the last category
                    c = comp[prev][int(r[j, 1] * len(comp[prev]))]
                else:
                    c = min(int(np.searchsorted(pref_cum[u], r[j, 2])), len(cats) - 1)
                b = blocks[c]
                pos = min(int(np.searchsorted(cum[u, b], r[j, 3])), len(b) - 1)
                item = int(b[pos])
            prev = int(cat_of[item])
            owned.append(item)
            users.append(u)
            items.append(item)
        times.append(np.sort(rng.uniform(0, days, k)))
    day = np.concatenate(times)
    start = pd.Timestamp(cfg.start)
    users_a, items_a = np.array(users), np.array(items)
    return pd.DataFrame(
        {
            "order_id": [f"O{i + 1:07d}" for i in range(len(users_a))],
            "user_id": [f"C{u + 1:06d}" for u in users_a],
            "item_id": products["item_id"].to_numpy()[items_a],
            "order_ts": start + pd.to_timedelta(np.floor(day * 86400), unit="s"),
            "price": price[items_a],
        }
    )


def _sentence(aspect: str, pol: str, spec: dict, flavor: str, rng: np.random.Generator) -> str:
    options = HARD[(aspect, pol)] if rng.random() < HARD_SHARE else TEMPLATES[(aspect, pol)]
    return options[int(rng.integers(len(options)))].format(flavor=flavor, noun=spec["noun"])


def _reviews(
    cfg: SynthConfig, products: pd.DataFrame, orders: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    first = orders.drop_duplicates(["user_id", "item_id"])
    first = first[rng.random(len(first)) < cfg.review_rate].reset_index(drop=True)
    prod = products.set_index("item_id")

    # the reformulated product: the best-selling protein powder
    protein = products[products["category"] == "Protein Powder"]["item_id"]
    sales = orders[orders["item_id"].isin(protein)]["item_id"].value_counts()
    changed = str(sales.index[0])
    change_ts = pd.Timestamp(cfg.start) + pd.DateOffset(months=cfg.reformulation_month - 1)
    new_taste = -0.7

    idx = products.reset_index().set_index("item_id")["index"]
    item_pos = idx.loc[first["item_id"]].to_numpy()
    q_all = products[[f"q_{a}" for a in ASPECTS]].to_numpy()
    cat_all = products["category"].to_numpy()
    flavor_all = products["flavor"].to_numpy()
    review_ts = first["order_ts"] + pd.to_timedelta(rng.uniform(3, 25, len(first)), unit="D")
    after = (first["item_id"].to_numpy() == changed) & (review_ts >= change_ts).to_numpy()
    draws = rng.random((len(first), len(ASPECTS), 2))
    titles = ["Do not buy", "Disappointed", "It's okay", "Pretty good", "Love it"]
    reviews, labels = [], []
    for i in range(len(first)):
        spec = CATEGORIES[cat_all[item_pos[i]]]
        q = q_all[item_pos[i]].copy()
        if after[i]:
            q[0] = new_taste
        mentioned = [
            (a, "positive" if draws[i, k, 1] < 1 / (1 + np.exp(-3.0 * q[k])) else "negative")
            for k, a in enumerate(ASPECTS)
            if draws[i, k, 0] < spec["mention"][k]
        ]
        sentences = []
        rest = mentioned
        if len(mentioned) >= 2 and rng.random() < 0.3:  # two aspects in one sentence
            (a1, p1), (a2, p2) = mentioned[0], mentioned[1]
            joiner = " and " if p1 == p2 else " but "
            c1 = CLAUSES[(a1, p1)]
            sentences.append(c1[0].upper() + c1[1:] + joiner + CLAUSES[(a2, p2)] + ".")
            rest = mentioned[2:]
        sentences += [_sentence(a, pol, spec, flavor_all[item_pos[i]], rng) for a, pol in rest]
        if rng.random() < 0.5:
            sentences.append(FILLERS[int(rng.integers(len(FILLERS)))])
        rng.shuffle(sentences)
        score = 4.3 + sum(0.25 if pol == "positive" else -1.15 for _, pol in mentioned)
        rating = int(np.clip(round(score + rng.normal(0, 0.55)), 1, 5))
        review_id = f"R{i + 1:06d}"
        reviews.append((review_id, rating, titles[rating - 1],
                        " ".join(sentences) if sentences else "Good."))  # fmt: skip
        got = dict(mentioned)
        labels.append({"review_id": review_id, **{a: got.get(a, "none") for a in ASPECTS}})
    rev = pd.DataFrame(reviews, columns=["review_id", "rating", "title", "text"])
    rev.insert(1, "user_id", first["user_id"].to_numpy())
    rev.insert(2, "item_id", first["item_id"].to_numpy())
    rev["review_ts"] = review_ts.dt.floor("s").to_numpy()
    rev["verified_purchase"] = True
    truth = {
        "reformulated_item": changed,
        "reformulated_title": str(prod.loc[changed, "title"]),
        "reformulation_month": str(change_ts.date())[:7],
        "taste_before": round(float(prod.loc[changed, "q_taste"]), 3),
        "taste_after": new_taste,
    }
    return rev, pd.DataFrame(labels), truth


def generate(cfg: SynthConfig | None = None) -> SynthData:
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    products = _catalogue(cfg, rng)
    orders = _orders(cfg, products, rng)
    reviews, review_labels, change = _reviews(cfg, products, orders, rng)
    quality = products[["item_id", *(f"q_{a}" for a in ASPECTS)]].copy()
    public = products.drop(columns=["popularity", *(f"q_{a}" for a in ASPECTS)])
    truth = {
        "store": "Fernway Market (fictional)",
        "seed": cfg.seed,
        "reformulation": change,
        "aspects": list(ASPECTS),
    }
    tables = {"products": public, "orders": orders, "reviews": reviews}
    # ground truth lives in separate tables that the API and agent never query
    tables["truth_review_labels"] = review_labels
    tables["truth_aspect_quality"] = quality
    return SynthData(tables=tables, truth=truth)


def write(data: SynthData, db_path: str | Path) -> Path:
    """Write all tables to a fresh DuckDB file (atomically) and the truth summary next to it."""
    import duckdb

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp.duckdb")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    try:
        for name, df in data.tables.items():
            con.register("_df", df)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM _df")
            con.unregister("_df")
    finally:
        con.close()
    tmp.replace(db_path)
    Path(db_path).with_name("truth.json").write_text(json.dumps(data.truth, indent=2))
    return db_path
