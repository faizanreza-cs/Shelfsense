"""Amazon Reviews 2023 (McAuley Lab, UCSD): real reviews in, the same pipeline out.

``shelfsense fetch`` streams the Health & Household reviews and product metadata, keeps
supplements (vitamins, minerals, supplements and sports nutrition) and builds the same three
tables as the synthetic store:

* **products**: one row per product (``parent_asin``) with title, brand (the store name),
  category (the level below supplements) and price
* **reviews**: stars, title, text and time; shopper ids are replaced with ``C`` + a number
* **orders**: a review counts as a purchase (the usual proxy: the data has no orders)

To keep a typical store's size, it keeps the ``max_items`` products with the most reviews since
``since``, shoppers with at least ``min_reviews`` reviews of them, then a random sample of
shoppers down to ``max_reviews`` reviews. There is no ground truth, so the review model is
checked against star ratings instead.

Data: Hou et al., "Bridging Language and Items for Retrieval and Recommendation" (2024),
https://amazon-reviews-2023.github.io/. Check the dataset's terms before sharing derived data.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import tempfile
import time
import zlib
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shelfsense import pipeline, synth
from shelfsense.config import Settings, get_settings

BASES = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw",
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw",
)
HF = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
SUPPLEMENTS = ("Vitamins, Minerals & Supplements", "Sports Nutrition", "Diet & Sports Nutrition")
_PARENT = re.compile(rb'"parent_asin":\s*"([^"]+)"')
_TAGS = re.compile(r"<[^>]+>")


def _log(msg: str) -> None:
    print(f"[fetch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sources(kind: str, category: str) -> list[str]:
    name = (f"review_categories/{category}.jsonl" if kind == "reviews"
            else f"meta_categories/meta_{category}.jsonl")  # fmt: skip
    base = os.environ.get("SHELFSENSE_AMAZON_BASE")  # a mirror URL or a local folder
    if base:
        return [f"{base.rstrip('/')}/{name}.gz", f"{base.rstrip('/')}/{name}"]
    return [f"{b}/{name}.gz" for b in BASES] + [f"{HF}/{name}"]


def _http_lines(url: str) -> Iterator[bytes]:
    import httpx

    timeout = httpx.Timeout(60.0, read=600.0)
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
        r.raise_for_status()
        inflate = zlib.decompressobj(31) if url.endswith(".gz") else None
        rest = b""
        for chunk in r.iter_raw(1 << 20):
            lines = (rest + (inflate.decompress(chunk) if inflate else chunk)).split(b"\n")
            rest = lines.pop()
            yield from lines
        if inflate:
            rest += inflate.flush()
        if rest.strip():
            yield rest


def _file_lines(path: str) -> Iterator[bytes]:
    opener: Any = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as fh:
        yield from fh


def _lines(kind: str, category: str) -> Iterator[bytes]:
    """Lines of the first source that answers. Falls back only before the first line arrives."""
    tried = []
    for src in _sources(kind, category):
        remote = src.startswith(("http://", "https://"))
        if not remote and not Path(src).exists():
            tried.append(f"{src}: not found")
            continue
        gen = _http_lines(src) if remote else _file_lines(src)
        try:
            first = next(gen)
        except StopIteration:
            tried.append(f"{src}: empty")
            continue
        except Exception as e:  # HTTP 404/403, DNS, TLS: try the next mirror
            tried.append(f"{src}: {type(e).__name__}: {e}")
            continue
        _log(f"reading {kind} from {src}")
        yield first
        yield from gen
        return
    raise RuntimeError(f"no source for {kind} worked:\n  " + "\n  ".join(tried))


def _price(v: Any) -> float | None:
    if isinstance(v, int | float) and not isinstance(v, bool):
        return float(v) if v > 0 else None
    if isinstance(v, str):
        try:
            p = float(v.replace("$", "").replace(",", "").strip())
            return p if p > 0 else None
        except ValueError:
            return None
    return None


def _is_supplement(cats: list[str]) -> bool:
    return len(cats) >= 2 and (
        cats[1] in SUPPLEMENTS or any("Supplement" in c or "Vitamin" in c for c in cats[1:])
    )


def supplement_items(category: str) -> dict[str, dict[str, Any]]:
    """Supplement products from the metadata file: parent_asin -> title, brand, category, price."""
    items: dict[str, dict[str, Any]] = {}
    seen: Counter[str] = Counter()
    for n, line in enumerate(_lines("meta", category), 1):
        if not line.strip():
            continue
        m = json.loads(line)
        cats = [str(c) for c in (m.get("categories") or [])]
        seen[cats[1] if len(cats) > 1 else "(none)"] += 1
        if not _is_supplement(cats) or not m.get("title") or not m.get("parent_asin"):
            continue
        brand = m.get("store") or (m.get("details") or {}).get("Brand") or "Unknown"
        items[str(m["parent_asin"])] = {
            "title": " ".join(str(m["title"]).split())[:140],
            "brand": " ".join(str(brand).split())[:60],
            "category": cats[2] if len(cats) > 2 else cats[1],
            "price": _price(m.get("price")),
        }
        if n % 200_000 == 0:
            _log(f"meta: {n:,} products read, {len(items):,} supplements")
    if not items:
        top = ", ".join(f"{k} ({v:,})" for k, v in seen.most_common(15))
        raise RuntimeError(f"no supplement products found; second-level categories seen: {top}")
    _log(f"meta: {len(items):,} supplement products")
    return items


def _clean(text: Any, limit: int) -> str:
    return " ".join(_TAGS.sub(" ", str(text or "")).split())[:limit]


def _scan_reviews(category: str, items: dict[str, Any], since_ms: int, tmp: Path) -> pd.DataFrame:
    """Pass 1: keep supplement reviews since ``since`` in a local file; return their keys."""
    users, asins, stamps = [], [], []
    kept = 0
    with gzip.open(tmp, "wt", compresslevel=1) as out:
        for n, line in enumerate(_lines("reviews", category), 1):
            if n % 2_000_000 == 0:
                _log(f"reviews: {n:,} read, {kept:,} kept")
            m = _PARENT.search(line)
            if not m or m.group(1).decode() not in items:
                continue
            r = json.loads(line)
            ts = int(r.get("timestamp") or 0)
            if ts < since_ms or not r.get("user_id"):
                continue
            row = {
                "u": r["user_id"], "i": r["parent_asin"], "t": ts,
                "s": float(r.get("rating") or 0), "h": _clean(r.get("title"), 120),
                "x": _clean(r.get("text"), 1000), "v": bool(r.get("verified_purchase")),
            }  # fmt: skip
            out.write(json.dumps(row) + "\n")
            users.append(row["u"])
            asins.append(row["i"])
            stamps.append(ts)
            kept += 1
    _log(f"reviews: {kept:,} supplement reviews since the start date")
    return pd.DataFrame({"user": users, "item": asins, "ts": stamps})


def _select(keys: pd.DataFrame, max_items: int, min_reviews: int, max_reviews: int,
            seed: int) -> pd.DataFrame:  # fmt: skip
    """Top products, then repeat reviewers, then a random sample of shoppers under the cap."""
    keys = keys.drop_duplicates(["user", "item"], keep="last")
    top = keys["item"].value_counts().head(max_items).index
    k = keys[keys["item"].isin(top)]
    for _ in range(3):  # shoppers need history; products need enough shoppers
        k = k[k.groupby("user")["item"].transform("size") >= min_reviews]
        k = k[k.groupby("item")["user"].transform("size") >= 5]
    if k.empty:
        raise RuntimeError("no shoppers with enough reviews; lower min_reviews or widen since")
    if len(k) > max_reviews:
        users = k["user"].drop_duplicates().sample(frac=1.0, random_state=seed)
        sizes = k.groupby("user").size().loc[users]
        k = k[k["user"].isin(sizes.index[sizes.cumsum() <= max_reviews])]
    _log(f"selected {len(k):,} reviews: {k['user'].nunique():,} shoppers, "
         f"{k['item'].nunique():,} products")  # fmt: skip
    return k


def tables_from(
    items: dict[str, dict[str, Any]], tmp: Path, keys: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Pass 2: the selected reviews with text, as products / orders / reviews tables."""
    want = set(zip(keys["user"], keys["item"], keys["ts"], strict=True))
    rows = []
    with gzip.open(tmp, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if (r["u"], r["i"], r["t"]) in want:
                rows.append(r)
    rev = pd.DataFrame(rows).drop_duplicates(["u", "i"], keep="last")
    rev = rev.sort_values("t").reset_index(drop=True)
    shopper = {u: f"C{n + 1:06d}" for n, u in enumerate(sorted(rev["u"].unique()))}
    prod = pd.DataFrame.from_dict(items, orient="index").loc[sorted(rev["i"].unique())]
    prod = prod.rename_axis("item_id").reset_index()
    by_cat = prod.groupby("category")["price"].transform("median")
    prod["price"] = prod["price"].fillna(by_cat).fillna(prod["price"].median()).fillna(20.0)
    prod["price"] = prod["price"].round(2)
    ts = pd.to_datetime(rev["t"], unit="ms")
    price = rev["i"].map(prod.set_index("item_id")["price"])
    reviews = pd.DataFrame({
        "review_id": [f"R{n + 1:07d}" for n in range(len(rev))],
        "user_id": rev["u"].map(shopper),
        "item_id": rev["i"],
        "rating": rev["s"].round().clip(1, 5).astype(int),
        "title": rev["h"],
        "text": rev["x"].where(rev["x"].str.len() > 0, rev["h"]),
        "review_ts": ts,
        "verified_purchase": rev["v"],
    })  # fmt: skip
    orders = pd.DataFrame({
        "order_id": [f"O{n + 1:07d}" for n in range(len(rev))],
        "user_id": reviews["user_id"],
        "item_id": reviews["item_id"],
        "order_ts": ts,
        "price": price.to_numpy(),
    })  # fmt: skip
    return {"products": prod, "orders": orders, "reviews": reviews}


def fetch_and_build(
    category: str = "Health_and_Household", max_reviews: int = 300_000, max_items: int = 2000,
    since: str = "2021-01-01", min_reviews: int = 3, labeller: str = "keyword", seed: int = 0,
    settings: Settings | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Download, select, run the whole pipeline and write the database and results."""
    s = settings or get_settings()
    t0 = time.time()
    items = supplement_items(category)
    since_ms = int(pd.Timestamp(since).timestamp() * 1000)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "reviews.jsonl.gz"
        keys = _scan_reviews(category, items, since_ms, tmp)
        keys = _select(keys, max_items, min_reviews, max_reviews, seed)
        tables = tables_from(items, tmp, keys)
    _log(f"tables ready after {time.time() - t0:.0f} s; running the pipeline")
    source = f"amazon-reviews-2023/{category} (supplements)"
    results, derived, model = pipeline.analyse(tables, None, labeller=labeller, source=source)
    truth = {"source": source, "note": "real data: no ground truth"}
    synth.write(synth.SynthData(tables={**tables, **derived}, truth=truth), s.db_path)
    pipeline.save(results, model, s.db_path)
    _log(f"done in {time.time() - t0:.0f} s")
    return results


def to_amazon_format(data: synth.SynthData, folder: Path, category: str) -> None:
    """Write synthetic data in the dataset's file format (used to test this module offline)."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta_categories").mkdir(exist_ok=True)
    (folder / "review_categories").mkdir(exist_ok=True)
    p = data.tables["products"]
    with gzip.open(folder / "meta_categories" / f"meta_{category}.jsonl.gz", "wt") as fh:
        for row in p.itertuples():
            fh.write(json.dumps({
                "parent_asin": row.item_id, "title": row.title, "store": row.brand,
                "price": row.price,
                "categories": ["Health & Household", SUPPLEMENTS[0], row.category],
            }) + "\n")  # fmt: skip
        other = {"parent_asin": "B000OTHER", "title": "Paper towels",
                 "categories": ["Health & Household", "Household Supplies"]}  # fmt: skip
        fh.write(json.dumps(other) + "\n")
    r = data.tables["reviews"]
    ms = (pd.to_datetime(r["review_ts"]).astype("int64") // 1_000_000).to_numpy()
    with gzip.open(folder / "review_categories" / f"{category}.jsonl.gz", "wt") as fh:
        for row, t in zip(r.itertuples(), np.asarray(ms), strict=True):
            fh.write(json.dumps({
                "rating": float(row.rating), "title": row.title, "text": row.text,
                "asin": row.item_id, "parent_asin": row.item_id, "user_id": f"AG{row.user_id}",
                "timestamp": int(t), "helpful_vote": 0, "verified_purchase": True,
            }) + "\n")  # fmt: skip
