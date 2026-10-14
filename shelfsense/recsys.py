"""Product recommendations: candidate models, a learned re-ranker and business rules.

Candidate models, all fitted on purchases before a cutoff date:

* **popularity**: the most bought products in the last 90 days
* **item-item**: cosine similarity between products' buyer sets ("bought together")
* **SVD**: matrix factorisation of the shopper x product matrix (``TruncatedSVD``)
* **content**: TF-IDF on title, brand, category and flavour; a shopper's profile is the mean of
  what they bought

**Hybrid**: a logistic-regression re-ranker over the union of the candidates. Its inputs are each
model's rank for the product plus popularity, price, rating, category fit and brand fit. It
learns from three earlier 90-day windows, so it never sees the test period. Gradient boosting
fitted the training window better but scored lower on the next period; the linear model on
log-ranks held up out of time.

**Business rules** on the final list: no out-of-stock products, at most ``max_per_brand`` per
brand, and an optional boost for higher-margin products.

Evaluation: 10 *new* products per shopper (not bought before the cutoff) against what they
bought in the next ``horizon`` days. Metrics: recall@10, NDCG@10 and catalogue coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

K = 10
POOL = 30  # candidates taken from each model for the re-ranker
MODELS = ("popularity", "item_item", "svd", "content")
FEATURES = (
    *(f"log_rank_{m}" for m in MODELS),
    "log_popularity", "price", "price_vs_usual", "rating", "category_fit", "favourite_brand",
)  # fmt: skip


@dataclass
class Fitted:
    """Everything learned from purchases before ``cutoff``."""

    cutoff: pd.Timestamp
    users: pd.Index  # user ids, row order of X
    items: pd.Index  # item ids, column order of X
    X: sparse.csr_matrix  # users x items, 1 = bought
    pop: np.ndarray  # recent purchases per item
    sim: np.ndarray  # item x item cosine similarity
    user_f: np.ndarray  # SVD user factors
    item_f: np.ndarray  # SVD item factors
    text: sparse.csr_matrix  # items x terms (L2-normalised TF-IDF)
    cat_share: np.ndarray  # users x categories, share of purchases
    fav_brand: np.ndarray  # users: index of most bought brand
    mean_price: np.ndarray  # users: mean price paid
    item_rating: np.ndarray  # items: mean review rating before cutoff (store mean if none)


class Recommender:
    def __init__(self, products: pd.DataFrame, n_factors: int = 32, seed: int = 0) -> None:
        self.products = products.reset_index(drop=True)
        self.items = pd.Index(self.products["item_id"])
        self.item_ids: list[str] = [str(i) for i in self.items]  # fast positional lookup
        self.n_factors = n_factors
        self.seed = seed
        self.cat_codes, self.cats = pd.factorize(self.products["category"])
        self.brand_codes, self.brands = pd.factorize(self.products["brand"])
        flavor = self.products["flavor"].fillna("") if "flavor" in self.products else ""
        text = (
            self.products["title"] + " " + self.products["brand"] + " "
            + self.products["category"] + " " + flavor
        )  # fmt: skip
        self._text = normalize(TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(text))
        self.reranker: Pipeline | None = None

    # ------------------------------------------------------------------ fitting
    def fit(
        self, inter: pd.DataFrame, reviews: pd.DataFrame | None, cutoff: pd.Timestamp
    ) -> Fitted:
        past = inter[inter["ts"] < cutoff]
        users = pd.Index(sorted(past["user_id"].unique()))
        rows = users.get_indexer(past["user_id"])
        cols = self.items.get_indexer(past["item_id"])
        keep = cols >= 0
        X = sparse.csr_matrix(
            (np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(len(users), len(self.items))
        )
        X.data[:] = 1.0  # bought at least once
        recent = past[past["ts"] >= cutoff - pd.Timedelta(days=90)]
        recent_idx = self.items.get_indexer(recent["item_id"])
        pop = np.bincount(recent_idx[recent_idx >= 0], minlength=len(self.items)).astype(float)
        Xn = normalize(X, axis=0)
        sim = np.asarray((Xn.T @ Xn).todense())
        np.fill_diagonal(sim, 0.0)
        svd = TruncatedSVD(n_components=min(self.n_factors, len(self.items) - 1),
                           random_state=self.seed)  # fmt: skip
        user_f = svd.fit_transform(normalize(X))
        item_f = svd.components_.T
        n_cat = len(self.cats)
        cat_counts = np.zeros((len(users), n_cat))
        np.add.at(cat_counts, (rows[keep], self.cat_codes[cols[keep]]), 1.0)
        cat_share = cat_counts / cat_counts.sum(axis=1, keepdims=True)
        # sparse: a real catalogue can have hundreds of brands
        brand_counts = sparse.csr_matrix(
            (np.ones(keep.sum()), (rows[keep], self.brand_codes[cols[keep]])),
            shape=(len(users), len(self.brands)),
        )
        prices = self.products["price"].to_numpy()
        spend = np.bincount(rows[keep], weights=prices[cols[keep]], minlength=len(users))
        n_buys = np.bincount(rows[keep], minlength=len(users))
        rating = np.full(len(self.items), np.nan)
        if reviews is not None and len(reviews):
            r = reviews[reviews["review_ts"] < cutoff].groupby("item_id")["rating"].mean()
            idx = self.items.get_indexer(r.index)
            rating[idx[idx >= 0]] = r.to_numpy()[idx >= 0]
        seen = ~np.isnan(rating)
        rating[~seen] = rating[seen].mean() if seen.any() else 0.0
        return Fitted(cutoff, users, self.items, X, pop, sim, user_f, item_f, self._text, cat_share,
                      np.asarray(brand_counts.argmax(axis=1)).ravel(),
                      spend / np.maximum(n_buys, 1), rating)  # fmt: skip

    # ------------------------------------------------------------------ scoring
    def scores(self, f: Fitted, rows: np.ndarray) -> dict[str, np.ndarray]:
        """Scores of every item for the given user rows (users x items), per model."""
        Xu = f.X[rows]
        profile = normalize(Xu @ f.text)
        return {
            "popularity": np.tile(f.pop, (len(rows), 1)),
            "item_item": np.asarray(Xu @ f.sim),
            "svd": f.user_f[rows] @ f.item_f.T,
            "content": np.asarray((profile @ f.text.T).todense()),
        }

    def _features(self, f: Fitted, rows: np.ndarray, s: dict[str, np.ndarray],
                  cand: np.ndarray) -> np.ndarray:  # fmt: skip
        """Features (``FEATURES``) of (user, candidate) pairs; cand is users x candidate item ids.

        Ranks rather than raw scores: a heavy buyer's item-item scores are on a different scale
        from a one-order shopper's, but "3rd best for this shopper" means the same for both.
        """
        r = np.repeat(np.arange(len(rows)), cand.shape[1])
        it = cand.ravel()
        ranks = [(-s[m]).argsort(axis=1).argsort(axis=1)[r, it] for m in MODELS]
        prices = self.products["price"].to_numpy()[it]
        feats = np.column_stack(
            [
                *np.log1p(ranks),
                np.log1p(f.pop[it]),
                prices,
                prices / np.maximum(f.mean_price[rows][r], 1.0),
                f.item_rating[it],
                f.cat_share[rows][r, self.cat_codes[it]],
                (f.fav_brand[rows][r] == self.brand_codes[it]).astype(float),
            ]
        )
        return feats

    def _candidates(
        self, f: Fitted, rows: np.ndarray, s: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top ``POOL`` new items from each model, merged: (users x n item ids, mask of real ones).

        Rows have different numbers of unique candidates; shorter rows are padded with their
        first candidate, and the mask marks the padding so training can skip it.
        """
        owned = f.X[rows].toarray() > 0
        picks = []
        for m in MODELS:
            sc = np.where(owned, -np.inf, s[m])
            picks.append(np.argsort(-sc, axis=1)[:, :POOL])
        allc = np.concatenate(picks, axis=1)
        out = np.empty((len(rows), allc.shape[1]), dtype=int)
        real = np.zeros(out.shape, dtype=bool)
        for i, row in enumerate(allc):
            u = pd.unique(row)
            out[i, : len(u)] = u
            out[i, len(u) :] = u[0]
            real[i, : len(u)] = True
        return out, real

    def train_reranker(
        self, inter: pd.DataFrame, reviews: pd.DataFrame | None, cutoff: pd.Timestamp,
        horizon: int = 90, windows: int = 3, users_per_window: int = 4000, neg_per_pos: int = 10,
    ) -> None:  # fmt: skip
        """Fit the re-ranker on ``windows`` periods of ``horizon`` days that end before ``cutoff``.

        For each period the candidate models are fitted on purchases before it starts, and a
        candidate is a positive if the shopper bought it during the period. Negatives are
        sampled down to ``neg_per_pos`` per positive.
        """
        rng = np.random.default_rng(self.seed)
        Xs, ys = [], []
        for w in range(1, windows + 1):
            start = cutoff - pd.Timedelta(days=horizon * w)
            f = self.fit(inter, reviews, start)
            targets = new_items(inter, f, start, horizon)
            users = sorted(targets)
            if len(users) > users_per_window:
                users = sorted(rng.choice(users, users_per_window, replace=False))
            if not users:
                continue
            rows = f.users.get_indexer(users)
            s = self.scores(f, rows)
            cand, real = self._candidates(f, rows, s)
            bought = [set(self.items.get_indexer(list(targets[u]))) for u in users]
            y = np.array([it in bought[i] for i, c in enumerate(cand) for it in c])
            Xs.append(self._features(f, rows, s, cand)[real.ravel()])
            ys.append(y[real.ravel()])
        if not ys or not np.concatenate(ys).any():
            raise ValueError("no purchases to learn from before the cutoff; use a later cutoff")
        X, y = np.vstack(Xs), np.concatenate(ys)
        pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
        neg = rng.choice(neg, min(len(neg), neg_per_pos * len(pos)), replace=False)
        keep = np.concatenate([pos, neg])
        self.reranker = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        self.reranker.fit(X[keep], y[keep])

    def reranker_weights(self) -> dict[str, float]:
        """Standardised logistic-regression coefficients: what the hybrid relies on."""
        if self.reranker is None:
            raise ValueError("train_reranker first")
        coef = self.reranker[-1].coef_[0]
        return {name: round(float(c), 3) for name, c in zip(FEATURES, coef, strict=True)}

    # ------------------------------------------------------------------ recommending
    def recommend(
        self, f: Fitted, users: list[str], model: str = "hybrid", k: int = K,
        rules: dict[str, Any] | None = None, batch: int = 2000,
    ) -> dict[str, list[str]]:  # fmt: skip
        """Top-k new item ids per user. ``rules``: in_stock_only, max_per_brand, margin_boost."""
        out: dict[str, list[str]] = {}
        rows_all = f.users.get_indexer(users)
        for b in range(0, len(users), batch):
            rows = rows_all[b : b + batch]
            s = self.scores(f, rows)
            if model == "hybrid":
                if self.reranker is None:
                    raise ValueError("train_reranker first")
                cand, _ = self._candidates(f, rows, s)
                p = self.reranker.predict_proba(self._features(f, rows, s, cand))[:, 1]
                score = np.full((len(rows), len(self.items)), -np.inf)
                np.put_along_axis(score, cand, p.reshape(cand.shape), axis=1)
            else:
                score = s[model].astype(float)
            score = np.where(f.X[rows].toarray() > 0, -np.inf, score)
            for i, u in enumerate(users[b : b + batch]):
                out[u] = self._top(score[i], k, rules)
        return out

    def _top(self, score: np.ndarray, k: int, rules: dict[str, Any] | None) -> list[str]:
        rules = rules or {}
        sc = score.copy()
        if rules.get("in_stock_only"):
            sc[~self.products["in_stock"].to_numpy()] = -np.inf
        boost = rules.get("margin_boost", 0.0)
        if boost:
            m = self.products["margin_pct"].to_numpy()
            finite = np.isfinite(sc)
            # rank-based so the boost means the same for every model's score scale
            ranks = np.full(len(sc), np.inf)
            ranks[finite] = (-sc[finite]).argsort().argsort()
            sc = np.where(finite, -ranks * (1 - boost * (m - m.mean()) / m.std()), -np.inf)
        cap = rules.get("max_per_brand")
        picked: list[int] = []
        per_brand: dict[int, int] = {}
        for j in np.argsort(-sc):
            if not np.isfinite(sc[j]) or len(picked) == k:
                break
            b = int(self.brand_codes[j])
            if cap and per_brand.get(b, 0) >= cap:
                continue
            per_brand[b] = per_brand.get(b, 0) + 1
            picked.append(int(j))
        return [self.item_ids[j] for j in picked]

    def similar(self, f: Fitted, item_id: str, k: int = K) -> list[tuple[str, float]]:
        """Customers who bought this also bought (item-item similarity)."""
        j = self.items.get_loc(item_id)
        order = np.argsort(-f.sim[j])[:k]
        return [(self.item_ids[i], round(float(f.sim[j, i]), 4)) for i in order]


# --------------------------------------------------------------------------- evaluation


def interactions(orders: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"user_id": orders["user_id"], "item_id": orders["item_id"],
         "ts": pd.to_datetime(orders["order_ts"])}
    )  # fmt: skip


def new_items(
    inter: pd.DataFrame, f: Fitted, cutoff: pd.Timestamp, horizon: int
) -> dict[str, set[str]]:
    """Per shopper with history: products first bought in (cutoff, cutoff + horizon]."""
    w = inter[(inter["ts"] >= cutoff) & (inter["ts"] < cutoff + pd.Timedelta(days=horizon))]
    rows = f.users.get_indexer(w["user_id"])
    cols = f.items.get_indexer(w["item_id"])
    ok = (rows >= 0) & (cols >= 0)
    w, rows, cols = w[ok], rows[ok], cols[ok]
    owned = np.asarray(f.X[rows, cols]).ravel() > 0
    fresh = w[~owned]
    return {str(u): set(g) for u, g in fresh.groupby("user_id")["item_id"]}


def evaluate(
    recs: dict[str, list[str]], targets: dict[str, set[str]], n_items: int,
    margin: pd.Series | None = None,
) -> dict[str, float]:  # fmt: skip
    recall, ndcg = [], []
    seen: set[str] = set()
    margins = []
    margin_of = margin.to_dict() if margin is not None else {}
    for u, truth in targets.items():
        top = recs.get(u, [])
        hits = [1.0 if i in truth else 0.0 for i in top]
        recall.append(sum(hits) / len(truth))
        dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
        idcg = sum(1 / np.log2(r + 2) for r in range(min(len(truth), len(top) or K)))
        ndcg.append(dcg / idcg if idcg else 0.0)
        seen.update(top)
        if margin_of and top:
            margins.append(float(np.mean([margin_of[i] for i in top])))
    out = {
        "recall_at_10": round(float(np.mean(recall)), 4),
        "ndcg_at_10": round(float(np.mean(ndcg)), 4),
        "coverage": round(len(seen) / n_items, 4),
        "users": len(targets),
    }
    if margins:
        out["avg_margin_pct"] = round(float(np.mean(margins)), 4)
    return out
