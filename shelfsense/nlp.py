"""Review aspect sentiment: an LLM labels a sample, a cheap model learns from it, scores the rest.

Every review gets a label per aspect (taste, digestion, effectiveness, value, packaging):
``positive``, ``negative`` or ``none`` (not mentioned).

1. **Labeller** (the teacher) labels a sample of reviews:

   * ``llm``: an LLM with structured output (Anthropic, OpenAI or Azure OpenAI)
   * ``keyword``: hand-written word lists with negation handling; offline, free, and the
     baseline the LLM has to beat
2. **Student**: TF-IDF features and one logistic regression per aspect, trained on the
   labeller's output. It scores every review in milliseconds at no cost per review.
3. **Product scores**: net sentiment per product and aspect,
   ``(positive - negative) / mentions``, overall and per month.

On synthetic data the true labels are known, so each step is scored against them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from shelfsense.synth import ASPECTS

# --------------------------------------------------------------------------- keyword labeller

ASPECT_TERMS: dict[str, set[str]] = {
    "taste": {"taste", "tastes", "tasting", "flavor", "flavour", "chalky", "aftertaste",
              "sweet", "bitter", "gritty", "texture", "bland", "delicious", "mixes", "clumps",
              "undrinkable"},
    "digestion": {"stomach", "bloating", "bloated", "gas", "cramps", "nausea", "digestion",
                  "digest", "sick", "gut"},
    "effectiveness": {"works", "work", "worked", "results", "difference", "effect", "energy",
                      "recovery", "notice", "noticed", "helped", "helps", "nothing"},
    "value": {"price", "value", "money", "expensive", "cheap", "cheaper", "overpriced", "worth",
              "penny", "cost", "deal", "pricey"},
    "packaging": {"seal", "sealed", "container", "scoop", "lid", "bottle", "packaging", "package",
                  "arrived", "leaking", "leak", "cracked", "tub", "bag", "box"},
}  # fmt: skip
POSITIVE = {
    "great", "good", "delicious", "best", "pleasant", "smooth", "easy", "easily", "gentle",
    "better", "love", "like", "works", "worked", "work", "real", "noticeably", "more", "improved",
    "definitely", "worth", "cheaper", "fair", "sturdy", "intact", "nice", "solid", "included",
    "well", "results", "effect", "helped", "helps", "value",
}  # fmt: skip
NEGATIVE = {
    "awful", "chalky", "artificial", "bitter", "terrible", "gritty", "bland", "too", "clumps",
    "aftertaste", "bloating", "gas", "cramps", "nausea", "sick", "upset", "issues", "nothing",
    "waste", "expensive", "overpriced", "broken", "cracked", "leaking", "leak", "hard", "tore",
    "pricey", "undrinkable", "complaints", "mess", "dent",
}  # fmt: skip
NEGATORS = {"not", "no", "never", "didn't", "doesn't", "don't", "can't", "couldn't", "won't",
            "isn't", "wasn't", "without", "zero"}  # fmt: skip
_CLAUSE = re.compile(r"[.!?;,]|\bbut\b|\band\b")
_WORD = re.compile(r"[a-z']+")


def keyword_labels(text: str) -> dict[str, str]:
    """Label one review with word lists: aspect words pick the aspect, sentiment words (flipped
    for three words after a negation) pick the polarity. A clause with a negation and no
    sentiment word counts as negative ("no scoop inside")."""
    score = dict.fromkeys(ASPECTS, 0)
    seen: set[str] = set()
    for clause in _CLAUSE.split(text.lower()):
        words = _WORD.findall(clause)
        aspects = [a for a in ASPECTS if ASPECT_TERMS[a].intersection(words)]
        if not aspects:
            continue
        polarity, negated_until, negation = 0, -1, False
        for i, w in enumerate(words):
            if w in NEGATORS:
                negated_until, negation = i + 3, True
                continue
            p = int(w in POSITIVE) - int(w in NEGATIVE)
            polarity += -p if i <= negated_until else p
        if polarity == 0 and negation:
            polarity = -1
        for a in aspects:
            seen.add(a)
            score[a] += polarity
    return {
        a: "positive" if score[a] > 0 else "negative" if score[a] < 0 else "none" for a in ASPECTS
    }


def label_with_keywords(reviews: pd.DataFrame) -> pd.DataFrame:
    rows = [{"review_id": rid, **keyword_labels(t)} for rid, t in zip(
        reviews["review_id"], reviews["text"], strict=True)]  # fmt: skip
    return pd.DataFrame(rows, columns=["review_id", *ASPECTS])


# --------------------------------------------------------------------------- LLM labeller

Label = Literal["positive", "negative", "none"]


class ReviewAspects(BaseModel):
    review_id: str
    taste: Label
    digestion: Label
    effectiveness: Label
    value: Label
    packaging: Label


class LabelBatch(BaseModel):
    reviews: list[ReviewAspects]


LABEL_PROMPT = """You label customer reviews of vitamins and supplements for aspect-based \
sentiment. For every review and every aspect, answer "positive", "negative" or "none" (the \
review doesn't talk about that aspect).

Aspects:
- taste: flavour, texture, smell, how it mixes
- digestion: stomach, bloating, side effects on the gut
- effectiveness: whether it works or has the intended effect
- value: price, whether it is worth the money
- packaging: seal, container, scoop, lid, condition on arrival

Judge what the reviewer means, including negation and idioms ("my stomach has never felt \
better" is positive digestion). Return one entry per review, with its review_id unchanged."""


def label_with_llm(reviews: pd.DataFrame, provider: object, batch_size: int = 20) -> pd.DataFrame:
    """Label reviews with an LLM, ``batch_size`` reviews per call (structured output).

    Reviews the model leaves out of its answer are dropped from the result rather than
    guessed; the caller sees fewer rows than it sent.
    """
    structured = getattr(provider, "structured", None)
    if structured is None:
        raise ValueError(f"provider {getattr(provider, 'name', provider)!r} can't label reviews; "
                         "use an LLM provider (anthropic, openai or azure)")  # fmt: skip
    rows: list[dict[str, str]] = []
    for start in range(0, len(reviews), batch_size):
        chunk = reviews.iloc[start : start + batch_size]
        sent = dict(zip(chunk["review_id"], chunk["text"], strict=True))
        payload = json.dumps([{"review_id": k, "text": v} for k, v in sent.items()])
        out: LabelBatch = structured(LABEL_PROMPT, payload, LabelBatch)
        rows += [r.model_dump() for r in out.reviews if r.review_id in sent]
    return pd.DataFrame(rows, columns=["review_id", *ASPECTS]).drop_duplicates("review_id")


# --------------------------------------------------------------------------- student model


class AspectModel:
    """TF-IDF on the review text and one logistic regression per aspect."""

    def __init__(self, C: float = 4.0) -> None:
        self.C = C
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.models: dict[str, LogisticRegression | str] = {}

    def fit(self, texts: Iterable[str], labels: pd.DataFrame) -> AspectModel:
        X = self.vectorizer.fit_transform(list(texts))
        for a in ASPECTS:
            y = labels[a].to_numpy()
            if len(set(y)) == 1:  # only one class in the sample: predict it
                self.models[a] = str(y[0])
            else:
                lr = LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced")
                self.models[a] = lr.fit(X, y)
        return self

    def predict(self, texts: Iterable[str]) -> pd.DataFrame:
        if not self.models:
            raise ValueError("fit the model first")
        X = self.vectorizer.transform(list(texts))
        out = {}
        for a, m in self.models.items():
            out[a] = np.full(X.shape[0], m) if isinstance(m, str) else m.predict(X)
        return pd.DataFrame(out, columns=list(ASPECTS))


# --------------------------------------------------------------------------- evaluation


def f1_by_aspect(pred: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    """Macro-F1 per aspect, joined on review_id, plus the mean.

    Averaged over the labels that occur in either column, so a small sample without, say, any
    negative packaging mentions isn't scored 0 for that class.
    """
    m = truth.merge(pred, on="review_id", suffixes=("_true", "_pred"))
    if m.empty:
        raise ValueError("no overlapping review ids")
    out = {
        a: round(float(f1_score(m[f"{a}_true"], m[f"{a}_pred"], average="macro",
                                zero_division=0)), 4)
        for a in ASPECTS
    }  # fmt: skip
    out["mean"] = round(float(np.mean([out[a] for a in ASPECTS])), 4)
    return out


# --------------------------------------------------------------------------- product scores


def long_labels(labels: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """One row per review and mentioned aspect: review_id, item_id, month, aspect, label."""
    info = reviews[["review_id", "item_id", "review_ts"]].copy()
    info["month"] = pd.to_datetime(info["review_ts"]).dt.strftime("%Y-%m")
    long = labels.melt(id_vars="review_id", value_vars=list(ASPECTS), var_name="aspect",
                       value_name="label")  # fmt: skip
    long = long[long["label"] != "none"]
    return long.merge(info.drop(columns="review_ts"), on="review_id")


def _net(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = df.assign(positive=df["label"].eq("positive"), negative=df["label"].eq("negative"))
    g = g.groupby(keys, as_index=False)[["positive", "negative"]].sum()
    g["mentions"] = g["positive"] + g["negative"]
    g["net_sentiment"] = ((g["positive"] - g["negative"]) / g["mentions"]).round(4)
    return g


def product_aspects(long: pd.DataFrame) -> pd.DataFrame:
    """Net sentiment per product and aspect over all reviews."""
    return _net(long, ["item_id", "aspect"])


def aspect_monthly(long: pd.DataFrame) -> pd.DataFrame:
    """Net sentiment per product, aspect and month."""
    return _net(long, ["item_id", "aspect", "month"])


def quality_correlation(
    scores: pd.DataFrame, quality: pd.DataFrame, min_mentions: int = 20
) -> dict[str, float]:
    """Spearman correlation of product net sentiment with the hidden quality, per aspect."""
    q = quality.melt(id_vars="item_id", var_name="aspect", value_name="quality")
    q["aspect"] = q["aspect"].str.removeprefix("q_")
    m = scores[scores["mentions"] >= min_mentions].merge(q, on=["item_id", "aspect"])
    return {
        a: round(float(g["net_sentiment"].corr(g["quality"], method="spearman")), 3)
        for a, g in m.groupby("aspect")
        if len(g) >= 10
    }
