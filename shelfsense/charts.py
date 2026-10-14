"""README charts, rendered from ``results.json`` and the warehouse with matplotlib.

Palette: a validated categorical order (adjacent pairs pass colour-blind separation); slots 3-4
are below 3:1 contrast on the light surface, so every chart carries a legend or direct labels.
The heatmap uses a diverging pair (orange negative, blue positive) around a neutral midpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from shelfsense import pipeline  # noqa: E402
from shelfsense.synth import ASPECTS  # noqa: E402
from shelfsense.warehouse import get_warehouse  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
)  # fmt: skip
NEUTRAL = "#f0efec"
MODEL_NAMES = {"popularity": "Popularity", "item_item": "Item-item", "svd": "SVD",
               "content": "Content (TF-IDF)", "hybrid": "Hybrid (learned)",
               "hybrid + rules": "Hybrid + business rules"}  # fmt: skip


def _axes(title: str, subtitle: str, size: tuple[float, float] = (8.0, 4.4), ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=size, dpi=200, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes[0]:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8.5, length=0)
        ax.set_axisbelow(True)
    fig.text(0.012, 0.965, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.905, subtitle, fontsize=9, color=INK2, va="top")
    return fig, axes[0]


def _q(text: str) -> str:
    """A string literal for SQL (real category names can contain quotes)."""
    return "'" + str(text).replace("'", "''") + "'"


def _save(fig: Any, path: Path) -> Path:
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def recommender_chart(res: dict, path: Path) -> Path:
    rows = pd.DataFrame(res["recommender"]["models"]).set_index("model")
    order = [m for m in MODEL_NAMES if m in rows.index]
    rows = rows.loc[order[::-1]]
    fig, (left, right) = _axes(
        "The learned hybrid has the best recall, from a narrower slice of the catalogue",
        f"Offline test on {res['recommender']['test_shoppers']:,} shoppers: new products bought "
        f"in the {res['recommender']['horizon_days']} days after {res['recommender']['cutoff']}",
        size=(8.6, 4.0), ncols=2,
    )  # fmt: skip
    y = np.arange(len(rows))
    hybrid = [m.startswith("hybrid") for m in rows.index]
    for ax, col, label in ((left, "recall_at_10", "Recall@10"), (right, "coverage",
                           "Catalogue coverage")):  # fmt: skip
        colors = [SLOTS[0] if h else "#a9c7ec" for h in hybrid]
        ax.barh(y, rows[col], color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, rows[col], strict=True):
            ax.text(v, yi, f"  {v:.1%}" if col == "coverage" else f"  {v:.3f}", va="center",
                    fontsize=8.5, color=INK2)  # fmt: skip
        ax.set_xlim(0, rows[col].max() * 1.28)
        ax.set_title(label, fontsize=9.5, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    left.set_yticks(y, [MODEL_NAMES[m] for m in rows.index])
    right.set_yticks([])
    right.spines["left"].set_visible(False)
    fig.subplots_adjust(top=0.78, wspace=0.08)
    return _save(fig, path)


def labels_chart(res: dict, path: Path) -> Path:
    a = res["aspects"]
    fig, (ax,) = _axes(
        "The student model is only as good as its labels",
        "Macro-F1 against the true labels on 10,000 unseen reviews, by number of labelled "
        "reviews it learns from",
    )
    curves = [("perfect_labels_curve", SLOTS[0], "Labels from a perfect labeller (the truth)"),
              ("keyword_labels_curve", SLOTS[1], "Labels from the keyword rules")]  # fmt: skip
    for key, color, label in curves:
        if key not in a:
            continue
        c = pd.DataFrame(a[key])
        ax.plot(c["labels"], c["f1"], color=color, lw=2, marker="o", ms=6, mec=SURFACE,
                mew=1.5, label=label)  # fmt: skip
        ax.annotate(f"{c['f1'].iloc[-1]:.3f}", (c["labels"].iloc[-1], c["f1"].iloc[-1]),
                    xytext=(7, 0), textcoords="offset points", va="center", fontsize=8.5,
                    color=INK2)  # fmt: skip
    if "labeller_f1" in a:
        ax.axhline(a["labeller_f1"]["mean"], color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.annotate("keyword rules on their own", (250, a["labeller_f1"]["mean"]),
                    xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK2)  # fmt: skip
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([250, 500, 1000, 2000], ["250", "500", "1,000", "2,000"])
    ax.set_xlabel("Labelled reviews (log scale)", color=INK2, fontsize=8.5)
    ax.set_ylabel("Macro-F1", color=INK2, fontsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_ylim(0.8, 1.01)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
    return _save(fig, path)


def monitoring_chart(res: dict, path: Path) -> Path:
    alerts = pd.DataFrame(res["monitoring"]["alerts"])
    wh = get_warehouse()
    if alerts.empty:  # nothing flagged: show the most-reviewed product instead
        top = wh.query("SELECT item_id, aspect FROM product_aspects ORDER BY mentions DESC LIMIT 1")
        item, aspect = top["item_id"].iloc[0], top["aspect"].iloc[0]
    else:
        item, aspect = alerts["item_id"].iloc[0], alerts["aspect"].iloc[0]
    prod = wh.query(f"SELECT title, category FROM products WHERE item_id = {_q(item)}").iloc[0]
    own = wh.query(
        "SELECT month, net_sentiment, mentions FROM aspect_monthly "
        f"WHERE item_id = {_q(item)} AND aspect = {_q(aspect)} ORDER BY month"
    )
    cat = wh.query(
        "SELECT m.month, (SUM(m.positive) - SUM(m.negative)) / SUM(m.mentions) AS net_sentiment "
        "FROM aspect_monthly m JOIN products p ON p.item_id = m.item_id "
        f"WHERE p.category = {_q(prod['category'])} AND m.aspect = {_q(aspect)} "
        f"AND m.item_id <> {_q(item)} GROUP BY m.month ORDER BY m.month"
    )
    if (own["mentions"] >= 5).sum() >= 3:  # hide months with a handful of mentions
        own = own[own["mentions"] >= 5]
    first = alerts["month"].min() if len(alerts) else None
    fig, (ax,) = _axes(
        f"Monitoring flagged {prod['title']} in its first bad month" if first else
        f"{prod['title']}: {aspect} sentiment",
        f"Net {aspect} sentiment per month (-1 all negative, +1 all positive) from the review "
        "model, against the rest of its category",
    )  # fmt: skip
    for df, color, label in ((own, SLOTS[0], prod["title"]),
                             (cat, SLOTS[1], f"Other {prod['category'].lower()}")):  # fmt: skip
        x = pd.to_datetime(df["month"])
        ax.plot(x, df["net_sentiment"], color=color, lw=2, label=label)
        ax.annotate(label, (x.iloc[-1], df["net_sentiment"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK2)  # fmt: skip
    if first:
        hit = alerts[alerts["item_id"] == item]
        xs = pd.to_datetime(hit["month"])
        ys = own.set_index("month").loc[hit["month"], "net_sentiment"]
        ax.plot(xs, ys, "v", ms=8, color=SLOTS[0], mec=INK, mew=1, label="Alert")
        top = hit.iloc[0]
        ax.annotate(f"alert: {top['negative_share']:.0%} negative vs "
                    f"{top['baseline_negative_share']:.0%} before (z {top['z']:.1f})",
                    (xs.iloc[0], ys.iloc[0]), xytext=(-22, 40), textcoords="offset points",
                    ha="right", fontsize=8.5, color=INK,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})  # fmt: skip
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower left")
    return _save(fig, path)


def brand_chart(path: Path) -> Path | None:
    from shelfsense.queries import run_saved

    df = run_saved("aspect_scorecard_by_brand", get_warehouse())
    if df.empty:  # no brand has enough reviewers to show
        return None
    top = df.groupby("brand")["mentions"].sum().nlargest(15).index  # real catalogues: many
    df = df[df["brand"].isin(top)]
    grid = df.pivot(index="brand", columns="aspect", values="net_sentiment")
    grid = grid.reindex(columns=[a for a in ASPECTS if a in grid.columns])
    grid = grid.loc[grid.mean(axis=1).sort_values(ascending=False).index]
    lim = float(np.nanmax(np.abs(grid.to_numpy())))
    cmap = LinearSegmentedColormap.from_list("div", [SLOTS[1], NEUTRAL, SLOTS[0]])
    fig, (ax,) = _axes(
        "What reviewers say, brand by brand" if len(top) < 15 else
        "What reviewers say about the 15 most-reviewed brands",
        "Net sentiment per aspect: share of positive minus negative mentions (blue positive, "
        "orange negative)", size=(8.0, 5.2),
    )  # fmt: skip
    ax.imshow(grid.to_numpy(), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(grid.to_numpy()):
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                color=INK if abs(v) < 0.6 * lim else SURFACE)  # fmt: skip
    ax.set_xticks(range(grid.shape[1]), [c.capitalize() for c in grid.columns])
    ax.set_yticks(range(grid.shape[0]), grid.index)
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(top=0.8)
    return _save(fig, path)


def render(out: str | Path = "docs/img") -> list[Path]:
    res = pipeline.load_results()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [recommender_chart(res, out / "recommenders.png")]
    if "perfect_labels_curve" in res["aspects"]:  # needs true labels: synthetic data only
        paths.append(labels_chart(res, out / "labels.png"))
    paths.append(monitoring_chart(res, out / "monitoring.png"))
    brands = brand_chart(out / "brand_aspects.png")
    return paths + ([brands] if brands else [])
