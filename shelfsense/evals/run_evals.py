"""Evaluate the Insights Copilot on golden questions.

Metrics
-------
* **tool_accuracy**   - the agent called at least one of the expected tools
* **answer_accuracy** - every ground-truth fact (number or label) appears in the answer
* **tool_error_rate** - questions where a tool returned an error
* latency and token usage per question

Run ``python -m shelfsense.evals.run_evals`` (mock provider in CI) or ``--provider anthropic`` (or
openai / azure) to measure a real model. Writes ``reports/eval_results_<provider>.json`` and
``reports/eval_summary_<provider>.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from shelfsense import pipeline
from shelfsense.agent.core import Agent
from shelfsense.agent.provider import get_provider
from shelfsense.queries import run_saved, saved_queries
from shelfsense.warehouse import get_warehouse

GOLDEN = Path(__file__).with_name("golden.yaml")
Fact = float | int | str


# --------------------------------------------------------------------------- ground truth


def _saved(name: str) -> pd.DataFrame:
    return run_saved(name, get_warehouse())


def _query_names() -> list[Fact]:
    names: list[Fact] = [n for n in saved_queries()]
    return names[:3]


def _sql(sql: str, col: str) -> Fact:
    v = get_warehouse().query(sql)[col].iloc[0]
    return v if isinstance(v, str) else float(v)


def _model(name: str, key: str) -> Fact:
    rows = pipeline.load_results()["recommender"]["models"]
    return next(r[key] for r in rows if r["model"] == name)


def _row(name: str, i: int = 0) -> pd.Series:
    return _saved(name).iloc[i]


def _brands() -> pd.DataFrame:
    df = _saved("aspect_scorecard_by_brand")
    df["w"] = df["net_sentiment"] * df["mentions"]
    g = df.groupby("brand")
    return (g["w"].sum() / g["mentions"].sum()).round(3).sort_values(ascending=False).to_frame("v")


def _vitalis_weakest() -> list[Fact]:
    df = _saved("aspect_scorecard_by_brand")
    store = df.groupby("aspect").apply(
        lambda g: (g["net_sentiment"] * g["mentions"]).sum() / g["mentions"].sum(),
        include_groups=False,
    )
    v = df[df["brand"] == "Vitalis Lab"].set_index("aspect")["net_sentiment"]
    gap = (v - store.round(3)).round(3).sort_values()
    return [str(gap.index[0]), float(gap.iloc[0])]


def _aspect(item: str, aspect: str, key: str) -> Fact:
    return _sql(f"SELECT {key} AS v FROM product_aspects WHERE item_id = '{item}' "
                f"AND aspect = '{aspect}'", "v")  # fmt: skip


def _first_alert() -> pd.Series:
    return get_warehouse().query("SELECT * FROM aspect_alerts ORDER BY month, z DESC").iloc[0]


def _similar(item: str) -> list[Fact]:
    m = pipeline.load_model()
    first = m["recommender"].similar(m["fitted"], item, 1)[0][0]
    return [_sql(f"SELECT title AS t FROM products WHERE item_id = '{first}'", "t")]


def _recommend(user: str) -> list[Fact]:
    m = pipeline.load_model()
    first = m["recommender"].recommend(m["fitted"], [user], "hybrid", rules=m["rules"])[user][0]
    return [_sql(f"SELECT title AS t FROM products WHERE item_id = '{first}'", "t")]


def _latest_drift() -> list[Fact]:
    wh = get_warehouse()
    month = wh.query("SELECT MAX(month) AS m FROM drift_monthly")["m"].iloc[0]
    top = wh.query(f"SELECT psi FROM drift_monthly WHERE month = '{month}' ORDER BY psi DESC")
    return [str(month), float(top["psi"].iloc[0])]


def _curve() -> list[Fact]:
    best = pipeline.load_results()["aspects"]["perfect_labels_curve"][-1]
    return [int(best["labels"]), float(best["f1"])]


TRUTH: dict[str, Callable[[], list[Fact]]] = {
    "saved_query_names": _query_names,
    "top_repeat_category": lambda: [str(_row("repeat_purchase_by_category")["category"]),
                                    float(_row("repeat_purchase_by_category")["repeat_rate"])],
    "top_pair": lambda: [str(_row("bought_together")["product_a"]),
                         str(_row("bought_together")["product_b"])],
    "first_month_sales": lambda: [str(_row("monthly_sales")["category"]),
                                  int(_row("monthly_sales")["orders"])],
    "first_trend_row": lambda: [str(_row("category_aspect_trend")["category"]),
                                float(_row("category_aspect_trend")["net_sentiment"])],
    "best_brand": lambda: [str(_brands().index[0]), float(_brands()["v"].iloc[0])],
    "vitalis_weakest": _vitalis_weakest,
    "p0015_taste": lambda: [int(_aspect("P0015", "taste", "mentions")),
                            float(_aspect("P0015", "taste", "net_sentiment"))],
    "p0015_similar": lambda: _similar("P0015"),
    "c000123_first": lambda: _recommend("C000123"),
    "hybrid_recall": lambda: ["hybrid", _model("hybrid", "recall_at_10")],
    "rules_tradeoff": lambda: [_model("hybrid + rules", "recall_at_10"),
                               _model("hybrid + rules", "avg_margin_pct")],
    "popularity_coverage": lambda: ["popularity", _model("popularity", "coverage")],
    "latest_drift": _latest_drift,
    "alert_product": lambda: [str(_first_alert()["item_id"]), str(_first_alert()["aspect"])],
    "alert_month": lambda: [str(_first_alert()["month"])],
    "student_f1": lambda: [float(pipeline.load_results()["aspects"]["student_f1"]["mean"])],
    "label_curve": _curve,
    "sql_shoppers": lambda: [_sql("SELECT COUNT(DISTINCT user_id) AS n FROM orders", "n")],
    "sql_top_category": lambda: [_sql(
        "SELECT p.category AS c FROM orders o JOIN products p ON p.item_id = o.item_id "
        "GROUP BY 1 ORDER BY SUM(o.price) DESC LIMIT 1", "c")],
    "sql_five_stars": lambda: [_sql("SELECT COUNT(*) AS n FROM reviews WHERE rating = 5", "n")],
}  # fmt: skip


# --------------------------------------------------------------------------- grading

_NUM = re.compile(r"(-?\d[\d,]*\.?\d*)\s*([kKmM%]?)")


def extract_numbers(text: str) -> list[float]:
    out = []
    for raw, suffix in _NUM.findall(text):
        try:
            n = float(raw.replace(",", ""))
        except ValueError:
            continue
        if suffix in "kK" and suffix:
            n *= 1_000
        elif suffix in "mM" and suffix:
            n *= 1_000_000
        out.append(n)
    return out


def fact_present(fact: Fact, answer: str) -> bool:
    if isinstance(fact, str):
        return fact.lower() in answer.lower()
    nums = extract_numbers(answer)
    v = float(fact)
    if abs(v) <= 1.5 and not float(v).is_integer():  # a rate / share
        return any(abs(n - v) <= 0.0015 or abs(n - 100 * v) <= 0.15 for n in nums)
    tol = max(abs(v) * 0.005, 0.6)  # counts and volumes: 0.5% (allows "62.7K")
    return any(abs(n - v) <= tol for n in nums)


@dataclass
class EvalRow:
    id: str
    question: str
    expected_tools: list[str]
    tools_used: list[str]
    tool_ok: bool
    facts: list[Fact]
    facts_found: list[bool]
    answer_ok: bool
    tool_error: bool
    latency_ms: int
    input_tokens: int
    output_tokens: int
    answer: str


def run(provider_name: str = "mock", only: list[str] | None = None) -> list[EvalRow]:
    cases = yaml.safe_load(GOLDEN.read_text())
    if only:
        cases = [c for c in cases if c["id"] in only]
    agent = Agent(get_provider(provider_name))
    rows = []
    for case in cases:
        res = agent.ask(case["question"])
        facts = TRUTH[case["truth"]]()
        found = [fact_present(f, res.answer) for f in facts]
        rows.append(
            EvalRow(
                id=case["id"],
                question=case["question"],
                expected_tools=case["expected_tools"],
                tools_used=res.tools_used,
                tool_ok=bool(set(case["expected_tools"]) & set(res.tools_used)),
                facts=facts,
                facts_found=found,
                answer_ok=all(found),
                tool_error=any(s.is_error for s in res.steps),
                latency_ms=res.latency_ms,
                input_tokens=res.usage.get("input_tokens", 0),
                output_tokens=res.usage.get("output_tokens", 0),
                answer=res.answer,
            )
        )
        mark = "PASS" if rows[-1].tool_ok and rows[-1].answer_ok else "FAIL"
        print(f"[{mark}] {case['id']:<20} tools={res.tools_used} facts={found}", flush=True)
    return rows


def summarise(rows: list[EvalRow], provider: str) -> dict[str, Any]:
    n = len(rows)
    return {
        "provider": provider,
        "questions": n,
        "tool_accuracy": round(sum(r.tool_ok for r in rows) / n, 4),
        "answer_accuracy": round(sum(r.answer_ok for r in rows) / n, 4),
        "pass_rate": round(sum(r.tool_ok and r.answer_ok for r in rows) / n, 4),
        "tool_error_rate": round(sum(r.tool_error for r in rows) / n, 4),
        "median_latency_ms": int(statistics.median(r.latency_ms for r in rows)),
        "total_input_tokens": sum(r.input_tokens for r in rows),
        "total_output_tokens": sum(r.output_tokens for r in rows),
    }


def write_reports(rows: list[EvalRow], summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = summary["provider"]
    (out_dir / f"eval_results_{tag}.json").write_text(
        json.dumps({"summary": summary, "rows": [r.__dict__ for r in rows]}, indent=2, default=str)
    )
    lines = [
        f"# Eval summary ({tag})",
        "",
        "| metric | value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in summary.items()],
        "",
        "| id | tool ok | answer ok | tools used |",
        "|---|---|---|---|",
        *[
            f"| {r.id} | {'✅' if r.tool_ok else '❌'} | {'✅' if r.answer_ok else '❌'} "
            f"| {', '.join(r.tools_used)} |"
            for r in rows
        ],
    ]
    (out_dir / f"eval_summary_{tag}.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--provider", default="mock", help="mock | anthropic | openai | azure")
    p.add_argument("--only", nargs="*", help="run only these case ids")
    p.add_argument("--out", default="reports")
    p.add_argument("--min-pass-rate", type=float, default=0.0, help="exit 1 below this")
    a = p.parse_args(argv)
    rows = run(a.provider, a.only)
    summary = summarise(rows, a.provider)
    write_reports(rows, summary, Path(a.out))
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass_rate"] >= a.min_pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
