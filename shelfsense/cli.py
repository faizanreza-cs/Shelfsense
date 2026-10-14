"""Command line: ``shelfsense build | label | monitor | recommend | query | sql | ask |
export-powerbi | charts | load | fetch``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shelfsense.config import get_settings


def _print_table(rows: list[dict]) -> None:
    import pandas as pd

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(pd.DataFrame(rows).to_string(index=False))


def _summary(res: dict) -> str:
    rec = {r["model"]: r["recall_at_10"] for r in res["recommender"]["models"]}
    a = res["aspects"]
    return json.dumps(
        {
            "data": res["data"],
            "recall_at_10": rec,
            "aspects": {k: a[k] for k in ("labeller", "labelled_reviews", "student_f1") if k in a},
            "monitoring": res["monitoring"].get("truth_check", res["monitoring"]["max_psi"]),
        },
        indent=2,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="shelfsense", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="generate the store, run every model, write the results")
    b.add_argument(
        "--labels", type=Path, help="CSV of review labels to train on (shelfsense label)"
    )
    b.add_argument("--labeller", default="keyword", choices=["keyword", "llm"])
    b.add_argument("--label-sample", type=int, default=2000)
    lb = sub.add_parser("label", help="label a sample of reviews (LLM or keywords) into a CSV")
    lb.add_argument("--labeller", default="llm", choices=["keyword", "llm"])
    lb.add_argument("-n", type=int, default=200, help="reviews to label")
    lb.add_argument("--out", type=Path, default=Path("data/labels.csv"))
    m = sub.add_parser("monitor", help="drift and alerts for a month; exits 1 on drift or alerts")
    m.add_argument("--month", help="YYYY-MM (default: latest)")
    r = sub.add_parser("recommend", help="next products for a shopper")
    r.add_argument("user_id")
    r.add_argument("-k", type=int, default=10)
    r.add_argument("--no-rules", action="store_true", help="skip the business rules")
    q = sub.add_parser("query", help="run a saved query (no name: list them)")
    q.add_argument("name", nargs="?")
    s = sub.add_parser("sql", help="run a guarded read-only Snowflake SQL query")
    s.add_argument("sql")
    a = sub.add_parser("ask", help="ask the Insights Copilot a question")
    a.add_argument("question")
    a.add_argument("--provider", default=None, help="anthropic | openai | azure | mock")
    e = sub.add_parser("export-powerbi", help="write the Power BI star schema as CSV files")
    e.add_argument("--out", default="powerbi/data")
    c = sub.add_parser("charts", help="render the README charts")
    c.add_argument("--out", default="docs/img")
    sub.add_parser("load", help="copy the tables to Snowflake (SNOWFLAKE_* variables)")
    f = sub.add_parser("fetch", help="download Amazon Reviews 2023 and run everything on it")
    f.add_argument("--category", default="Health_and_Household")
    f.add_argument("--max-reviews", type=int, default=300_000)
    f.add_argument("--max-items", type=int, default=2000)
    f.add_argument("--since", default="2021-01-01")
    f.add_argument("--min-reviews", type=int, default=3, help="reviews per shopper")
    f.add_argument("--labeller", default="keyword", choices=["keyword", "llm"])
    args = p.parse_args(argv)

    if args.cmd == "build":
        import pandas as pd

        from shelfsense.pipeline import build

        labels = pd.read_csv(args.labels) if args.labels else None
        res = build(labeller=args.labeller, label_sample=args.label_sample, labels=labels)
        print(_summary(res))
        print(f"wrote {get_settings().db_path}, results.json and recommender.pkl")
        return 0
    if args.cmd == "label":
        return _label(args.labeller, args.n, args.out)
    if args.cmd == "fetch":
        from shelfsense.amazon import fetch_and_build

        res = fetch_and_build(
            args.category,
            args.max_reviews,
            args.max_items,
            args.since,
            args.min_reviews,
            args.labeller,
        )
        print(_summary(res))
        return 0

    from shelfsense.agent import tools

    if args.cmd == "monitor":
        out, err = tools.execute("monitoring_report", {"month": args.month} if args.month else {})
        if err:
            print(json.loads(out)["error"], file=sys.stderr)
            return 1
        payload = json.loads(out)
        _print_table(payload["rows"])
        month = payload["month"]
        hits = [x for x in payload["alerts"] if x["month"] == month]
        drift = [x for x in payload["rows"] if x["status"] == "drift"]
        for x in hits:
            share = f"{x['negative_share']:.0%} vs {x['baseline_negative_share']:.0%}"
            print(f"ALERT {x['item_id']} {x['title']}: {x['aspect']} negative {share} (z {x['z']})")
        print(f"\n{month}: {len(drift)} drifted metrics, {len(hits)} product alerts")
        return 1 if (hits or drift) else 0
    if args.cmd == "recommend":
        out, err = tools.execute(
            "recommend_for_shopper",
            {"user_id": args.user_id, "k": args.k, "apply_rules": not args.no_rules},
        )
    elif args.cmd == "query":
        out, err = tools.execute(
            "run_saved_query" if args.name else "list_saved_queries",
            {"name": args.name} if args.name else {},
        )
    elif args.cmd == "sql":
        out, err = tools.execute("run_sql", {"sql": args.sql})
    elif args.cmd == "ask":
        from shelfsense.agent.core import Agent
        from shelfsense.agent.provider import get_provider

        reply = Agent(get_provider(args.provider)).ask(args.question)
        print(reply.answer)
        print(f"\n[{reply.provider} | tools: {', '.join(reply.tools_used) or 'none'} | "
              f"{reply.latency_ms} ms]")  # fmt: skip
        return 0
    elif args.cmd == "export-powerbi":
        from shelfsense.powerbi import export

        for path in export(args.out):
            print(f"wrote {path}")
        return 0
    elif args.cmd == "charts":
        try:
            from shelfsense.charts import render
        except ImportError:
            print("charts need matplotlib: pip install 'shelfsense[charts]'")
            return 1

        for path in render(args.out):
            print(f"wrote {path}")
        return 0
    else:  # load
        from shelfsense.warehouse import TABLES, DuckDBWarehouse, SnowflakeWarehouse

        local, remote = DuckDBWarehouse(get_settings().db_path), SnowflakeWarehouse()
        for t in TABLES:
            n = remote.write(t, local.query(f"SELECT * FROM {t}"))
            print(f"{t}: {n:,} rows")
        return 0

    payload = json.loads(out)
    if err:
        print(payload["error"], file=sys.stderr)
        return 1
    _print_table(payload["rows"])
    if payload.get("note"):
        print(f"\n{payload['note']}")
    return 0


def _label(labeller: str, n: int, out: Path) -> int:
    """Label ``n`` reviews and save them; on synthetic data, also score them against the truth."""
    from shelfsense import nlp
    from shelfsense.warehouse import DuckDBWarehouse

    wh = DuckDBWarehouse(get_settings().db_path)
    reviews = wh.query("SELECT review_id, text FROM reviews").sample(n, random_state=1)
    if labeller == "llm":
        from shelfsense.agent.provider import get_provider

        labels = nlp.label_with_llm(reviews, get_provider())
    else:
        labels = nlp.label_with_keywords(reviews)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out, index=False)
    print(f"wrote {len(labels)} labels to {out}")
    try:  # the truth table is only there for the synthetic store (not exposed via the API)
        truth = wh.query("SELECT * FROM truth_review_labels")
    except Exception:
        return 0
    print(json.dumps({"macro_f1_vs_truth": nlp.f1_by_aspect(labels, truth)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
