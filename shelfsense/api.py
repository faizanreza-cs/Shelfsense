"""REST API: recommendations, review insights, monitoring, metrics and the Insights Copilot.

Run: ``uvicorn shelfsense.api:app``. When ``SHELFSENSE_API_TOKEN`` is set, every request must send
it in the ``X-API-Key`` header (set it before exposing the API, e.g. to a Copilot Studio
connector).
"""

from __future__ import annotations

import json
import os
import secrets
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from shelfsense import __version__
from shelfsense.agent import tools
from shelfsense.agent.core import Agent
from shelfsense.agent.provider import get_provider


def check_token(x_api_key: str = Header(default="")) -> None:
    token = os.environ.get("SHELFSENSE_API_TOKEN")
    if token and not secrets.compare_digest(x_api_key.encode(), token.encode()):
        raise HTTPException(401, "missing or wrong X-API-Key")


app = FastAPI(
    title="shelfsense",
    version=__version__,
    description="Consumer insights for a health and nutrition store: product recommendations, "
    "review aspect sentiment, model monitoring and an analytics agent.",
    dependencies=[Depends(check_token)],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    tools_used: list[str]
    provider: str
    latency_ms: int


def _call(tool: str, **args: Any) -> dict[str, Any]:
    out, is_error = tools.execute(tool, {k: v for k, v in args.items() if v is not None})
    payload = json.loads(out)
    if is_error:
        msg = payload["error"]
        status = 404 if ("unknown" in msg or "no product matches" in msg) else 400
        raise HTTPException(status, msg)
    return dict(payload)


@app.get("/health")
def health() -> dict[str, Any]:
    from shelfsense.pipeline import load_results

    res = load_results()
    return {"status": "ok", "data": res["data"], "results_generated_at": res["generated_at"]}


@app.get("/queries")
def list_queries() -> dict[str, Any]:
    return _call("list_saved_queries")


@app.get("/queries/{name}")
def run_query(name: str) -> dict[str, Any]:
    return _call("run_saved_query", name=name)


@app.get("/recommend/{user_id}")
def recommend(
    user_id: str, k: int = Query(10, ge=1, le=20), apply_rules: bool = True
) -> dict[str, Any]:
    return _call("recommend_for_shopper", user_id=user_id, k=k, apply_rules=apply_rules)


@app.get("/similar/{item_id}")
def similar(item_id: str, k: int = Query(10, ge=1, le=20)) -> dict[str, Any]:
    return _call("similar_products", item=item_id, k=k)


@app.get("/insights/{item_id}")
def insights(item_id: str) -> dict[str, Any]:
    return _call("product_insights", item=item_id)


@app.get("/brands")
def brands(brand: str | None = None) -> dict[str, Any]:
    return _call("brand_scorecard", brand=brand)


@app.get("/monitoring")
def monitoring(month: str | None = Query(None, pattern=r"^\d{4}-\d{2}$")) -> dict[str, Any]:
    return _call("monitoring_report", month=month)


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return {
        "recommender": _call("recommender_metrics"),
        "aspects": _call("aspect_model_metrics"),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    # the provider comes from server settings (SHELFSENSE_PROVIDER), never from the caller
    r = Agent(get_provider()).ask(req.question)
    return AskResponse(
        answer=r.answer, tools_used=r.tools_used, provider=r.provider, latency_ms=r.latency_ms
    )
