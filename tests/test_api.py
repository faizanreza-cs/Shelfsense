from fastapi.testclient import TestClient

from shelfsense.api import app

client = TestClient(app)


def test_endpoints():
    assert client.get("/health").json()["data"]["source"] == "synthetic"
    assert "bought_together" in [q["name"] for q in client.get("/queries").json()["rows"]]
    assert client.get("/queries/nope").status_code == 404
    recs = client.get("/recommend/C000123", params={"k": 5}).json()
    assert recs["row_count"] == 5
    assert "best sellers" in client.get("/recommend/C999999").json()["note"]
    assert client.get("/recommend/C000123", params={"k": 50}).status_code == 422
    assert client.get("/similar/P0015").json()["row_count"] == 10
    insights = client.get("/insights/P0015").json()
    assert insights["product"]["item_id"] == "P0015" and insights["alerts"]
    assert client.get("/insights/P9999").status_code == 404
    assert client.get("/brands", params={"brand": "Solstice"}).json()["rows"]
    assert client.get("/monitoring", params={"month": "2026-05"}).json()["month"] == "2026-05"
    assert client.get("/monitoring", params={"month": "May"}).status_code == 422
    assert set(client.get("/metrics").json()) == {"recommender", "aspects"}
    r = client.post("/ask", json={"question": "What do reviewers say about P0015?"})
    assert r.json()["tools_used"] == ["product_insights"]


def test_api_key_is_enforced_when_set(monkeypatch):
    monkeypatch.setenv("SHELFSENSE_API_TOKEN", "s3cret")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/metrics", headers={"X-API-Key": "s3cret"}).status_code == 200
    odd = {"X-API-Key": "café".encode("latin-1")}  # non-ASCII: a clean 401, not a 500
    assert client.get("/metrics", headers=odd).status_code == 401
