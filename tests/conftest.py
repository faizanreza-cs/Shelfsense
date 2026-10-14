"""Build the synthetic store and every model once per test session."""

from __future__ import annotations

import os

import pytest

from shelfsense import pipeline, warehouse

ENV = ("SHELFSENSE_DB", "SHELFSENSE_PROVIDER", "SHELFSENSE_BACKEND", "SHELFSENSE_API_TOKEN")


@pytest.fixture(scope="session", autouse=True)
def built(tmp_path_factory: pytest.TempPathFactory) -> dict:
    db = tmp_path_factory.mktemp("data") / "shelfsense.duckdb"
    old = {k: os.environ.get(k) for k in ENV}
    os.environ.update(
        {"SHELFSENSE_DB": str(db), "SHELFSENSE_PROVIDER": "mock", "SHELFSENSE_BACKEND": "duckdb"}
    )
    os.environ.pop("SHELFSENSE_API_TOKEN", None)
    results = pipeline.build()
    yield results
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    warehouse.reset_cache()


@pytest.fixture
def wh() -> warehouse.Warehouse:
    return warehouse.get_warehouse()
