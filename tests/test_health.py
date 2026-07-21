"""Tests for the /health endpoint.

Dependency checks are patched rather than exercised for real, so these tests
run anywhere -- including in CI, where there is no Postgres or MinIO. What they
verify is the contract a load balancer depends on: 200 when everything is up,
503 when anything is down (spec section 14).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_healthy_when_all_dependencies_are_up(client, monkeypatch):
    monkeypatch.setattr("app.api.main.database_healthy", lambda: True)
    monkeypatch.setattr("app.api.main.storage_healthy", lambda: True)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["dependencies"] == {"database": True, "storage": True}


@pytest.mark.parametrize(
    ("db_up", "storage_up"),
    [(False, True), (True, False), (False, False)],
)
def test_returns_503_when_any_dependency_is_down(client, monkeypatch, db_up, storage_up):
    """A container that cannot reach its dependencies must be pulled from
    rotation rather than served traffic it cannot handle."""
    monkeypatch.setattr("app.api.main.database_healthy", lambda: db_up)
    monkeypatch.setattr("app.api.main.storage_healthy", lambda: storage_up)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_health_names_the_failing_dependency(client, monkeypatch):
    """The breakdown is the point -- it says *which* service is down."""
    monkeypatch.setattr("app.api.main.database_healthy", lambda: True)
    monkeypatch.setattr("app.api.main.storage_healthy", lambda: False)

    body = client.get("/health").json()

    assert body["dependencies"]["database"] is True
    assert body["dependencies"]["storage"] is False


def test_root_endpoint_points_to_docs(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["health"] == "/health"
