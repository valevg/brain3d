"""Тест healthcheck-эндпоинта FastAPI (используется Docker HEALTHCHECK и CI)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.fastapi_app import create_app


def test_health_check_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
