"""Health endpoints, with the probe list overridden by fakes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from visionforge import __version__
from visionforge.api.dependencies import get_health_service
from visionforge.api.main import create_app
from visionforge.application.health_service import HealthService
from visionforge.domain.health import ComponentHealth, ComponentStatus


class StubProbe:
    def __init__(self, name: str, status: ComponentStatus, *, required: bool = True) -> None:
        self.name = name
        self.required_for_readiness = required
        self._status = status

    async def check(self) -> ComponentHealth:
        return ComponentHealth(name=self.name, status=self._status, latency_ms=0.5)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_health_is_shallow_and_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_liveness(client: TestClient) -> None:
    assert client.get("/health/live").json() == {"status": "ok"}


def test_version_reports_package_version(client: TestClient) -> None:
    body = client.get("/version").json()
    assert body["version"] == __version__
    assert body["name"] == "visionforge-api"


def test_request_id_header_is_echoed(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "trace-me"})
    assert response.headers["X-Request-ID"] == "trace-me"


def test_readiness_ok_when_dependencies_healthy(client: TestClient) -> None:
    app = client.app
    app.dependency_overrides[get_health_service] = lambda: HealthService(  # type: ignore[attr-defined]
        [StubProbe("postgres", ComponentStatus.OK), StubProbe("redis", ComponentStatus.OK)]
    )

    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_returns_503_when_required_dependency_is_down(client: TestClient) -> None:
    app = client.app
    app.dependency_overrides[get_health_service] = lambda: HealthService(  # type: ignore[attr-defined]
        [StubProbe("postgres", ComponentStatus.FAILED)]
    )

    response = client.get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["components"][0]["name"] == "postgres"
