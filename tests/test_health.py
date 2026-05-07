from fastapi.testclient import TestClient

from mlxserve.api.app import app


def test_health_returns_ok():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["memory_zone"] in {"green", "yellow", "red"}
