from fastapi.testclient import TestClient

from mlxserve.api.app import app


def test_recommended_models_endpoint():
    client = TestClient(app)
    response = client.get("/v1/models/recommended")
    assert response.status_code == 200
    payload = response.json()
    assert payload["machine_ram_gb"] > 0
    assert isinstance(payload["data"], list)
    assert len(payload["data"]) >= 1
    assert "id" in payload["data"][0]
