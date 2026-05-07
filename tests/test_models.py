from fastapi.testclient import TestClient

from mlxserve.api.app import app


def test_models_endpoint_returns_list():
    client = TestClient(app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    assert data["data"][0]["object"] == "model"
