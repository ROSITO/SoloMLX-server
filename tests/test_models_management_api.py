from fastapi.testclient import TestClient

from mlxserve.api.app import app
from mlxserve.api.deps import model_manager


def test_local_models_endpoint(monkeypatch):
    class StubModel:
        id = "qwen25-3b"
        source = "mlx-community/Qwen2.5-3B-Instruct-4bit"
        local_path = "/tmp/model"
        pulled_at = "2026-01-01T00:00:00+00:00"
        size_bytes = 1024
        quantization = "4bit"

    monkeypatch.setattr(model_manager, "list_local", lambda: [StubModel()])
    client = TestClient(app)
    resp = client.get("/v1/models/local")
    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "qwen25-3b"


def test_pull_and_delete_model_endpoints(monkeypatch):
    class StubModel:
        id = "mistral-7b"
        source = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
        local_path = "/tmp/mistral"
        pulled_at = "2026-01-01T00:00:00+00:00"
        size_bytes = 2048
        quantization = "4bit"

    monkeypatch.setattr(model_manager, "pull", lambda model: StubModel())
    monkeypatch.setattr(model_manager, "remove", lambda model: True)

    client = TestClient(app)
    resp = client.post("/v1/models/pull", json={"model": "mlx-community/Mistral-7B-Instruct-v0.3-4bit"})
    assert resp.status_code == 200
    assert resp.json()["id"] == "mistral-7b"

    resp = client.delete("/v1/models/mistral-7b")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
