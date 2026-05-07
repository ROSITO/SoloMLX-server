from fastapi.testclient import TestClient

from mlxserve.api.app import app, display_hf_model_id


def test_display_hf_model_id_from_cache_path():
    path = (
        "/Users/x/.cache/huggingface/hub/models--mlx-community--Mistral-7B-Instruct-v0.3-4bit"
        "/snapshots/a4b8f870474b0eb527f466a03fbc187830d271f5"
    )
    assert display_hf_model_id(path) == "mlx-community/Mistral-7B-Instruct-v0.3-4bit"


def test_display_hf_model_id_passes_through_repo_id():
    assert display_hf_model_id("mlx-community/Qwen2.5-0.5B-Instruct-4bit") == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def test_models_endpoint_returns_list():
    client = TestClient(app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    assert data["data"][0]["object"] == "model"
