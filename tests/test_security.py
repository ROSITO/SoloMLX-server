from fastapi.testclient import TestClient

from mlxserve.api.app import app
from mlxserve.config import settings


def test_api_key_required_when_enabled():
    original = settings.api_key
    settings.api_key = "secret-key"
    try:
        client = TestClient(app)
        denied = client.get("/v1/models")
        assert denied.status_code == 401

        allowed = client.get("/v1/models", headers={"Authorization": "Bearer secret-key"})
        assert allowed.status_code == 200
    finally:
        settings.api_key = original
