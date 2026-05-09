import jwt
import pytest
from fastapi.testclient import TestClient

from mlxserve.api.app import app
from mlxserve.config import settings


@pytest.fixture
def jwt_secret_setup(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "")
    monkeypatch.setattr(
        settings,
        "jwt_hs256_secret",
        "unit-test-jwt-hs256-secret-key-32bytes!",
    )
    monkeypatch.setattr(settings, "jwt_audience", "")


def test_jwt_bearer_accepted_for_v1(jwt_secret_setup):
    token = jwt.encode({"sub": "tester"}, settings.jwt_hs256_secret, algorithm="HS256")
    client = TestClient(app)
    resp = client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
