from fastapi.testclient import TestClient

from mlxserve.api.app import _rate_bucket, app
from mlxserve.api.deps import engine, guardian
from mlxserve.config import settings
from mlxserve.runtime.backends import StubBackend


def test_security_headers_present():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"


def test_rate_limit_blocks_requests():
    original_limit = settings.rate_limit_per_minute
    settings.rate_limit_per_minute = 1
    try:
        _rate_bucket.clear()
        client = TestClient(app)
        first = client.get("/health")
        second = client.get("/health")
        assert first.status_code == 200
        assert second.status_code == 429
    finally:
        _rate_bucket.clear()
        settings.rate_limit_per_minute = original_limit


def test_chat_still_works_with_stub():
    original = (guardian.soft_limit_gb, guardian.hard_limit_gb, engine.backend, engine.loaded_model)
    guardian.soft_limit_gb = 10_000
    guardian.hard_limit_gb = 20_000
    engine.backend = StubBackend()
    engine.loaded_model = None
    try:
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]})
        assert resp.status_code == 200
    finally:
        guardian.soft_limit_gb, guardian.hard_limit_gb, engine.backend, engine.loaded_model = original
