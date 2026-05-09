from unittest.mock import patch

from fastapi.testclient import TestClient

from mlxserve.api.app import app
from mlxserve.api.deps import engine, guardian
from mlxserve.runtime.backends import StubBackend


def _relax():
    original = (guardian.soft_limit_gb, guardian.hard_limit_gb, engine.backend, engine.loaded_model)
    guardian.soft_limit_gb = 10_000
    guardian.hard_limit_gb = 20_000
    engine.backend = StubBackend()
    engine.loaded_model = None
    return original


def test_stream_includes_finish_reason_chunk():
    original = _relax()
    client = TestClient(app)
    try:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Stream finish test"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            assert response.status_code == 200
            body = "".join(chunk for chunk in response.iter_text())
    finally:
        guardian.soft_limit_gb, guardian.hard_limit_gb, engine.backend, engine.loaded_model = original

    assert "data: [DONE]" in body
    assert '"finish_reason"' in body
    assert '"usage"' in body


def test_stop_sequence_trims_completion():
    original = _relax()
    client = TestClient(app)
    try:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "stop": "MLXServe",
            },
        )
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        assert "MLXServe" not in content
    finally:
        guardian.soft_limit_gb, guardian.hard_limit_gb, engine.backend, engine.loaded_model = original


def test_v1_error_envelope_on_memory_deny():
    original = _relax()
    try:
        with patch("mlxserve.api.app.guardian.classify_detail", return_value=("red", "test_envelope")):
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "x"}], "stream": False},
            )
        assert resp.status_code == 503
        data = resp.json()
        assert "error" in data
        assert "message" in data["error"]
    finally:
        guardian.soft_limit_gb, guardian.hard_limit_gb, engine.backend, engine.loaded_model = original
