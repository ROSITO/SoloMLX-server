from fastapi.testclient import TestClient

from mlxserve.api.app import app
from mlxserve.api.deps import engine, guardian
from mlxserve.runtime.backends import StubBackend


def _relax_memory_limits():
    original = (guardian.soft_limit_gb, guardian.hard_limit_gb, engine.backend, engine.loaded_model)
    guardian.soft_limit_gb = 10_000
    guardian.hard_limit_gb = 20_000
    engine.backend = StubBackend()
    engine.loaded_model = None
    return original


def test_chat_non_stream():
    original = _relax_memory_limits()
    client = TestClient(app)
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Bonjour"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["object"] == "chat.completion"
        assert payload["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(payload["choices"][0]["message"]["content"], str)
        usage = payload["usage"]
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        assert usage["prompt_tokens"] >= 0
        assert usage["completion_tokens"] >= 0
    finally:
        guardian.soft_limit_gb, guardian.hard_limit_gb, engine.backend, engine.loaded_model = original


def test_chat_stream_contains_done():
    original = _relax_memory_limits()
    client = TestClient(app)
    try:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Stream test"}],
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            body = "".join(chunk for chunk in response.iter_text())
        assert "data: [DONE]" in body
    finally:
        guardian.soft_limit_gb, guardian.hard_limit_gb, engine.backend, engine.loaded_model = original
