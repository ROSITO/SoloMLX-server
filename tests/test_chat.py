from fastapi.testclient import TestClient

from mlxserve.api.app import app


def test_chat_non_stream():
    client = TestClient(app)
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


def test_chat_stream_contains_done():
    client = TestClient(app)
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
