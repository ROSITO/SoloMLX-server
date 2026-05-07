from fastapi.testclient import TestClient

from mlxserve.api.app import app


def test_root_serves_chat_ui():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "MLXServe Chat" in resp.text
    assert "POST /v1/chat/completions" in resp.text
    assert "code-block-wrap" in resp.text
    assert "renderAssistantHtml" in resp.text
    assert "normalizeFenceMarkdown" in resp.text
    assert "catalogRecommended" in resp.text
    assert "refreshListedModel" in resp.text
    assert "humanizeModelId" in resp.text
