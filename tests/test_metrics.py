from unittest.mock import patch

from fastapi.testclient import TestClient

from mlxserve.api.app import app
from mlxserve.api.deps import metrics


def test_metrics_endpoint_exposes_prometheus_text():
    client = TestClient(app)
    _ = client.get("/health")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "mlxserve_requests_total" in body
    assert "mlxserve_request_latency_seconds_avg" in body
    assert "mlxserve_macos_memory_pressure" in body
    assert "mlxserve_memory_chat_denied_total" in body


def test_red_memory_zone_increments_denial_counter():
    before = metrics.memory_chat_denied_total
    with patch("mlxserve.api.app.guardian.classify_detail", return_value=("red", "test_reason")):
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
    assert resp.status_code == 503
    assert metrics.memory_chat_denied_total == before + 1
    assert metrics.memory_chat_denied_by_reason.get("test_reason", 0) >= 1
