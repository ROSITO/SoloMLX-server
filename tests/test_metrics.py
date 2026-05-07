from fastapi.testclient import TestClient

from mlxserve.api.app import app


def test_metrics_endpoint_exposes_prometheus_text():
    client = TestClient(app)
    _ = client.get("/health")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "mlxserve_requests_total" in body
    assert "mlxserve_request_latency_seconds_avg" in body
