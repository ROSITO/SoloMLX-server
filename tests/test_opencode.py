import json

from fastapi.testclient import TestClient

from mlxserve.api.app import app
from mlxserve.cli import build_parser
from mlxserve.config import settings
from mlxserve.opencode import opencode_config


def test_cli_opencode_passes_args_through():
    args = build_parser().parse_args(["opencode", "run", "fix the tests", "--model", "x"])
    assert args.command == "opencode"
    assert args.args == ["run", "fix the tests", "--model", "x"]


def test_opencode_config_uses_served_model_and_context():
    cfg = opencode_config("http://127.0.0.1:8080", {"id": "org/model-4bit", "context_length": 40960})
    m = cfg["provider"]["mlxserve"]["models"]["org/model-4bit"]
    assert cfg["model"] == "mlxserve/org/model-4bit"
    assert cfg["provider"]["mlxserve"]["options"]["baseURL"] == "http://127.0.0.1:8080/v1"
    assert m["tool_call"] is True
    assert m["limit"] == {"context": 40960, "output": settings.max_output_tokens}
    json.dumps(cfg)  # must be serializable for OPENCODE_CONFIG_CONTENT


def test_models_endpoint_reports_context_length():
    data = TestClient(app).get("/v1/models").json()["data"]
    assert data[0]["context_length"] > 0


def test_opencode_config_disables_given_mcp_servers():
    cfg = opencode_config("http://h", {"id": "m"}, ["rea", "other"])
    assert cfg["mcp"] == {"rea": {"enabled": False}, "other": {"enabled": False}}
    assert opencode_config("http://h", {"id": "m"})["mcp"] == {}


def test_prompt_over_context_length_is_rejected_fast(monkeypatch):
    from mlxserve.api.deps import engine, guardian
    from mlxserve.runtime.backends import StubBackend

    monkeypatch.setattr(settings, "context_tokens", 50)
    monkeypatch.setattr(guardian, "hard_limit_gb", 20_000)
    monkeypatch.setattr(guardian, "soft_limit_gb", 10_000)
    monkeypatch.setattr(engine, "backend", StubBackend())
    monkeypatch.setattr(engine, "loaded_model", None)
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "mot " * 200}]})
    assert r.status_code == 400 and "context_length_exceeded" in r.json()["error"]["message"]
    ok = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "court"}]})
    assert ok.status_code == 200


def test_advertised_id_of_a_path_configured_model_resolves_to_the_path(monkeypatch):
    from mlxserve.api.app import _resolve_model

    path = "/Volumes/X/hub/models--mlx-community--Some-Model-4bit/snapshots/abc123"
    monkeypatch.setattr(settings, "default_model", path)
    advertised = TestClient(app).get("/v1/models").json()["data"][0]["id"]
    assert advertised == "mlx-community/Some-Model-4bit"
    assert _resolve_model(advertised) == path          # no second load under another name
    assert _resolve_model(None) == path
    assert _resolve_model("other/model") == "other/model"
