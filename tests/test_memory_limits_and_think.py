import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mlxserve.api.app import _empty_think_skip, _sanitize_completion_text, app
from mlxserve.api.deps import engine
from mlxserve.config import Settings
from tests.test_chat_advanced import _relax


@pytest.mark.parametrize("total_gib,soft,hard", [(16, 14.0, 15.0), (36, 31.5, 33.8)])
def test_memory_limits_scale_with_ram(monkeypatch, total_gib, soft, hard):
    monkeypatch.delenv("MLXSERVE_MAX_MEMORY_GB", raising=False)
    monkeypatch.delenv("MLXSERVE_HARD_MEMORY_GB", raising=False)
    monkeypatch.setattr("mlxserve.config.psutil.virtual_memory", lambda: SimpleNamespace(total=total_gib * 1024**3))
    s = Settings()
    assert (s.max_memory_gb, s.hard_memory_gb) == (soft, hard)


def test_memory_limits_env_override(monkeypatch):
    monkeypatch.setenv("MLXSERVE_MAX_MEMORY_GB", "20")
    monkeypatch.setenv("MLXSERVE_HARD_MEMORY_GB", "22")
    s = Settings()
    assert (s.max_memory_gb, s.hard_memory_gb) == (20.0, 22.0)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<th", None),
        ("<think>\n\n</thi", None),
        ("<think>\n\n</think>\n\n", None),
        ("<think>\n\n</think>\n\nHi", 19),
        ("Hello", 0),
        ("<b>bold</b>", 0),
        ("<think>real reasoning", 0),
    ],
)
def test_empty_think_skip(text, expected):
    assert _empty_think_skip(text) == expected


def test_sanitize_strips_empty_think_only():
    assert _sanitize_completion_text("<think>\n\n</think>\n\n391.") == "391."
    assert _sanitize_completion_text("<think>x</think>391.") == "<think>x</think>391."


def test_stream_hides_empty_think():
    async def fake_stream(**_):
        for piece in ["<think>", "\n\n", "</think>", "\n\n", "17*23", " = 391."]:
            yield piece

    original = _relax()
    try:
        with patch.object(engine, "stream_text", fake_stream):
            with TestClient(app).stream(
                "POST", "/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}], "stream": True}
            ) as r:
                body = "".join(r.iter_text())
    finally:
        from mlxserve.api.deps import guardian

        guardian.soft_limit_gb, guardian.hard_limit_gb, engine.backend, engine.loaded_model = original
    deltas = [
        json.loads(line[6:])["choices"][0]["delta"].get("content", "")
        for line in body.splitlines()
        if line.startswith("data: {")
    ]
    assert "".join(deltas) == "17*23 = 391."
