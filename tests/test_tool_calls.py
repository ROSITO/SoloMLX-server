import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mlxserve.api.app import app
from mlxserve.api.deps import engine, guardian
from mlxserve.models.schemas import ChatMessage
from mlxserve.runtime.backends import StubBackend
from mlxserve.runtime.tool_calls import extract_tool_calls, partial_marker_len, template_messages

FMT = ("<tool_call>", "</tool_call>", lambda body, tools: json.loads(body))
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
CALL = '<tool_call>{"name": "get_weather", "arguments": {"city": "Lyon"}}</tool_call>'


class ToolBackend(StubBackend):
    def __init__(self, pieces):
        super().__init__()
        self.pieces = pieces
        self.seen_tools = "unset"

    async def build_chat_prompt(self, messages, tools=None):
        self.seen_tools = tools
        return "prompt"

    async def generate(self, prompt, max_tokens, temperature=0.2, top_p=0.95, stop_sequences=None):
        return "".join(self.pieces)

    async def stream(self, prompt, max_tokens, temperature=0.2, top_p=0.95, stop_sequences=None):
        for p in self.pieces:
            yield p

    def tool_call_format(self):
        return FMT


@pytest.fixture
def backend_with(monkeypatch):
    monkeypatch.setattr(guardian, "soft_limit_gb", 10_000)
    monkeypatch.setattr(guardian, "hard_limit_gb", 20_000)
    monkeypatch.setattr(engine, "loaded_model", None)

    def make(pieces):
        b = ToolBackend(pieces)
        monkeypatch.setattr(engine, "backend", b)
        return b

    return make


def _post(body, stream=False):
    body = {"messages": [{"role": "user", "content": "Météo à Lyon ?"}], "stream": stream, **body}
    client = TestClient(app)
    if not stream:
        return client.post("/v1/chat/completions", json=body).json()
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        return [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: {")]


def test_extract_tool_calls_keeps_content_drops_truncated():
    text = "Je regarde." + CALL + CALL.replace("Lyon", "Paris") + '<tool_call>{"name": "get_wea'
    content, calls = extract_tool_calls(text, FMT, TOOLS)
    assert content == "Je regarde."
    assert [json.loads(c["function"]["arguments"])["city"] for c in calls] == ["Lyon", "Paris"]
    assert calls[0]["type"] == "function" and calls[0]["id"].startswith("call_")


def test_partial_marker_len():
    assert partial_marker_len("abc<tool_c", "<tool_call>") == 7
    assert partial_marker_len("abc", "<tool_call>") == 0
    assert partial_marker_len("a <", "<tool_call>") == 1


def test_messages_validation_and_template_conversion():
    with pytest.raises(ValidationError):
        ChatMessage(role="tool", tool_call_id="x")
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="")
    parts = ChatMessage(role="user", content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert parts.content == "ab"
    call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Lyon"}'}}
    msgs = template_messages([ChatMessage(role="assistant", tool_calls=[call]), ChatMessage(role="tool", content="21°C", tool_call_id="c1")])
    assert msgs[0]["content"] == "" and msgs[0]["tool_calls"][0]["function"]["arguments"] == {"city": "Lyon"}
    assert msgs[1] == {"role": "tool", "content": "21°C", "tool_call_id": "c1"}


def test_non_stream_returns_tool_calls(backend_with):
    b = backend_with([CALL])
    out = _post({"tools": TOOLS})
    choice = out["choices"][0]
    assert b.seen_tools == TOOLS
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"] == {"name": "get_weather", "arguments": '{"city": "Lyon"}'}


def test_tool_choice_none_drops_tools(backend_with):
    b = backend_with(["Il fait beau."])
    out = _post({"tools": TOOLS, "tool_choice": "none"})
    assert b.seen_tools is None
    assert out["choices"][0]["message"]["content"] == "Il fait beau."
    assert out["choices"][0]["finish_reason"] == "stop"


def test_stream_splits_content_and_tool_calls(backend_with):
    backend_with(["Je ", "regarde.", "<tool", '_call>{"name": "get_weather", ', '"arguments": {"city": "Lyon"}}</tool_call>'])
    chunks = _post({"tools": TOOLS}, stream=True)
    deltas = [c["choices"][0]["delta"] for c in chunks]
    assert "".join(d.get("content", "") for d in deltas) == "Je regarde."
    calls = [tc for d in deltas for tc in d.get("tool_calls", [])]
    assert len(calls) == 1 and calls[0]["index"] == 0 and calls[0]["function"]["name"] == "get_weather"
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_stream_held_back_tail_is_flushed(backend_with):
    backend_with(["a ", "<"])  # looks like the start of a marker, but the stream ends
    chunks = _post({"tools": TOOLS}, stream=True)
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "a <"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_max_tokens_clamped_not_rejected():
    from mlxserve.config import settings
    from mlxserve.models.schemas import ChatCompletionRequest

    req = ChatCompletionRequest(messages=[{"role": "user", "content": "x"}], max_tokens=32_000)
    assert req.max_tokens == settings.max_output_tokens
