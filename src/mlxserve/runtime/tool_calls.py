"""OpenAI tool calling on top of mlx-lm's per-model tool parsers (Qwen JSON, Qwen3-Coder XML, Gemma 4, ...)."""
import json
import logging
import uuid
from typing import Any, Callable

# (tool_call_start, tool_call_end, parser) as exposed by mlx-lm's TokenizerWrapper.
ToolCallFormat = tuple[str, str, Callable[[str, Any], Any]]

log = logging.getLogger(__name__)


def template_messages(messages: list) -> list[dict[str, Any]]:
    """ChatMessage list -> dicts for apply_chat_template (templates want arguments as a dict)."""
    out = []
    for m in messages:
        d = m.model_dump(exclude_none=True)
        d["content"] = d.get("content") or ""
        for tc in d.get("tool_calls", []):
            try:
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                pass  # leave the raw string; most templates print it as-is
        out.append(d)
    return out


def partial_marker_len(text: str, marker: str) -> int:
    """Length of the longest suffix of text that is a strict prefix of marker (held back while streaming)."""
    for n in range(min(len(marker) - 1, len(text)), 0, -1):
        if text.endswith(marker[:n]):
            return n
    return 0


def extract_tool_calls(text: str, fmt: ToolCallFormat, tools: list[dict] | None) -> tuple[str, list[dict]]:
    """Split generated text into (content, OpenAI tool_calls). Unparseable (e.g. truncated) calls are dropped."""
    start, end, parser = fmt
    content, calls, rest = [], [], text
    while (i := rest.find(start)) != -1:
        content.append(rest[:i])
        j = rest.find(end, i + len(start))
        body = rest[i + len(start) : j if j != -1 else len(rest)]
        rest = "" if j == -1 else rest[j + len(end) :]
        try:
            parsed = parser(body.strip(), tools)
        except Exception as exc:  # model output: never let a bad call 500 the request
            log.warning("Dropping unparseable tool call (%s): %.200s", type(exc).__name__, body)
            continue
        for tc in parsed if isinstance(parsed, list) else [parsed]:
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc.get("arguments") or {}, ensure_ascii=False),
                    },
                }
            )
    content.append(rest)
    return "".join(content), calls
