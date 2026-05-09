"""OpenAI-style stop sequences: truncate completion without including the matched stop."""

from __future__ import annotations


def normalize_stop_sequences(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    return [s for s in stop if isinstance(s, str) and s]


def truncate_at_stop_sequences(text: str, stops: list[str] | None) -> tuple[str, bool]:
    if not stops:
        return text, False
    best = len(text)
    for s in stops:
        if not s:
            continue
        i = text.find(s)
        if i != -1 and i < best:
            best = i
    if best < len(text):
        return text[:best], True
    return text, False


def merge_chat_boundary_stops(model_id: str | None, stops: list[str] | None) -> list[str]:
    """Append model-specific substring stops so degenerate chat-template spill stops early.

    Qwen2 / Qwen2.5 (ChatML) often repeats ``<|im_start|>`` when it should have stopped; that
    string is never valid inside an assistant reply, so truncating there fixes garbled UI.
    """
    out: list[str] = []
    seen: set[str] = set()
    for s in stops or []:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    mid = (model_id or "").lower()
    extras: list[str] = []
    if "qwen" in mid:
        extras.append("<|im_start|>")
    if "llama-3" in mid or "llama_3" in mid:
        extras.append("<|start_header_id|>")
    for s in extras:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
