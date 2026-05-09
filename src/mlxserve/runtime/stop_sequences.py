"""OpenAI-style stop sequences: truncate completion without including the matched stop."""


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
