"""Remove empty reasoning blocks that models emit with thinking off, anywhere in a reply.

Seen in practice: Qwen3 `<think>\\n\\n</think>`, Gemma 4 `<|channel>thought\\n<channel|>` at the start,
at the end before a tool call, and after a stray `thought` word mid-reply.
"""
import re

_PAIRS = (("<think>", "</think>"), ("<|channel>thought", "<channel|>"))
_STRAY = "thought"  # Gemma 4 sometimes writes it right before an empty block
EMPTY_REASONING_RE = re.compile(
    r"(?:" + _STRAY + r"\s*)?(?:" + "|".join(re.escape(o) + r"\s*" + re.escape(c) for o, c in _PAIRS) + r")\s*"
)


def strip_empty_reasoning(text: str) -> str:
    return EMPTY_REASONING_RE.sub("", text)


def _may_start_block(s: str) -> bool:
    """True if s could still grow into an empty block (so it must not be emitted yet)."""
    if s.startswith(_STRAY):
        rest = s[len(_STRAY) :].lstrip()
        if not rest:
            return True
        s = rest
    elif _STRAY.startswith(s):
        return True
    for opener, closer in _PAIRS:
        if opener.startswith(s):
            return True
        if s.startswith(opener) and closer.startswith(s[len(opener) :].lstrip()):
            return True
    return False


class EmptyReasoningFilter:
    """Streaming version: drops complete empty blocks, holds back text that may start one."""

    def __init__(self) -> None:
        self._buf = ""
        self._eat_ws = False  # a block was just dropped: the whitespace after it goes too

    def feed(self, text: str) -> str:
        raw = self._buf + text
        buf = strip_empty_reasoning(raw)
        if self._eat_ws:
            buf = buf.lstrip()
            self._eat_ws = not buf
        if any(m.end() == len(raw) for m in EMPTY_REASONING_RE.finditer(raw)):
            self._eat_ws = True
        hold = next((i for i in range(len(buf)) if buf[i] in "<t" and _may_start_block(buf[i:])), len(buf))
        self._buf = buf[hold:]
        return buf[:hold]

    def flush(self) -> str:
        out, self._buf = strip_empty_reasoning(self._buf), ""
        return out
