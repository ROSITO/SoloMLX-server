import pytest

from mlxserve.runtime.reasoning import EmptyReasoningFilter, strip_empty_reasoning

# Raw Gemma 4 outputs captured from a real OpenCode session.
TRAILING = "I will now run the tests to verify.\n\n<|channel>thought\n<channel|>"
STRAY_LEADING = "thought\n<|channel>thought\n<channel|>All 14 tests passed successfully."


@pytest.mark.parametrize(
    "raw,clean",
    [
        (TRAILING, "I will now run the tests to verify.\n\n"),
        (STRAY_LEADING, "All 14 tests passed successfully."),
        ("<think>\n\n</think>\n\n391.", "391."),
        ("a <|channel>thought\n<channel|>b", "a b"),
        ("Done. <|channel>thought\n<channel|>\nNext", "Done. Next"),
        ("<think>real reasoning</think>ok", "<think>real reasoning</think>ok"),  # non-empty: kept
        ("I thought about it", "I thought about it"),
        ("if a < b and c > d:", "if a < b and c > d:"),
    ],
)
def test_strip_and_stream_agree(raw, clean):
    assert strip_empty_reasoning(raw) == clean
    for size in (1, 2, 3, 7, 1000):  # any chunking gives the same text
        f = EmptyReasoningFilter()
        out = "".join(f.feed(raw[i : i + size]) for i in range(0, len(raw), size)) + f.flush()
        assert out == clean, size


def test_stream_holds_back_only_while_ambiguous():
    f = EmptyReasoningFilter()
    assert f.feed("Done. <|chan") == "Done. "
    assert f.feed("nel>thought\n") == ""
    assert f.feed("<channel|>Next") == "Nex"  # a trailing "t" may start "thought": held one chunk
    assert f.flush() == "t"
