from mlxserve.runtime.stop_sequences import (
    merge_chat_boundary_stops,
    normalize_stop_sequences,
    truncate_at_stop_sequences,
)


def test_truncate_at_first_stop():
    t, hit = truncate_at_stop_sequences("hello STOP world", ["STOP"])
    assert hit
    assert t == "hello "


def test_normalize_stop():
    assert normalize_stop_sequences(None) == []
    assert normalize_stop_sequences("x") == ["x"]
    assert normalize_stop_sequences(["a", "", "b"]) == ["a", "b"]


def test_merge_chat_boundary_stops_qwen_adds_im_start():
    m = merge_chat_boundary_stops("mlx-community/Qwen2.5-7B-Instruct-4bit", ["STOP"])
    assert m[0] == "STOP"
    assert "<|im_start|>" in m


def test_merge_chat_boundary_stops_dedupes():
    m = merge_chat_boundary_stops("Qwen2.5-3B", ["<|im_start|>", "x"])
    assert m.count("<|im_start|>") == 1


def test_truncate_with_merged_qwen_stops():
    raw = "Bonjour.<|im_start|>user\ngarbage"
    stops = merge_chat_boundary_stops("qwen2.5-7b", None)
    t, hit = truncate_at_stop_sequences(raw, stops)
    assert t == "Bonjour."
    assert hit is True
