from mlxserve.runtime.stop_sequences import normalize_stop_sequences, truncate_at_stop_sequences


def test_truncate_at_first_stop():
    t, hit = truncate_at_stop_sequences("hello STOP world", ["STOP"])
    assert hit
    assert t == "hello "


def test_normalize_stop():
    assert normalize_stop_sequences(None) == []
    assert normalize_stop_sequences("x") == ["x"]
    assert normalize_stop_sequences(["a", "", "b"]) == ["a", "b"]
