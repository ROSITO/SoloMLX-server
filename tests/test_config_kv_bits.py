import os

import pytest

from mlxserve.config import Settings


def test_kv_bits_default_none(monkeypatch):
    monkeypatch.delenv("MLXSERVE_KV_BITS", raising=False)
    s = Settings()
    assert s.kv_bits is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", None),
        ("none", None),
        ("OFF", None),
        ("0", None),
        ("-1", None),
        ("4", 4),
    ],
)
def test_kv_bits_env_coercion(monkeypatch, raw, expected):
    if raw == "":
        monkeypatch.delenv("MLXSERVE_KV_BITS", raising=False)
    else:
        monkeypatch.setenv("MLXSERVE_KV_BITS", raw)
    s = Settings()
    assert s.kv_bits == expected
