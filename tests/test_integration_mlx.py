"""Opt-in MLX integration: skipped when mlx-lm is not installed."""

import pytest

pytest.importorskip("mlx_lm")


def test_mlx_import_available():
    import mlx_lm  # noqa: F401

    assert mlx_lm is not None
