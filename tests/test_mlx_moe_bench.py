"""mlx_moe_bench helpers; full bench skipped without mlx-lm + network."""

from __future__ import annotations

import pytest

from training.mlx_moe_bench import _model_summary


class _Args:
    model_type = "mixtral"
    num_local_experts = 8
    num_experts_per_tok = 2


class _Model:
    args = _Args()


def test_model_summary_moe_fields() -> None:
    s = _model_summary(_Model())
    assert s["num_local_experts"] == 8
    assert s["num_experts_per_tok"] == 2
    assert s["expert_activation_ratio"] == 0.25


def test_mlx_moe_bench_module_imports_when_mlx_lm_present() -> None:
    pytest.importorskip("mlx_lm")
    import training.mlx_moe_bench as m

    assert callable(m.main)
