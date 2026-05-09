import json
from pathlib import Path

import pytest

from training.moe_eval_ab import _load_prompts
from training.merge_moe_eval_ab_tri import merge_tri


def test_merge_moe_eval_ab_tri() -> None:
    full_run = {
        "model_id": "m",
        "layers": [10],
        "moe_config": {"num_experts": 8},
        "adapter_path": "full.pt",
        "adapter_layers_loaded": 1,
        "dense": {"avg_loss": 5.0, "p95_latency_ms": 100.0},
        "moe_bridge": {"avg_loss": 4.9, "p95_latency_ms": 110.0},
        "delta": {"avg_loss": -0.1, "p95_latency_ms": 10.0},
        "samples_dense": [],
        "samples_moe_bridge": [],
    }
    shrunk_run = {
        "adapter_path": "shrunk.pt",
        "adapter_layers_loaded": 1,
        "moe_config": {"num_experts": 4},
        "dense": {"avg_loss": 5.01, "p95_latency_ms": 101.0},
        "moe_bridge": {"avg_loss": 4.95, "p95_latency_ms": 105.0},
        "delta": {"avg_loss": -0.06, "p95_latency_ms": 4.0},
        "samples_moe_bridge": [{"id": "1"}],
    }
    m = merge_tri(full_run, shrunk_run)
    assert m["moe_bridge_shrunk"]["avg_loss"] == 4.95
    assert m["delta_shrunk_vs_moe"]["avg_loss"] == pytest.approx(0.05)


def test_load_prompts_valid(tmp_path: Path) -> None:
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps([{"id": "a", "prompt": "hello"}]), encoding="utf-8")
    data = _load_prompts(str(p))
    assert len(data) == 1
    assert data[0]["prompt"] == "hello"
