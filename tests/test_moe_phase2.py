import json
import subprocess
import sys
import tempfile
from pathlib import Path

from mlxserve.runtime.backends import ExperimentalMoEStubBackend
from mlxserve.runtime.engine import InferenceEngine


def test_engine_selects_moe_stub_backend() -> None:
    engine = InferenceEngine(backend_mode="moe_stub")
    assert engine.backend.__class__.__name__ == "ExperimentalMoEStubBackend"


def test_moe_stub_backend_generates_routing_metadata() -> None:
    backend = ExperimentalMoEStubBackend(num_experts=4, top_k=2, num_shared_experts=1)
    text = __import__("asyncio").run(
        backend.generate(
            prompt="user: explain a python function and compare tradeoff",
            max_tokens=64,
        )
    )
    assert "routing" in text
    assert "experts=" in text


def test_bench_ab_script_outputs_report() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ab.json"
        cmd = [sys.executable, "scripts/bench_ab.py", "--out-json", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert "stub" in payload
        assert "moe_stub" in payload
        assert "delta" in payload
