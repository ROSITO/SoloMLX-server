import sys
import types

from mlxserve.runtime import autotune
from mlxserve.runtime.autotune import TuneResult


def test_autotune_prefers_fp16_kv(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(load=lambda _: (None, None)))

    def fake_run(model, tok, prompt, max_tokens, *, prefill_step_size, kv_bits, kv_group_size, quantized_kv_start):
        tps = 100.0 if kv_bits else 90.0  # quantized is faster, must still lose
        return TuneResult(prefill_step_size, kv_bits, kv_group_size, quantized_kv_start, tps + prefill_step_size / 1000, 0, 1, 1, 1)

    monkeypatch.setattr(autotune, "_run_once", fake_run)
    out = autotune.run_autotune("m", "p")
    assert out["best"]["kv_bits"] is None
    assert out["best"]["prefill_step_size"] == 2048
    assert out["recommended_env"]["MLXSERVE_KV_BITS"] == "none"
    assert {r["prefill_step_size"] for r in out["all_runs"]} >= {512}
