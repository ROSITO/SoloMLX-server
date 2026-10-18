from mlxserve.memory.estimate import admission_memory_gb
from mlxserve.runtime.backends import MLXLMBackend

GENERIC = dict(tokens_per_gb=4500, cap_gb=2.0, kv_enabled=True, kv_max_gb=4.0, kv_layers=40, kv_heads=8,
               kv_head_dim=128, kv_bytes_per_element=2.0)


def test_kv_estimate_from_profile_caps_sliding_layers():
    b = MLXLMBackend(prefill_step_size=1024)
    assert b.kv_gb_estimate(1000) is None  # not loaded: unknown
    # Gemma 4 26B-A4B layout: 5 global layers (4 KB/token), 25 sliding layers (8 KB/token, window 1024)
    b._kv_profile = (5 * 4096, [(8192, 1024)] * 25, 0)
    assert abs(b.kv_gb_estimate(16384) - 0.755) < 0.01  # measured on the real model: 0.755 GB
    assert abs(b.kv_gb_estimate(1000) - (5 * 4096 + 25 * 8192) * 1000 / 1e9) < 1e-9


def test_admission_uses_measured_kv_instead_of_generic_bound():
    generic = admission_memory_gb(14_000, 8192, **GENERIC)
    measured = admission_memory_gb(14_000, 8192, **GENERIC, kv_gb=0.9)
    assert round(generic, 2) == 5.39  # 2 GB base + generic KV bound (3.39 GB for 22k tokens)
    assert round(measured, 2) == 2.9
