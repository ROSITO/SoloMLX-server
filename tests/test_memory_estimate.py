from mlxserve.memory.estimate import (
    admission_memory_gb,
    estimate_kv_cache_gb,
    estimated_request_gb,
)


def test_estimate_kv_cache_gb_scales_with_seq_len() -> None:
    a = estimate_kv_cache_gb(512, num_layers=32, num_kv_heads=8, head_dim=128, bytes_per_element=2.0)
    b = estimate_kv_cache_gb(1024, num_layers=32, num_kv_heads=8, head_dim=128, bytes_per_element=2.0)
    assert b > a > 0


def test_admission_memory_gb_adds_capped_kv() -> None:
    base_only = estimated_request_gb(1000, 500, tokens_per_gb=5000.0, cap_gb=2.0)
    with_kv = admission_memory_gb(
        1000,
        500,
        tokens_per_gb=5000.0,
        cap_gb=2.0,
        kv_enabled=True,
        kv_max_gb=0.5,
        kv_layers=32,
        kv_heads=8,
        kv_head_dim=128,
        kv_bytes_per_element=2.0,
    )
    assert with_kv >= base_only
    assert with_kv <= base_only + 0.5001


def test_admission_memory_gb_kv_disabled_equals_base() -> None:
    a = admission_memory_gb(
        2000,
        256,
        tokens_per_gb=4000.0,
        cap_gb=1.5,
        kv_enabled=False,
        kv_max_gb=9.0,
        kv_layers=40,
        kv_heads=8,
        kv_head_dim=128,
        kv_bytes_per_element=2.0,
    )
    b = estimated_request_gb(2000, 256, 4000.0, 1.5)
    assert abs(a - b) < 1e-9
