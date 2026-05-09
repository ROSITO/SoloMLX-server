from tools.estimate_active_params_proxy import estimate_proxy


def test_estimate_active_params_proxy() -> None:
    report = {
        "total_input_keys": 100,
        "converted_keys": 10,
        "top_k": 1,
        "shared_experts": 1,
    }
    out = estimate_proxy(report)
    assert out["dense_active_proxy"] == 100.0
    assert out["moe_active_proxy"] == 110.0
    assert out["moe_vs_dense_ratio"] > 1.0
