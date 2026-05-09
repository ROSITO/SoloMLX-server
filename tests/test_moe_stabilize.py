from training.moe_stabilize import run_stabilization


def test_moe_stabilize_smoke() -> None:
    report = run_stabilization(
        steps=10,
        batch_size=2,
        seq_len=8,
        dim=32,
        hidden_dim=64,
        num_experts=4,
        top_k=1,
        shared_experts=1,
        device="cpu",
    )
    assert "final_loss" in report
    assert report["final_loss"] >= 0.0
    usage = report["expert_usage_avg"]
    assert len(usage) == 4
    assert all(v >= 0.0 for v in usage.values())
