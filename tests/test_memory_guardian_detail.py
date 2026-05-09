from unittest.mock import patch

from mlxserve.memory.guardian import MemoryGuardian, MemorySnapshot


def test_classify_detail_red_hard_limit() -> None:
    g = MemoryGuardian(soft_limit_gb=1.0, hard_limit_gb=2.0, idle_unload_minutes=1)
    snap = MemorySnapshot(used_gb=0.5, swap_used_gb=0.1, total_gb=16.0, pressure="normal")
    with patch.object(g, "snapshot", return_value=snap):
        zone, reason = g.classify_detail(estimated_request_gb=5.0)
    assert zone == "red"
    assert reason == "projected_over_hard_limit_gb"


def test_classify_matches_classify_detail_zone() -> None:
    g = MemoryGuardian(soft_limit_gb=1000.0, hard_limit_gb=1001.0, idle_unload_minutes=1)
    z1 = g.classify(0.0)
    z2, _ = g.classify_detail(0.0)
    assert z1 == z2
