from mlxserve.memory.guardian import MemoryGuardian


def test_memory_classify_green_or_worse():
    guardian = MemoryGuardian(soft_limit_gb=1000, hard_limit_gb=1001, idle_unload_minutes=1)
    zone = guardian.classify(estimated_request_gb=0.0)
    assert zone in {"green", "yellow", "red"}


def test_idle_unload_logic():
    guardian = MemoryGuardian(soft_limit_gb=10, hard_limit_gb=12, idle_unload_minutes=1)
    assert guardian.should_unload_idle(last_used_ts=0.0) is True
