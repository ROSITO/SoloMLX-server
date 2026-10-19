from collections import Counter

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.cache import KVCache, RotatingKVCache  # noqa: E402

from mlxserve.runtime.mtp import _rewind, accept_sampled, assistant_repo  # noqa: E402


def test_assistant_repo_naming():
    assert assistant_repo("mlx-community/gemma-4-26B-A4B-it-qat-4bit") == "mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit"
    assert assistant_repo("mlx-community/gemma-4-31B-it-bf16") == "mlx-community/gemma-4-31B-it-assistant-bf16"
    assert assistant_repo("mlx-community/Qwen3-14B-4bit") is None


def _kv(start, n):
    a = mx.arange(start, start + n, dtype=mx.float32).reshape(1, 1, n, 1)
    return a, a


@pytest.mark.parametrize("make", [KVCache, lambda: RotatingKVCache(max_size=8)])
def test_rewind_matches_a_cache_that_never_saw_the_rejected_tokens(make):
    # Verify step appends [y, d1, d2, d3]; d2, d3 rejected -> rewind 2; the next verify appends 2 tokens.
    a, b = make(), make()
    for c in (a, b):
        c.update_and_fetch(*_kv(0, 20))  # prompt, past the sliding window
    a.update_and_fetch(*_kv(100, 4))
    _rewind([a], 2)
    b.update_and_fetch(*_kv(100, 2))
    a_k, _ = a.update_and_fetch(*_kv(200, 2))
    b_k, _ = b.update_and_fetch(*_kv(200, 2))
    assert a.offset == b.offset == 24
    assert a_k.reshape(-1).tolist() == b_k.reshape(-1).tolist()


def test_speculative_sampling_keeps_the_target_distribution():
    mx.random.seed(0)
    p = mx.array([[0.5, 0.3, 0.15, 0.05], [0.25, 0.25, 0.25, 0.25]])  # target at the draft position, then bonus
    q = mx.array([0.1, 0.6, 0.2, 0.1])  # drafter, deliberately different
    counts = Counter()
    trials = 4000
    for _ in range(trials):
        d = mx.random.categorical(mx.log(q)).item()
        m, nxt = accept_sampled(p, [q], [d], eos=set())
        counts[d if m == 1 else nxt] += 1
    for tok, want in enumerate(p[0].tolist()):
        assert abs(counts[tok] / trials - want) < 0.03, (tok, counts[tok] / trials, want)


def test_eos_draft_is_never_accepted_blindly():
    p = mx.array([[0.0, 1.0], [0.5, 0.5]])
    m, nxt = accept_sampled(p, [mx.array([1.0, 0.0])], [0], eos={0})
    assert m == 0 and nxt == 1  # resampled from the target, which never ends here


def test_backend_runs_without_mtp_when_drafter_is_unavailable(monkeypatch):
    import asyncio
    import types

    import mlxserve.runtime.mtp as mtp_mod
    from mlxserve.runtime.backends import MLXLMBackend

    monkeypatch.setitem(__import__("sys").modules, "mlx_lm", types.SimpleNamespace(load=lambda m: (object(), object())))

    def offline(*a, **k):
        raise OSError("offline and not cached")

    monkeypatch.setattr(mtp_mod, "load_mtp", offline)
    b = MLXLMBackend(mtp=True)
    monkeypatch.setattr(b, "_measure_kv_profile", lambda: None)
    asyncio.run(b.load("mlx-community/gemma-4-26B-A4B-it-qat-4bit"))
    assert b.model is not None and b._mtp is None
