from types import SimpleNamespace

import pytest

mlx_lm = pytest.importorskip("mlx_lm")
from mlx_lm.models import cache as mlx_cache  # noqa: E402

from mlxserve.runtime.backends import MLXLMBackend  # noqa: E402


class FakeCache:
    nbytes = 10

    def is_trimmable(self):
        return False  # like Qwen3.8 linear-attention layers: only exact-prefix reuse


class FakeTok:
    bos_token = None

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]


def test_second_turn_prefills_only_new_suffix(monkeypatch):
    prefilled = []

    def fake_stream_generate(model, tok, prompt, prompt_cache, **kw):
        assert prompt_cache is not None
        prefilled.append(list(prompt))
        for t in (1, 2):  # generated tokens, last one stands for EOS
            yield SimpleNamespace(token=t, text="x")

    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream_generate)
    monkeypatch.setattr(mlx_cache, "make_prompt_cache", lambda model: [FakeCache()])

    b = MLXLMBackend(prompt_cache_gb=1)
    b.model, b.tokenizer, b.model_name = object(), FakeTok(), "m"
    b._prompt_cache = mlx_cache.LRUPromptCache(max_bytes=10**9)

    def run(prompt):
        assert "".join(r.text for r in b._generate_sync(prompt, {})) == "xx"
        return b.last_cached_tokens

    assert run("abc") == 0
    # Next turn = previous prompt + generated tokens + new message: only "d" is prefilled.
    assert run("abc\x01\x02d") == 5
    assert prefilled[-1] == [ord("d")]
    # Unrelated conversation: full prefill.
    assert run("xyz") == 0
    assert prefilled[-1] == [ord(c) for c in "xyz"]


def test_cache_disabled(monkeypatch):
    seen = []
    monkeypatch.setattr(
        mlx_lm, "stream_generate",
        lambda model, tok, prompt, prompt_cache, **kw: seen.append(prompt_cache) or iter([SimpleNamespace(token=1, text="y")]),
    )
    b = MLXLMBackend(prompt_cache_gb=0)
    b.model, b.tokenizer, b.model_name = object(), FakeTok(), "m"
    assert "".join(r.text for r in b._generate_sync("abc", {})) == "y"
    assert seen == [None] and b.last_cached_tokens == 0


def test_snapshot_before_generation_prompt(monkeypatch):
    """Gemma 4 case: the generation prompt is not re-rendered in history, and the cache cannot be trimmed."""
    import mlx.core as mx

    class Cache(FakeCache):
        state = mx.array(0)

    class Tok(FakeTok):
        def apply_chat_template(self, msgs, add_generation_prompt, **kw):
            return "U:x" + ("|G" if add_generation_prompt else "")

    prefilled, model_fed = [], []
    monkeypatch.setattr(
        mlx_lm, "stream_generate",
        lambda model, tok, prompt, prompt_cache, **kw: prefilled.append(list(prompt)) or iter([SimpleNamespace(token=1, text="r")]),
    )
    monkeypatch.setattr(mlx_cache, "make_prompt_cache", lambda model: [Cache()])

    b = MLXLMBackend(prompt_cache_gb=1)
    b.model = lambda x, cache: model_fed.append(x.tolist()[0])
    b.tokenizer, b.model_name = Tok(), "m"
    b._prompt_cache = mlx_cache.LRUPromptCache(max_bytes=10**9)

    def run(prompt):
        "".join(r.text for r in b._generate_sync(prompt, {}))
        return b.last_cached_tokens

    assert run("hello|G") == 0
    assert model_fed == [[ord(c) for c in "hello"]]  # stable part prefilled, then snapshotted
    assert prefilled[-1] == [ord("|"), ord("G")]
    # Next turn re-renders the reply differently ("R" instead of the generated token 1): the snapshot still hits.
    assert run("helloRnext|G") == len("hello")
    assert model_fed[-1] == [ord(c) for c in "Rnext"]


def test_generate_and_stream_run_on_one_mlx_thread_and_stream_errors_surface():
    """A KV cache reused across requests must stay on the thread that created it (MLX streams are per thread)."""
    import asyncio
    import threading

    b = MLXLMBackend()
    b.model, b.tokenizer, b.model_name = object(), FakeTok(), "m"
    threads = []

    def fake_sync(prompt, gen_kw):
        threads.append(threading.get_ident())
        if prompt == "boom":
            raise RuntimeError("mlx failure")
        yield SimpleNamespace(token=1, text="ok")

    b._generate_sync = fake_sync

    async def scenario():
        assert await b.generate("a", 8) == "ok"
        assert [c async for c in b.stream("b", 8)] == ["ok"]
        assert [c async for c in b.stream("c", 8)] == ["ok"]
        with pytest.raises(RuntimeError, match="mlx failure"):
            [c async for c in b.stream("boom", 8)]

    asyncio.run(scenario())
    assert len(set(threads)) == 1 and threads[0] != threading.get_ident()


def test_reused_entry_is_refreshed_in_the_lru(monkeypatch):
    """The agent's history snapshot is reused every turn: it must not be the first one evicted."""
    monkeypatch.setattr(mlx_lm, "stream_generate", lambda model, tok, prompt, prompt_cache, **kw: iter([SimpleNamespace(token=1, text="x")]))
    monkeypatch.setattr(mlx_cache, "make_prompt_cache", lambda model: [FakeCache()])
    b = MLXLMBackend(prompt_cache_gb=1)
    b.model, b.tokenizer, b.model_name = object(), FakeTok(), "m"
    b._prompt_cache = mlx_cache.LRUPromptCache(max_size=2)

    def run(prompt):
        "".join(r.text for r in b._generate_sync(prompt, {}))
        return b.last_cached_tokens

    run("history")                     # stores "history" + generated
    assert run("history\x01A") == 8    # reuses it (and refreshes it)
    run("other")                       # a third entry: the LRU (size 2) evicts the least recent one
    assert run("history\x01B") == 8    # the reused history survived
