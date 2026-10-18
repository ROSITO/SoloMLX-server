from mlxserve.runtime.engine import InferenceEngine


def test_engine_auto_backend_is_available():
    engine = InferenceEngine(backend_mode="auto")
    assert engine.backend is not None


def test_engine_stub_backend_selection():
    engine = InferenceEngine(backend_mode="stub")
    assert engine.backend.__class__.__name__ == "StubBackend"


def test_concurrent_first_requests_load_the_model_once():
    import asyncio

    from mlxserve.runtime.backends import StubBackend
    from mlxserve.runtime.engine import InferenceEngine

    loads = []

    class SlowStub(StubBackend):
        async def load(self, model):
            loads.append(model)
            await asyncio.sleep(0.05)
            await super().load(model)

    async def scenario():
        e = InferenceEngine("stub")
        e.backend = SlowStub()
        await asyncio.gather(e.ensure_model("m"), e.ensure_model("m"), e.ensure_model("m"))

    asyncio.run(scenario())
    assert loads == ["m"]
