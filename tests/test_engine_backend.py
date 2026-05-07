from mlxserve.runtime.engine import InferenceEngine


def test_engine_auto_backend_is_available():
    engine = InferenceEngine(backend_mode="auto")
    assert engine.backend is not None


def test_engine_stub_backend_selection():
    engine = InferenceEngine(backend_mode="stub")
    assert engine.backend.__class__.__name__ == "StubBackend"
