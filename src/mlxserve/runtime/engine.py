import time

from mlxserve.runtime.backends import MLXLMBackend, RuntimeBackend, StubBackend

class InferenceEngine:
    """Inference engine with optional mlx-lm backend and safe fallback."""

    def __init__(self, backend_mode: str = "auto") -> None:
        self.loaded_model: str | None = None
        self.last_used_ts: float = time.time()
        self.backend = self._build_backend(backend_mode)

    @staticmethod
    def _build_backend(backend_mode: str) -> RuntimeBackend:
        mode = backend_mode.lower()
        if mode == "stub":
            return StubBackend()
        if mode == "mlx":
            return MLXLMBackend()
        if mode == "auto":
            try:
                import mlx_lm  # noqa: F401
                return MLXLMBackend()
            except Exception:
                return StubBackend()
        return StubBackend()

    async def ensure_model(self, model: str) -> None:
        if self.loaded_model != model:
            await self.backend.load(model)
            self.loaded_model = model
        self.last_used_ts = time.time()

    async def generate_text(self, model: str, prompt: str, max_tokens: int = 256) -> str:
        await self.ensure_model(model)
        self.last_used_ts = time.time()
        return await self.backend.generate(prompt=prompt, max_tokens=max_tokens)

    async def stream_text(self, model: str, prompt: str, max_tokens: int = 256):
        await self.ensure_model(model)
        self.last_used_ts = time.time()
        async for token in self.backend.stream(prompt=prompt, max_tokens=max_tokens):
            yield token

    def unload_model(self) -> None:
        self.backend.unload()
        self.loaded_model = None
