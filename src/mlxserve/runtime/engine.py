import time

from mlxserve.config import settings
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
            return MLXLMBackend(
                prefill_step_size=settings.prefill_step_size,
                kv_bits=settings.kv_bits,
                kv_group_size=settings.kv_group_size,
                quantized_kv_start=settings.quantized_kv_start,
            )
        if mode == "auto":
            try:
                import mlx_lm  # noqa: F401
                return MLXLMBackend(
                    prefill_step_size=settings.prefill_step_size,
                    kv_bits=settings.kv_bits,
                    kv_group_size=settings.kv_group_size,
                    quantized_kv_start=settings.quantized_kv_start,
                )
            except Exception:
                return StubBackend()
        return StubBackend()

    async def ensure_model(self, model: str) -> None:
        if self.loaded_model != model:
            await self.backend.load(model)
            self.loaded_model = model
        self.last_used_ts = time.time()

    async def generate_text(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ) -> str:
        await self.ensure_model(model)
        self.last_used_ts = time.time()
        return await self.backend.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    async def stream_text(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ):
        await self.ensure_model(model)
        self.last_used_ts = time.time()
        async for token in self.backend.stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            yield token

    def unload_model(self) -> None:
        self.backend.unload()
        self.loaded_model = None
