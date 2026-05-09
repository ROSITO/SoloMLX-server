import time

from mlxserve.config import settings
from mlxserve.runtime.backends import (
    ExperimentalMoEStubBackend,
    MLXLMBackend,
    RuntimeBackend,
    StubBackend,
)

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
        if mode == "moe_stub":
            return ExperimentalMoEStubBackend(
                num_experts=settings.moe_num_experts,
                top_k=settings.moe_top_k,
                num_shared_experts=settings.moe_num_shared_experts,
            )
        if mode == "mlx":
            return MLXLMBackend(
                prefill_step_size=settings.prefill_step_size,
                kv_bits=settings.kv_bits,
                kv_group_size=settings.kv_group_size,
                quantized_kv_start=settings.quantized_kv_start,
                moe_resident_experts=settings.moe_resident_experts,
                moe_resident_strategy=settings.moe_resident_strategy,
                moe_single_expert_fastpath=settings.moe_single_expert_fastpath,
                repetition_penalty=settings.generation_repetition_penalty,
                repetition_context_size=settings.generation_repetition_context_size,
            )
        if mode == "auto":
            try:
                import mlx_lm  # noqa: F401
                return MLXLMBackend(
                    prefill_step_size=settings.prefill_step_size,
                    kv_bits=settings.kv_bits,
                    kv_group_size=settings.kv_group_size,
                    quantized_kv_start=settings.quantized_kv_start,
                    moe_resident_experts=settings.moe_resident_experts,
                    moe_resident_strategy=settings.moe_resident_strategy,
                    moe_single_expert_fastpath=settings.moe_single_expert_fastpath,
                    repetition_penalty=settings.generation_repetition_penalty,
                    repetition_context_size=settings.generation_repetition_context_size,
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
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop_sequences: list[str] | None = None,
    ) -> str:
        await self.ensure_model(model)
        self.last_used_ts = time.time()
        prompt = await self.backend.build_chat_prompt(messages)
        return await self.backend.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
        )

    async def stream_text(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop_sequences: list[str] | None = None,
    ):
        await self.ensure_model(model)
        self.last_used_ts = time.time()
        prompt = await self.backend.build_chat_prompt(messages)
        async for token in self.backend.stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
        ):
            yield token

    async def prompt_and_completion_token_counts(
        self,
        model: str,
        messages: list[dict[str, str]],
        completion: str,
    ) -> tuple[int, int]:
        await self.ensure_model(model)
        prompt = await self.backend.build_chat_prompt(messages)
        prompt_tokens = await self.backend.count_tokens(prompt)
        completion_tokens = await self.backend.count_tokens(completion)
        return prompt_tokens, completion_tokens

    def unload_model(self) -> None:
        self.backend.unload()
        self.loaded_model = None
