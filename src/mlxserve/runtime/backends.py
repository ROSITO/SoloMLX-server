import asyncio
from queue import Queue
from threading import Thread
from typing import Any


class RuntimeBackend:
    async def load(self, model: str) -> None:
        raise NotImplementedError

    async def build_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    async def generate(self, prompt: str, max_tokens: int, temperature: float = 0.2, top_p: float = 0.95) -> str:
        raise NotImplementedError

    async def stream(self, prompt: str, max_tokens: int, temperature: float = 0.2, top_p: float = 0.95):
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError


class StubBackend(RuntimeBackend):
    def __init__(self) -> None:
        self.model: str | None = None

    async def load(self, model: str) -> None:
        self.model = model

    async def build_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    async def generate(self, prompt: str, max_tokens: int, temperature: float = 0.2, top_p: float = 0.95) -> str:
        clipped_prompt = prompt.strip().replace("\n", " ")
        return f"MLXServe response: {clipped_prompt[: min(len(clipped_prompt), max_tokens)]}"

    async def stream(self, prompt: str, max_tokens: int, temperature: float = 0.2, top_p: float = 0.95):
        text = await self.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        for token in text.split(" "):
            await asyncio.sleep(0)
            yield token + " "

    def unload(self) -> None:
        self.model = None


class MLXLMBackend(RuntimeBackend):
    def __init__(
        self,
        prefill_step_size: int = 2048,
        kv_bits: int | None = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
    ) -> None:
        self.model_name: str | None = None
        self.model: Any = None
        self.tokenizer: Any = None
        self.prefill_step_size = prefill_step_size
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start

    async def build_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Model is not loaded")

        def _from_template() -> str:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        if getattr(self.tokenizer, "has_chat_template", False):
            return await asyncio.to_thread(_from_template)
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    async def load(self, model: str) -> None:
        if self.model_name == model and self.model is not None and self.tokenizer is not None:
            return
        try:
            from mlx_lm import load as mlx_load
        except Exception as exc:
            raise RuntimeError("mlx-lm is not installed or failed to import") from exc
        self.model, self.tokenizer = await asyncio.to_thread(mlx_load, model)
        self.model_name = model

    async def generate(self, prompt: str, max_tokens: int, temperature: float = 0.2, top_p: float = 0.95) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded")
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        return await asyncio.to_thread(
            mlx_generate,
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
            sampler=make_sampler(temp=temperature, top_p=top_p),
            prefill_step_size=self.prefill_step_size,
            kv_bits=self.kv_bits,
            kv_group_size=self.kv_group_size,
            quantized_kv_start=self.quantized_kv_start,
        )

    async def stream(self, prompt: str, max_tokens: int, temperature: float = 0.2, top_p: float = 0.95):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded")
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        q: Queue[str | None] = Queue()

        def worker() -> None:
            try:
                for item in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=make_sampler(temp=temperature, top_p=top_p),
                    prefill_step_size=self.prefill_step_size,
                    kv_bits=self.kv_bits,
                    kv_group_size=self.kv_group_size,
                    quantized_kv_start=self.quantized_kv_start,
                ):
                    if item.text:
                        q.put(item.text)
            finally:
                q.put(None)

        Thread(target=worker, daemon=True).start()
        while True:
            token = await asyncio.to_thread(q.get)
            if token is None:
                break
            yield token

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_name = None
