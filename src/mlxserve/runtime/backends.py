import asyncio
from typing import Any


class RuntimeBackend:
    async def load(self, model: str) -> None:
        raise NotImplementedError

    async def generate(self, prompt: str, max_tokens: int) -> str:
        raise NotImplementedError

    async def stream(self, prompt: str, max_tokens: int):
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError


class StubBackend(RuntimeBackend):
    def __init__(self) -> None:
        self.model: str | None = None

    async def load(self, model: str) -> None:
        self.model = model

    async def generate(self, prompt: str, max_tokens: int) -> str:
        clipped_prompt = prompt.strip().replace("\n", " ")
        return f"MLXServe response: {clipped_prompt[: min(len(clipped_prompt), max_tokens)]}"

    async def stream(self, prompt: str, max_tokens: int):
        text = await self.generate(prompt=prompt, max_tokens=max_tokens)
        for token in text.split(" "):
            await asyncio.sleep(0)
            yield token + " "

    def unload(self) -> None:
        self.model = None


class MLXLMBackend(RuntimeBackend):
    def __init__(self) -> None:
        self.model_name: str | None = None
        self.model: Any = None
        self.tokenizer: Any = None

    async def load(self, model: str) -> None:
        if self.model_name == model and self.model is not None and self.tokenizer is not None:
            return
        try:
            from mlx_lm import load as mlx_load
        except Exception as exc:
            raise RuntimeError("mlx-lm is not installed or failed to import") from exc
        self.model, self.tokenizer = await asyncio.to_thread(mlx_load, model)
        self.model_name = model

    async def generate(self, prompt: str, max_tokens: int) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded")
        from mlx_lm import generate as mlx_generate

        return await asyncio.to_thread(
            mlx_generate,
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

    async def stream(self, prompt: str, max_tokens: int):
        text = await self.generate(prompt=prompt, max_tokens=max_tokens)
        for token in text.split(" "):
            await asyncio.sleep(0)
            yield token + " "

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_name = None
