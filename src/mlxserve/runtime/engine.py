import asyncio
import time


class InferenceEngine:
    """MVP engine; can be replaced by mlx-lm backend later."""

    def __init__(self) -> None:
        self.loaded_model: str | None = None
        self.last_used_ts: float = time.time()

    async def ensure_model(self, model: str) -> None:
        if self.loaded_model != model:
            await asyncio.sleep(0)
            self.loaded_model = model
        self.last_used_ts = time.time()

    async def generate_text(self, model: str, prompt: str, max_tokens: int = 256) -> str:
        await self.ensure_model(model)
        self.last_used_ts = time.time()
        clipped_prompt = prompt.strip().replace("\n", " ")
        return f"MLXServe response: {clipped_prompt[: min(len(clipped_prompt), max_tokens)]}"

    async def stream_text(self, model: str, prompt: str, max_tokens: int = 256):
        text = await self.generate_text(model=model, prompt=prompt, max_tokens=max_tokens)
        for token in text.split(" "):
            await asyncio.sleep(0)
            yield token + " "

    def unload_model(self) -> None:
        self.loaded_model = None
