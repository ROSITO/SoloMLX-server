import asyncio
import re
from queue import Queue
from threading import Thread
from typing import Any

from mlxserve.runtime.stop_sequences import truncate_at_stop_sequences
from mlxserve.runtime.moe_offload import apply_moe_expert_offload


class RuntimeBackend:
    async def load(self, model: str) -> None:
        raise NotImplementedError

    async def build_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop_sequences: list[str] | None = None,
    ) -> str:
        raise NotImplementedError

    async def stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop_sequences: list[str] | None = None,
    ):
        raise NotImplementedError

    async def count_tokens(self, text: str) -> int:
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

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop_sequences: list[str] | None = None,
    ) -> str:
        clipped_prompt = prompt.strip().replace("\n", " ")
        raw = f"MLXServe response: {clipped_prompt[: min(len(clipped_prompt), max_tokens)]}"
        trimmed, _ = truncate_at_stop_sequences(raw, stop_sequences)
        return trimmed

    async def stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop_sequences: list[str] | None = None,
    ):
        text = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
        )
        for token in text.split(" "):
            await asyncio.sleep(0)
            yield token + " "

    async def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())

    def unload(self) -> None:
        self.model = None


class ExperimentalMoEStubBackend(RuntimeBackend):
    """Prototype MoE backend for phase-2 experimentation.

    This is not a trained MoE model. It emulates routing + shared experts
    so we can validate A/B integration, observability and API behavior.
    """

    def __init__(self, num_experts: int = 4, top_k: int = 2, num_shared_experts: int = 1) -> None:
        self.model: str | None = None
        self.num_experts = max(2, num_experts)
        self.top_k = max(1, min(top_k, self.num_experts))
        self.num_shared_experts = max(0, min(num_shared_experts, self.num_experts - 1))

    async def load(self, model: str) -> None:
        self.model = model

    async def build_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    @staticmethod
    def _tokenize_for_routing(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    def _route_experts(self, prompt: str) -> list[int]:
        tokens = self._tokenize_for_routing(prompt)
        # Lightweight task-specialized scoring for deterministic routing.
        score_buckets = {
            0: {"code", "python", "javascript", "function", "class", "bug"},
            1: {"math", "percent", "equation", "proof", "derive", "sum"},
            2: {"why", "compare", "tradeoff", "reason", "plan", "strategy"},
            3: {"translate", "resume", "explain", "brief", "summary", "note"},
        }
        scores = [0] * self.num_experts
        for tok in tokens:
            for idx in range(self.num_experts):
                bucket = score_buckets.get(idx, set())
                if tok in bucket:
                    scores[idx] += 2
                # hash fallback keeps experts utilized for unknown domains.
                if (hash(tok) % self.num_experts) == idx:
                    scores[idx] += 1
        ranked = sorted(range(self.num_experts), key=lambda i: scores[i], reverse=True)
        selected = ranked[: self.top_k]
        if self.num_shared_experts > 0:
            selected += list(range(self.num_shared_experts))
        # preserve order while removing duplicates
        seen: set[int] = set()
        out: list[int] = []
        for idx in selected:
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
        return out

    def _expert_fragment(self, idx: int, prompt: str) -> str:
        short = prompt.strip().replace("\n", " ")[:180]
        templates = {
            0: f"[expert-code] Proposition de solution: {short}",
            1: f"[expert-math] Analyse numerique: {short}",
            2: f"[expert-reasoning] Decomposition des etapes: {short}",
            3: f"[expert-general] Reponse synthetique: {short}",
        }
        return templates.get(idx, f"[expert-{idx}] {short}")

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop_sequences: list[str] | None = None,
    ) -> str:
        experts = self._route_experts(prompt)
        fragments = [self._expert_fragment(i, prompt) for i in experts]
        raw = (
            "MLXServe MoE-proto response:\n"
            + "\n".join(fragments)
            + f"\n[routing] experts={experts} top_k={self.top_k} shared={self.num_shared_experts}"
        )
        clipped = raw[: max(1, max_tokens * 8)]
        trimmed, _ = truncate_at_stop_sequences(clipped, stop_sequences)
        return trimmed

    async def stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop_sequences: list[str] | None = None,
    ):
        text = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
        )
        for token in text.split(" "):
            await asyncio.sleep(0)
            yield token + " "

    async def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())

    def unload(self) -> None:
        self.model = None


class MLXLMBackend(RuntimeBackend):
    def __init__(
        self,
        prefill_step_size: int = 2048,
        kv_bits: int | None = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
        moe_resident_experts: int = 0,
        moe_resident_strategy: str = "l2",
        moe_single_expert_fastpath: bool = True,
    ) -> None:
        self.model_name: str | None = None
        self.model: Any = None
        self.tokenizer: Any = None
        self.prefill_step_size = prefill_step_size
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start
        self.moe_resident_experts = moe_resident_experts
        self.moe_resident_strategy = moe_resident_strategy
        self.moe_single_expert_fastpath = moe_single_expert_fastpath

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

    async def count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Model is not loaded")

        def _run() -> int:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            return len(encoded)

        return await asyncio.to_thread(_run)

    async def load(self, model: str) -> None:
        if self.model_name == model and self.model is not None and self.tokenizer is not None:
            return
        try:
            from mlx_lm import load as mlx_load
        except Exception as exc:
            raise RuntimeError("mlx-lm is not installed or failed to import") from exc
        self.model, self.tokenizer = await asyncio.to_thread(mlx_load, model)
        _ = apply_moe_expert_offload(
            self.model,
            keep_experts=self.moe_resident_experts,
            strategy=self.moe_resident_strategy,
            enable_single_expert_fastpath=self.moe_single_expert_fastpath,
        )
        self.model_name = model

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop_sequences: list[str] | None = None,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded")
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        raw = await asyncio.to_thread(
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
        trimmed, _ = truncate_at_stop_sequences(raw, stop_sequences)
        return trimmed

    async def stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop_sequences: list[str] | None = None,
    ):
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
        assembled = ""
        emitted_upto = 0
        while True:
            piece = await asyncio.to_thread(q.get)
            if piece is None:
                trimmed, _ = truncate_at_stop_sequences(assembled, stop_sequences)
                tail = trimmed[emitted_upto:]
                if tail:
                    yield tail
                break
            assembled += piece
            trimmed, hit = truncate_at_stop_sequences(assembled, stop_sequences)
            chunk = trimmed[emitted_upto:]
            if chunk:
                yield chunk
                emitted_upto = len(trimmed)
            if hit:
                while True:
                    drain = await asyncio.to_thread(q.get)
                    if drain is None:
                        break
                break

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_name = None
