import asyncio
import copy
import re
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

from mlxserve.runtime.stop_sequences import merge_chat_boundary_stops, truncate_at_stop_sequences
from mlxserve.runtime.moe_offload import apply_moe_expert_offload
from mlxserve.runtime.tool_calls import ToolCallFormat


class RuntimeBackend:
    async def load(self, model: str) -> None:
        raise NotImplementedError

    async def build_chat_prompt(self, messages: list[dict], tools: list[dict] | None = None) -> str:
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

    def tool_call_format(self) -> ToolCallFormat | None:
        """(start, end, parser) for the loaded model, or None when it has no tool calling."""
        return None

    def unload(self) -> None:
        raise NotImplementedError


class StubBackend(RuntimeBackend):
    def __init__(self) -> None:
        self.model: str | None = None

    async def load(self, model: str) -> None:
        self.model = model

    async def build_chat_prompt(self, messages: list[dict], tools: list[dict] | None = None) -> str:
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

    async def build_chat_prompt(self, messages: list[dict], tools: list[dict] | None = None) -> str:
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
        repetition_penalty: float = 1.0,
        repetition_context_size: int = 64,
        enable_thinking: bool = False,
        prompt_cache_gb: float = 0.0,
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
        self.repetition_penalty = repetition_penalty
        self.repetition_context_size = repetition_context_size
        self.enable_thinking = enable_thinking
        self.prompt_cache_gb = prompt_cache_gb
        self._prompt_cache: Any = None  # mlx_lm LRUPromptCache, reset on (re)load
        self._prompt_cache_lock = Lock()
        # All MLX work runs on this one thread: arrays (e.g. a reused KV cache) are bound to the GPU
        # stream of the thread that created them, and MLX runs one generation at a time anyway.
        self._mlx_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._gen_suffix: str | None = None  # chat template's generation prompt, per loaded model
        self.last_cached_tokens = 0

    def _mlx_sample_and_logits_kwargs(
        self, max_tokens: int, temperature: float, top_p: float
    ) -> dict[str, Any]:
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        out: dict[str, Any] = {
            "max_tokens": max_tokens,
            "sampler": make_sampler(temp=temperature, top_p=top_p),
            "prefill_step_size": self.prefill_step_size,
            "kv_bits": self.kv_bits,
            "kv_group_size": self.kv_group_size,
            "quantized_kv_start": self.quantized_kv_start,
        }
        if self.repetition_penalty is not None and self.repetition_penalty > 1.0:
            out["logits_processors"] = make_logits_processors(
                repetition_penalty=self.repetition_penalty,
                repetition_context_size=self.repetition_context_size,
            )
        return out

    async def build_chat_prompt(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Model is not loaded")

        def _from_template() -> str:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,  # ignored by templates without it
                **({"tools": tools} if tools else {}),
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

    def tool_call_format(self) -> ToolCallFormat | None:
        t = self.tokenizer
        if t is None or not getattr(t, "has_tool_calling", False):
            return None
        return t.tool_call_start, t.tool_call_end, t.tool_parser

    async def load(self, model: str) -> None:
        if self.model_name == model and self.model is not None and self.tokenizer is not None:
            return
        try:
            from mlx_lm import load as mlx_load
        except Exception as exc:
            raise RuntimeError("mlx-lm is not installed or failed to import") from exc
        self.model, self.tokenizer = await self._on_mlx_thread(lambda: mlx_load(model))
        self._prompt_cache = None
        self._gen_suffix = None
        if self.prompt_cache_gb > 0:
            from mlx_lm.models.cache import LRUPromptCache

            self._prompt_cache = LRUPromptCache(max_bytes=int(self.prompt_cache_gb * 1e9))
        _ = apply_moe_expert_offload(
            self.model,
            keep_experts=self.moe_resident_experts,
            strategy=self.moe_resident_strategy,
            enable_single_expert_fastpath=self.moe_single_expert_fastpath,
        )
        self.model_name = model

    def _generation_suffix(self) -> str:
        """Text the template appends for add_generation_prompt (e.g. Gemma 4: model turn opener + empty thought block).

        Templates often render it differently once the reply is in the history, so the KV is only
        reusable up to just before it.
        """
        if self._gen_suffix is None:
            msgs = [{"role": "user", "content": "x"}]
            try:
                kw = {"tokenize": False, "enable_thinking": self.enable_thinking}
                with_gen = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, **kw)
                without = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=False, **kw)
                self._gen_suffix = with_gen[len(without) :] if with_gen.startswith(without) else ""
            except Exception:
                self._gen_suffix = ""
        return self._gen_suffix

    async def _on_mlx_thread(self, fn):
        return await asyncio.get_running_loop().run_in_executor(self._mlx_thread, fn)

    def _prefill(self, tokens: list[int], cache: list[Any], step: int) -> None:
        import mlx.core as mx
        from mlx_lm.generate import generation_stream

        with mx.stream(generation_stream):  # same stream stream_generate uses for this cache
            for i in range(0, len(tokens), step):
                self.model(mx.array(tokens[i : i + step])[None], cache=cache)
                mx.eval([c.state for c in cache])

    def _generate_sync(self, prompt: str, gen_kw: dict[str, Any]):
        """stream_generate, resuming from the cached KV of the longest known token prefix.

        Agent loops resend the whole conversation each turn; only the new suffix is prefilled.
        The cache holds prompt + every yielded token (EOS included), which is the key stored back.
        """
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache

        tok = self.tokenizer
        add_special = tok.bos_token is None or not prompt.startswith(tok.bos_token)
        tokens = tok.encode(prompt, add_special_tokens=add_special)
        store = self._prompt_cache
        cache, rest = None, tokens
        if store is not None:
            with self._prompt_cache_lock:
                cache, rest = store.fetch_nearest_cache(self.model_name, tokens)
            if cache is None or not rest:
                cache, rest = make_prompt_cache(self.model), tokens
        self.last_cached_tokens = len(tokens) - len(rest)
        suffix = self._generation_suffix() if store is not None else ""
        if suffix and prompt.endswith(suffix):
            # Snapshot the KV at the end of the stable part (history, before the generation prompt):
            # the next turn starts with exactly these tokens even when the reply is re-rendered.
            stable = tok.encode(prompt[: -len(suffix)], add_special_tokens=add_special)
            start = len(tokens) - len(rest)
            if len(stable) > start and tokens[: len(stable)] == stable:
                self._prefill(tokens[start : len(stable)], cache, gen_kw.get("prefill_step_size") or 2048)
                with self._prompt_cache_lock:
                    store.insert_cache(self.model_name, stable, copy.deepcopy(cache))
                rest = tokens[len(stable) :]
        generated: list[int] = []
        for item in stream_generate(self.model, tok, prompt=rest, prompt_cache=cache, **gen_kw):
            generated.append(item.token)
            yield item
        if store is not None:
            with self._prompt_cache_lock:
                store.insert_cache(self.model_name, tokens + generated, cache)

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
        gen_kw = self._mlx_sample_and_logits_kwargs(max_tokens, temperature, top_p)
        raw = await self._on_mlx_thread(lambda: "".join(r.text for r in self._generate_sync(prompt, gen_kw)))
        merged = merge_chat_boundary_stops(self.model_name, stop_sequences)
        trimmed, _ = truncate_at_stop_sequences(raw, merged)
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
        merged_stops = merge_chat_boundary_stops(self.model_name, stop_sequences)
        gen_kw = self._mlx_sample_and_logits_kwargs(max_tokens, temperature, top_p)

        q: Queue[str | BaseException | None] = Queue()

        def worker() -> None:
            try:
                for item in self._generate_sync(prompt, gen_kw):
                    if item.text:
                        q.put(item.text)
            except BaseException as exc:  # surface it: an empty 200 makes agents loop on "continue"
                q.put(exc)
            finally:
                q.put(None)

        self._mlx_thread.submit(worker)
        assembled = ""
        emitted_upto = 0
        while True:
            piece = await asyncio.to_thread(q.get)
            if isinstance(piece, BaseException):
                raise piece
            if piece is None:
                trimmed, _ = truncate_at_stop_sequences(assembled, merged_stops)
                tail = trimmed[emitted_upto:]
                if tail:
                    yield tail
                break
            assembled += piece
            trimmed, hit = truncate_at_stop_sequences(assembled, merged_stops)
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
        self._prompt_cache = None
