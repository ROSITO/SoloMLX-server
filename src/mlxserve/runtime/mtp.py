"""Gemma 4 multi-token prediction (MTP): speculative decoding with Google's assistant drafter.

The drafter (4 layers, ~225 MB) reads the target's last hidden state and the K/V of its last full and
sliding attention layers, and guesses the next tokens; the target checks them in one forward pass and
keeps the longest agreeing prefix plus one token of its own. Rejected tokens are rewound out of the KV
cache. With temperature > 0, standard speculative sampling keeps the target's distribution exactly.

Reference algorithm: transformers' Gemma 4 assistant candidate generator. mlx-lm has no MTP loop (and
its speculative decoding refuses Gemma 4's sliding-window cache past 1k tokens), hence this module.
"""
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import mlx.core as mx

from mlxserve.runtime import gemma4_assistant

_QUANT_SUFFIX = re.compile(r"-(\d+bit|bf16|fp16|mxfp4|nvfp4)$")


def assistant_repo(model_id: str) -> str | None:
    """mlx-community naming: gemma-4-26B-A4B-it-qat-4bit -> gemma-4-26B-A4B-it-qat-assistant-4bit."""
    m = _QUANT_SUFFIX.search(model_id)
    return f"{model_id[: m.start()]}-assistant{m.group(0)}" if m and "gemma-4" in model_id else None


def load_mtp(model_id: str, model: Any, tokenizer: Any, draft_id: str = "", max_draft: int = 3) -> "GemmaMTP | None":
    """GemmaMTP for a Gemma 4 target, or None (other model, no drafter found)."""
    lm = getattr(model, "language_model", None)
    if lm is None or not str(getattr(lm, "model_type", "")).startswith("gemma4"):
        return None
    draft_id = draft_id or assistant_repo(model_id)
    if not draft_id:
        return None
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_model

    path = Path(draft_id) if Path(draft_id).exists() else Path(snapshot_download(draft_id))
    draft, _ = load_model(path, get_model_classes=lambda config: (gemma4_assistant.Model, gemma4_assistant.ModelArgs))
    return GemmaMTP(model, tokenizer, draft, max_draft=max_draft)


def _rewind(cache: list[Any], n: int) -> None:
    """Drop the last n tokens from every layer's KV cache."""
    if n <= 0:
        return
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    for c in cache:
        if isinstance(c, RotatingKVCache):
            if c.keys.shape[2] == c._idx:
                # After a multi-token (concat) update the buffer is in temporal order: drop the tail.
                c.keys, c.values = c.keys[..., :-n, :], c.values[..., :-n, :]
            elif c._idx < n:
                raise RuntimeError("cannot rewind a sliding-window cache across its rotation point")
            c.offset -= n
            c._idx -= n
        elif isinstance(c, KVCache):
            c.trim(n)
        else:
            raise TypeError(f"MTP cannot rewind {type(c).__name__}")


def _probs(logits: mx.array, temperature: float, top_p: float) -> mx.array:
    p = mx.softmax(logits.astype(mx.float32) / temperature, axis=-1)
    if top_p < 1.0:
        sp = mx.sort(p, axis=-1)[..., ::-1]
        kept = (mx.cumsum(sp, axis=-1) - sp) < top_p
        floor = mx.min(mx.where(kept, sp, mx.inf), axis=-1, keepdims=True)
        p = mx.where(p >= floor, p, 0.0)
        p = p / mx.sum(p, axis=-1, keepdims=True)
    return p


def _sample(p: mx.array) -> int:
    return mx.random.categorical(mx.log(p)).item()  # categorical normalizes: p need not sum to 1


def accept_sampled(p: mx.array, q: list[mx.array], drafts: list[int], eos: set[int]) -> tuple[int, int]:
    """Speculative sampling. p: target probs (k+1, V); q[i]: draft probs for drafts[i].

    Returns (number of accepted drafts m, next token): drafts[:m] are kept, then the next token is
    resampled from max(p - q, 0) at the first rejection, or drawn from p after the last draft.
    """
    k = len(drafts)
    if k:  # all acceptance tests in one GPU round trip
        idx, d = mx.arange(k), mx.array(drafts)
        pd, qd, u = mx.stack([p[idx, d], mx.stack(q)[idx, d], mx.random.uniform(shape=(k,))]).tolist()
    for i, tok in enumerate(drafts):
        if tok in eos:
            return i, _sample(p[i])
        if u[i] * qd[i] >= pd[i]:  # reject with probability 1 - p/q
            residual = mx.maximum(p[i] - q[i], 0.0)
            return i, _sample(mx.where(mx.sum(residual) > 0, residual, p[i]))
    return k, _sample(p[k])


class GemmaMTP:
    # Keep drafting while the drafter is at least this sure of its token. With max_draft 3, tuned on chat, code,
    # 8.5k-token agent and tool-call prompts: x1.33 geometric mean (0.6: x1.31; max_draft 2: x1.27).
    confidence = 0.4

    def __init__(self, model: Any, tokenizer: Any, draft: Any, max_draft: int = 3):
        self.model, self.tokenizer, self.draft = model, tokenizer, draft
        self.lm = model.language_model
        self.tm = self.lm.model
        types = self.lm.args.layer_types
        self.kv_layer = {t: max(i for i, lt in enumerate(types) if lt == t) for t in ("full_attention", "sliding_attention")}
        for layer in draft.layers:  # the drafter's Q must be rotated exactly like the target's K
            layer.self_attn.rope = self.tm.layers[self.kv_layer[layer.layer_type]].self_attn.rope
        self.eos = set(tokenizer.eos_token_ids)
        self.max_draft = max_draft
        self.fed: list[int] = []  # tokens added to the KV cache after the prompt, for the prompt-cache key
        self.stats = {"cycles": 0, "drafted": 0, "accepted": 0}

    def _target(self, tokens: list[int], cache: list[Any]) -> tuple[mx.array, mx.array]:
        h = self.tm(mx.array(tokens)[None], cache=cache)
        logits = self.tm.embed_tokens.as_linear(h)
        sc = self.lm.final_logit_softcapping
        return h, (mx.tanh(logits / sc) * sc if sc else logits)

    def _shared_kv(self, cache: list[Any]) -> dict[str, tuple[mx.array, mx.array]]:
        from mlx_lm.models.cache import RotatingKVCache

        out = {}
        for t, i in self.kv_layer.items():
            c = cache[i]
            out[t] = c.keys_and_values() if isinstance(c, RotatingKVCache) else (c.keys[..., : c.offset, :], c.values[..., : c.offset, :])
        return out

    def _draft(self, y: int, h_last: mx.array, pos: int, cache: list[Any], temperature: float, top_p: float):
        kv, hid, posa = self._shared_kv(cache), h_last, mx.array([[pos]])
        drafts, qs = [], []
        for _ in range(self.max_draft):
            emb = self.tm.embed_tokens(mx.array([[y]])) * self.tm.embed_scale
            hid, logits = self.draft(mx.concatenate([emb.astype(hid.dtype), hid], axis=-1), kv, position_ids=posa)
            if temperature == 0:
                q = mx.softmax(logits[0, -1].astype(mx.float32))
                tok = mx.argmax(q)
            else:
                q = _probs(logits[0, -1], temperature, top_p)
                tok = mx.random.categorical(mx.log(q))
                qs.append(q)
            y, conf = mx.stack([tok.astype(mx.float32), q[tok]]).tolist()  # one GPU round trip per step
            y = int(y)
            drafts.append(y)
            if y in self.eos or conf < self.confidence:
                break
        return drafts, qs

    def generate(self, prompt: list[int], cache: list[Any], max_tokens: int, temperature: float, top_p: float,
                 prefill_step: int = 2048) -> Iterator[SimpleNamespace]:
        """Yield SimpleNamespace(text, token) like mlx_lm.stream_generate. `prompt` = tokens not in `cache` yet."""
        from mlx_lm.generate import generation_stream, wired_limit

        # Same execution context as stream_generate: dedicated GPU stream, weights wired in memory.
        with wired_limit(self.model, [generation_stream]), mx.stream(generation_stream):
            yield from self._generate(prompt, cache, max_tokens, temperature, top_p, prefill_step)

    def _generate(self, prompt, cache, max_tokens, temperature, top_p, prefill_step):
        self.fed = []
        for i in range(0, len(prompt), prefill_step):
            h, logits = self._target(prompt[i : i + prefill_step], cache)
        last = logits[0, -1]
        y = mx.argmax(last).item() if temperature == 0 else _sample(_probs(last, temperature, top_p))
        h_last = h[:, -1:]
        pos = cache[self.kv_layer["full_attention"]].offset
        detok = self.tokenizer.detokenizer
        detok.reset()
        produced = 0
        while True:
            if y in self.eos:
                break
            detok.add_token(y)
            produced += 1
            yield SimpleNamespace(text=detok.last_segment, token=y)
            if produced >= max_tokens:
                break
            drafts, qs = self._draft(y, h_last, pos, cache, temperature, top_p)
            h, logits = self._target([y] + drafts, cache)
            if temperature == 0:
                pred = mx.argmax(logits[0], axis=-1).tolist()
                m = 0
                while m < len(drafts) and drafts[m] == pred[m] and drafts[m] not in self.eos:
                    m += 1
                nxt = pred[m]
            else:
                m, nxt = accept_sampled(_probs(logits[0], temperature, top_p), qs, drafts, self.eos)
            _rewind(cache, len(drafts) - m)
            self.fed += [y] + drafts[:m]
            self.stats["cycles"] += 1
            self.stats["drafted"] += len(drafts)
            self.stats["accepted"] += m
            for d in drafts[:m]:
                detok.add_token(d)
                produced += 1
                yield SimpleNamespace(text=detok.last_segment, token=d)
                if produced >= max_tokens:
                    break
            if produced >= max_tokens:
                break
            y, h_last, pos = nxt, h[:, m : m + 1], pos + m + 1
        detok.finalize()
        if detok.last_segment:
            yield SimpleNamespace(text=detok.last_segment, token=None)
