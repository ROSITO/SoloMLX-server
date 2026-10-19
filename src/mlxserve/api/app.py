import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from mlxserve.api.deps import engine, guardian, metrics, model_manager, require_api_key
from mlxserve.config import settings
from mlxserve.memory.estimate import admission_memory_gb, heuristic_prompt_tokens
from mlxserve.models.schemas import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChoiceMessage,
    HealthResponse,
    ModelItem,
    ModelsResponse,
    LocalModelItem,
    LocalModelsResponse,
    PullModelRequest,
    RecommendedModelItem,
    RecommendedModelsResponse,
    Usage,
)
from mlxserve.models.catalog import machine_ram_gb, model_context_length, recommended_for_machine
from mlxserve.runtime.stop_sequences import normalize_stop_sequences
from mlxserve.runtime.reasoning import EmptyReasoningFilter, strip_empty_reasoning
from mlxserve.runtime.tool_calls import extract_tool_calls, partial_marker_len, template_messages

logger = logging.getLogger("mlxserve.api")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Optional eager load so the first client request is not paying cold-start."""
    if settings.preload_default_model:
        model = settings.default_model
        try:
            await engine.ensure_model(model)
            logger.info("Preloaded default model: %s", model)
        except Exception:
            logger.exception(
                "Preload of default model failed; server continues without weights in RAM: %s",
                model,
            )
    yield


app = FastAPI(title="MLXServe", version="0.1.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_allow_origins.split(",")] if settings.cors_allow_origins else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
_rate_bucket: dict[str, deque[float]] = defaultdict(deque)
_chat_ui_path = Path(__file__).resolve().parents[1] / "web" / "chat.html"
_HF_CACHE_DIR_RE = re.compile(r"models--([^/\\]+)[/\\]snapshots[/\\]", re.IGNORECASE)
_HF_CACHE_TAIL_RE = re.compile(r"models--([^/\\]+)$")


def display_hf_model_id(raw: str) -> str:
    """Turn a Hugging Face hub cache path into org/name when possible (else unchanged)."""
    if not raw or ("/" not in raw and "\\" not in raw):
        return raw
    norm = raw.replace("\\", "/")
    m = _HF_CACHE_DIR_RE.search(norm) or _HF_CACHE_TAIL_RE.search(norm)
    if not m:
        return raw
    encoded = m.group(1)
    i = encoded.find("--")
    if i == -1:
        return encoded
    org, rest = encoded[:i], encoded[i + 2 :]
    return f"{org}/{rest.replace('--', '/')}"


def _metrics_model_label(model_id: str) -> str:
    s = model_id.replace("\\", "/").replace("\n", " ")[: settings.metrics_model_label_max_len]
    return s if s else "unknown"


# Role-turn echoes: must not use naive ``.find("system:")`` (matches inside ``ecosystem:``).
_ROLEPLAY_SPOOF_RE = re.compile(
    r"(?:(?:^|\n)\s*|(?<!\w))(assistant|user|system)\s*:",
    re.IGNORECASE,
)

def _strip_trailing_role_prefix_fragment(text: str) -> str:
    """Remove a trailing whitespace + incomplete 'assistant' token (e.g. ' Ass') from streaming."""
    stripped = text.rstrip()
    m = re.search(r"(\s+)(\S+)$", stripped)
    if not m:
        return text
    last = m.group(2).lower().rstrip(":")
    assistant = "assistant"
    if assistant.startswith(last) and 0 < len(last) < len(assistant):
        return stripped[: m.start(1)].rstrip()
    if last == assistant or last == f"{assistant}:":
        return stripped[: m.start(1)].rstrip()
    return text


def _sanitize_completion_text(text: str) -> str:
    # Keep ``` fences so clients can render code blocks; strip outer whitespace only.
    cleaned = strip_empty_reasoning(text).strip()
    lowered = cleaned.lower().lstrip()
    cleaned = cleaned.lstrip()

    # Drop leading roleplay prefix if model starts with "assistant:" etc.
    changed = True
    while changed:
        changed = False
        for prefix in ("assistant:", "user:", "system:"):
            if lowered.startswith(prefix):
                cleaned = cleaned[len(prefix):].lstrip()
                lowered = cleaned.lower()
                changed = True
                break

    m = _ROLEPLAY_SPOOF_RE.search(cleaned)
    if m:
        cleaned = cleaned[: m.start()].rstrip()

    # Qwen / ChatML: degenerate generations repeat the next-turn header; cut before it.
    sp = cleaned.find("<|im_start|>")
    if sp != -1:
        cleaned = cleaned[:sp].rstrip()
    eos_vis = "<|im_end|>"
    while cleaned.endswith(eos_vis):
        cleaned = cleaned[: -len(eos_vis)].rstrip()

    return _strip_trailing_role_prefix_fragment(cleaned)


@app.exception_handler(HTTPException)
async def mlxserve_http_exception_handler(request: Request, exc: HTTPException):
    if not request.url.path.startswith("/v1/"):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    err_type = (
        "invalid_request_error"
        if exc.status_code
        in (400, 401, 403, 404, 405, 409, 413, 415, 422, 429)
        else "server_error"
    )
    msg = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": msg, "type": err_type, "code": None}},
    )


@app.exception_handler(RequestValidationError)
async def mlxserve_validation_handler(request: Request, exc: RequestValidationError):
    if not request.url.path.startswith("/v1/"):
        return await request_validation_exception_handler(request, exc)
    errs = exc.errors()
    if errs:
        e0 = errs[0]
        loc = ".".join(str(x) for x in e0.get("loc", ()))
        msg = f"{e0.get('msg', 'Invalid request')}" + (f" ({loc})" if loc else "")
    else:
        msg = "Invalid request"
    return JSONResponse(
        status_code=422,
        content={"error": {"message": msg, "type": "invalid_request_error", "code": None}},
    )


@app.middleware("http")
async def security_and_metrics_middleware(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    now = time.time()
    auth = request.headers.get("authorization") or ""
    if auth.startswith("Bearer "):
        raw = auth[7:].strip()
        key = "bearer:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    else:
        key = request.client.host if request.client else "anonymous"
    bucket = _rate_bucket[key]
    window = 60.0
    while bucket and bucket[0] < now - window:
        bucket.popleft()
    if len(bucket) >= settings.rate_limit_per_minute:
        metrics.rate_limited_total += 1
        metrics.errors_total += 1
        logger.warning("rate_limited path=%s", request.url.path)
        return PlainTextResponse("Rate limit exceeded", status_code=429)
    bucket.append(now)

    metrics.requests_total += 1
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    metrics.observe_latency(duration)
    metrics.memory_zone = guardian.classify()
    snap = guardian.snapshot()
    metrics.memory_used_gb = snap.used_gb
    metrics.swap_used_gb = snap.swap_used_gb
    metrics.macos_memory_pressure = (
        0 if snap.pressure == "normal" else 1 if snap.pressure == "warning" else 2
    )
    if response.status_code >= 400:
        metrics.errors_total += 1
    logger.info(
        "request id=%s path=%s status=%s latency_s=%.4f",
        getattr(request.state, "request_id", "-"),
        request.url.path,
        response.status_code,
        duration,
    )

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Request-ID"] = getattr(request.state, "request_id", "")
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    mid = engine.loaded_model
    return HealthResponse(
        memory_zone=guardian.classify(),
        model_loaded=mid is not None,
        loaded_model_id=mid,
    )


@app.get("/", response_class=HTMLResponse)
def chat_ui() -> HTMLResponse:
    return HTMLResponse(_chat_ui_path.read_text(encoding="utf-8"))


@app.get("/metrics")
def get_metrics() -> PlainTextResponse:
    body = metrics.render_prometheus(
        label_chat_zone=settings.metrics_label_chat_by_zone,
        label_chat_model=settings.metrics_label_chat_by_model,
    )
    return PlainTextResponse(body, media_type="text/plain; version=0.0.4")


@app.get("/v1/models", response_model=ModelsResponse, dependencies=[Depends(require_api_key)])
def list_models() -> ModelsResponse:
    model_id = engine.loaded_model or settings.default_model
    return ModelsResponse(
        data=[ModelItem(id=display_hf_model_id(model_id), context_length=model_context_length(model_id))]
    )


@app.get("/v1/models/local", response_model=LocalModelsResponse, dependencies=[Depends(require_api_key)])
def list_local_models() -> LocalModelsResponse:
    models = [
        LocalModelItem(
            id=m.id,
            source=m.source,
            local_path=m.local_path,
            pulled_at=m.pulled_at,
            size_bytes=m.size_bytes,
            quantization=m.quantization,
        )
        for m in model_manager.list_local()
    ]
    return LocalModelsResponse(data=models)


@app.post("/v1/models/pull", response_model=LocalModelItem, dependencies=[Depends(require_api_key)])
def pull_model(req: PullModelRequest) -> LocalModelItem:
    model = model_manager.pull(req.model)
    return LocalModelItem(
        id=model.id,
        source=model.source,
        local_path=model.local_path,
        pulled_at=model.pulled_at,
        size_bytes=model.size_bytes,
        quantization=model.quantization,
    )


@app.delete("/v1/models/{model_alias}", dependencies=[Depends(require_api_key)])
def remove_model(model_alias: str) -> dict:
    removed = model_manager.remove(model_alias)
    if not removed:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"deleted": True, "id": model_alias}


@app.get("/v1/models/recommended", response_model=RecommendedModelsResponse, dependencies=[Depends(require_api_key)])
def list_recommended_models() -> RecommendedModelsResponse:
    ram_gb = machine_ram_gb()
    data = [
        RecommendedModelItem(
            id=model.id,
            label=model.label,
            ram_min_gb=model.ram_min_gb,
            ram_max_gb=model.ram_max_gb,
            context=model.context,
            notes=model.notes,
        )
        for model in recommended_for_machine(ram_gb)
    ]
    return RecommendedModelsResponse(machine_ram_gb=ram_gb, data=data)


async def _admit(estimated_gb: float) -> tuple[str, str]:
    """Memory zone for a request. Before refusing (red): free the MLX buffer cache, then half of the
    prompt cache, then wait for other processes to release memory (agents pause between turns to run tools)."""
    zone, reason = guardian.classify_detail(estimated_request_gb=estimated_gb)
    for drop_prompt_cache in (False, True):
        if zone != "red":
            return zone, reason
        await engine.backend.release_memory(drop_prompt_cache=drop_prompt_cache)
        zone, reason = guardian.classify_detail(estimated_request_gb=estimated_gb)
    deadline = time.monotonic() + settings.memory_wait_seconds
    while zone == "red" and time.monotonic() < deadline:
        await asyncio.sleep(1.0)
        zone, reason = guardian.classify_detail(estimated_request_gb=estimated_gb)
    return zone, reason


def _loaded_kv_gb(model: str, seq_len: int) -> float | None:
    """Real KV size for the already-loaded model (None: fall back to the generic bound)."""
    estimate = getattr(engine.backend, "kv_gb_estimate", None)
    return estimate(seq_len) if estimate and engine.loaded_model == model else None


def _cached_tokens_details() -> dict[str, int] | None:
    n = getattr(engine.backend, "last_cached_tokens", None)
    return None if n is None else {"cached_tokens": n}


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(req: ChatCompletionRequest):
    metrics.chat_requests_total += 1
    chat_started = time.time()
    model = req.model or settings.default_model
    msg_dicts = template_messages(req.messages)
    tools = req.tools if req.tools and req.tool_choice != "none" else None
    prompt_est = heuristic_prompt_tokens(msg_dicts)
    estimated_gb = admission_memory_gb(
        prompt_est,
        req.max_tokens,
        settings.memory_admission_tokens_per_gb,
        settings.memory_admission_cap_gb,
        kv_enabled=settings.memory_admission_kv_enabled,
        kv_max_gb=settings.memory_admission_kv_max_gb,
        kv_layers=settings.memory_admission_kv_layers,
        kv_heads=settings.memory_admission_kv_heads,
        kv_head_dim=settings.memory_admission_kv_head_dim,
        kv_bytes_per_element=settings.memory_admission_kv_bytes_per_element,
        kv_gb=_loaded_kv_gb(model, prompt_est + req.max_tokens),
    )
    zone, deny_reason = await _admit(estimated_gb)
    if zone == "red":
        metrics.observe_memory_chat_denied(deny_reason)
        raise HTTPException(status_code=503, detail="Server memory pressure too high")

    if settings.idle_unload_enabled and guardian.should_unload_idle(engine.last_used_ts):
        engine.unload_model()

    context_length = model_context_length(model)
    prompt_tokens_in, _ = await engine.prompt_and_completion_token_counts(model, msg_dicts, "", tools)
    if prompt_tokens_in > context_length:
        # Fail fast (agents compact on this) instead of grinding through a prefill that cannot fit.
        raise HTTPException(
            status_code=400,
            detail=f"context_length_exceeded: this model's maximum context length is {context_length} tokens, "
            f"but the prompt has {prompt_tokens_in} tokens.",
        )

    stops = normalize_stop_sequences(req.stop)
    include_usage = bool(req.stream_options and req.stream_options.include_usage)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    mlabel = _metrics_model_label(model)

    def _chunk(delta: dict) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    if req.stream:
        async def event_stream():
            full_text = ""
            emitted_len = 0
            reasoning = EmptyReasoningFilter()
            tool_fmt = None  # resolved after the first chunk (model is loaded by then)
            tool_at: int | None = None  # index of the first tool-call marker; no content past it

            try:
                # SSE comment: flushes the stream immediately so clients do not sit on 0 bytes
                # while the model loads (can take minutes for large MLX weights).
                yield ": mlxserve\n\n"
                # Stream **raw** completion deltas. `_sanitize_completion_text` can shorten the
                # string when it finds roleplay / ChatML markers; applying it per chunk made
                # `emitted_len` track a shrinking prefix and re-slice past deltas → garbled UI
                # (“Osal”, “OBien”, repeated fragments). Sanitize only for metrics / token counts.
                async for chunk in engine.stream_text(
                    model=model,
                    messages=msg_dicts,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    stop_sequences=stops or None,
                    tools=tools,
                ):
                    full_text += chunk
                    if tools and tool_fmt is None:
                        tool_fmt = engine.backend.tool_call_format() or False
                    limit = len(full_text)
                    if tool_fmt:
                        # Content stops at the first tool-call marker; hold back a partial marker.
                        if tool_at is None and (i := full_text.find(tool_fmt[0], emitted_len)) != -1:
                            tool_at = i
                        limit = tool_at if tool_at is not None else limit - partial_marker_len(full_text, tool_fmt[0])
                    if limit > emitted_len:
                        # Empty reasoning blocks are dropped wherever they appear (held back while partial).
                        if out := reasoning.feed(full_text[emitted_len:limit]):
                            yield _chunk({"content": out})
                        emitted_len = limit

                tool_calls: list[dict] = []
                if tool_fmt:
                    _, tool_calls = extract_tool_calls(full_text, tool_fmt, tools)
                    if tool_at is None and emitted_len < len(full_text):
                        # Held-back tail was not a tool-call marker after all.
                        if out := reasoning.feed(full_text[emitted_len:]):
                            yield _chunk({"content": out})
                if out := reasoning.flush():
                    yield _chunk({"content": out})
                if tool_calls:
                    yield _chunk({"tool_calls": [dict(tc, index=n) for n, tc in enumerate(tool_calls)]})

                final_text = full_text if tool_calls else _sanitize_completion_text(full_text)
                pt_done, ct_done = await engine.prompt_and_completion_token_counts(
                    model, msg_dicts, final_text, tools
                )
                finish_reason = "tool_calls" if tool_calls else "length" if ct_done >= req.max_tokens else "stop"
                final_chunk: dict = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                }
                if include_usage:
                    final_chunk["usage"] = {
                        "prompt_tokens": max(0, pt_done),
                        "completion_tokens": max(0, ct_done),
                        "total_tokens": max(0, pt_done) + max(0, ct_done),
                        "prompt_tokens_details": _cached_tokens_details(),
                    }
                yield f"data: {json.dumps(final_chunk)}\n\n"
            finally:
                metrics.observe_chat_completion(
                    zone,
                    mlabel,
                    label_zone=settings.metrics_label_chat_by_zone,
                    label_model=settings.metrics_label_chat_by_model,
                )
                try:
                    ft = _sanitize_completion_text(full_text)
                    _, ctk = await engine.prompt_and_completion_token_counts(model, msg_dicts, ft, tools)
                    metrics.generated_tokens_total += max(0, ctk)
                    metrics.chat_generation_tps = max(0, ctk) / max(1e-6, (time.time() - chat_started))
                except Exception:
                    pass
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    text = await engine.generate_text(
        model=model,
        messages=msg_dicts,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop_sequences=stops or None,
        tools=tools,
    )
    raw_text = text
    tool_calls: list[dict] = []
    if tools and (tool_fmt := engine.backend.tool_call_format()):
        text, tool_calls = extract_tool_calls(text, tool_fmt, tools)
    text = _sanitize_completion_text(text)
    prompt_tokens, completion_tokens = await engine.prompt_and_completion_token_counts(
        model, msg_dicts, raw_text if tool_calls else text, tools
    )
    usage = Usage(
        prompt_tokens=max(0, prompt_tokens),
        completion_tokens=max(0, completion_tokens),
        total_tokens=max(0, prompt_tokens) + max(0, completion_tokens),
        prompt_tokens_details=_cached_tokens_details(),
    )
    metrics.generated_tokens_total += usage.completion_tokens
    metrics.chat_generation_tps = usage.completion_tokens / max(1e-6, (time.time() - chat_started))
    metrics.observe_chat_completion(
        zone,
        mlabel,
        label_zone=settings.metrics_label_chat_by_zone,
        label_model=settings.metrics_label_chat_by_model,
    )
    finish_reason = "tool_calls" if tool_calls else "length" if completion_tokens >= req.max_tokens else "stop"
    message = ChoiceMessage(content=text or None, tool_calls=tool_calls) if tool_calls else ChoiceMessage(content=text)
    response = ChatCompletionResponse(
        id=completion_id,
        model=model,
        choices=[ChatChoice(message=message, finish_reason=finish_reason)],
        usage=usage,
    )
    return response
