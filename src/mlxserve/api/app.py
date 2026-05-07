import json
import logging
import re
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from mlxserve.api.deps import engine, guardian, metrics, model_manager, require_api_key
from mlxserve.config import settings
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
from mlxserve.models.catalog import machine_ram_gb, recommended_for_machine

app = FastAPI(title="MLXServe", version="0.1.0")
logger = logging.getLogger("mlxserve.api")
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
# Only use markers that match chat transcripts, not substrings common in code
# (e.g. "\nuser" matches "\nusername" and used to truncate Python mid-file).
_ROLEPLAY_MARKERS = [
    "assistant:",
    "user:",
    "system:",
    "\nuser:",
    "\nassistant:",
    "\nsystem:",
    " user:",
    " assistant:",
    " system:",
]

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
    cleaned = text.strip()
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

    lowered = cleaned.lower()
    cut = len(cleaned)
    for marker in _ROLEPLAY_MARKERS:
        idx = lowered.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    cleaned = cleaned[:cut].strip()

    return _strip_trailing_role_prefix_fragment(cleaned)


@app.middleware("http")
async def security_and_metrics_middleware(request: Request, call_next):
    now = time.time()
    key = request.headers.get("authorization") or request.client.host or "anonymous"
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
    if response.status_code >= 400:
        metrics.errors_total += 1
    logger.info("request path=%s status=%s latency_s=%.4f", request.url.path, response.status_code, duration)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(memory_zone=guardian.classify())


@app.get("/", response_class=HTMLResponse)
def chat_ui() -> HTMLResponse:
    return HTMLResponse(_chat_ui_path.read_text(encoding="utf-8"))


@app.get("/metrics")
def get_metrics() -> PlainTextResponse:
    return PlainTextResponse(metrics.render_prometheus(), media_type="text/plain; version=0.0.4")


@app.get("/v1/models", response_model=ModelsResponse, dependencies=[Depends(require_api_key)])
def list_models() -> ModelsResponse:
    model_id = engine.loaded_model or settings.default_model
    return ModelsResponse(data=[ModelItem(id=display_hf_model_id(model_id))])


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


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(req: ChatCompletionRequest):
    metrics.chat_requests_total += 1
    chat_started = time.time()
    model = req.model or settings.default_model
    estimated_gb = min(req.max_tokens / 10000.0, 1.0)
    zone = guardian.classify(estimated_request_gb=estimated_gb)
    if zone == "red":
        raise HTTPException(status_code=503, detail="Server memory pressure too high")

    if guardian.should_unload_idle(engine.last_used_ts):
        engine.unload_model()

    msg_dicts = [{"role": m.role, "content": m.content} for m in req.messages]

    if req.stream:
        async def event_stream():
            full_text = ""
            emitted_len = 0

            def _emit_delta(sanitized: str) -> str:
                nonlocal emitted_len
                delta = sanitized[emitted_len:]
                emitted_len = len(sanitized)
                return delta

            async for chunk in engine.stream_text(
                model=model,
                messages=msg_dicts,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            ):
                full_text += chunk
                # Hold the first bytes so partial role labels never reach the client.
                if len(full_text) < 24:
                    continue
                sanitized = _sanitize_completion_text(full_text)
                delta = _emit_delta(sanitized)
                if not delta:
                    continue
                payload = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(payload)}\n\n"

            if full_text:
                sanitized = _sanitize_completion_text(full_text)
                delta = sanitized[emitted_len:]
                if delta:
                    emitted_len = len(sanitized)
                    payload = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    text = await engine.generate_text(
        model=model,
        messages=msg_dicts,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    text = _sanitize_completion_text(text)
    prompt_for_estimate = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
    usage = Usage(
        prompt_tokens=max(1, len(prompt_for_estimate.split())),
        completion_tokens=max(1, len(text.split())),
        total_tokens=max(2, len(prompt_for_estimate.split()) + len(text.split())),
    )
    metrics.generated_tokens_total += usage.completion_tokens
    metrics.chat_generation_tps = usage.completion_tokens / max(1e-6, (time.time() - chat_started))
    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        model=model,
        choices=[ChatChoice(message=ChoiceMessage(content=text))],
        usage=usage,
    )
    return response
