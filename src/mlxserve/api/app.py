import json
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from mlxserve.api.deps import engine, guardian, require_api_key
from mlxserve.config import settings
from mlxserve.models.schemas import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChoiceMessage,
    HealthResponse,
    ModelItem,
    ModelsResponse,
    RecommendedModelItem,
    RecommendedModelsResponse,
    Usage,
)
from mlxserve.models.catalog import machine_ram_gb, recommended_for_machine

app = FastAPI(title="MLXServe", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(memory_zone=guardian.classify())


@app.get("/v1/models", response_model=ModelsResponse, dependencies=[Depends(require_api_key)])
def list_models() -> ModelsResponse:
    model_id = engine.loaded_model or settings.default_model
    return ModelsResponse(data=[ModelItem(id=model_id)])


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


def _prompt_from_messages(req: ChatCompletionRequest) -> str:
    return "\n".join([f"{m.role}: {m.content}" for m in req.messages])


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(req: ChatCompletionRequest):
    model = req.model or settings.default_model
    estimated_gb = min(req.max_tokens / 10000.0, 1.0)
    zone = guardian.classify(estimated_request_gb=estimated_gb)
    if zone == "red":
        raise HTTPException(status_code=503, detail="Server memory pressure too high")

    if guardian.should_unload_idle(engine.last_used_ts):
        engine.unload_model()

    prompt = _prompt_from_messages(req)

    if req.stream:
        async def event_stream():
            async for chunk in engine.stream_text(model=model, prompt=prompt, max_tokens=req.max_tokens):
                payload = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    text = await engine.generate_text(model=model, prompt=prompt, max_tokens=req.max_tokens)
    usage = Usage(
        prompt_tokens=max(1, len(prompt.split())),
        completion_tokens=max(1, len(text.split())),
        total_tokens=max(2, len(prompt.split()) + len(text.split())),
    )
    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        model=model,
        choices=[ChatChoice(message=ChoiceMessage(content=text))],
        usage=usage,
    )
    return response
