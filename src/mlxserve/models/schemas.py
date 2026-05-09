from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    memory_zone: Literal["green", "yellow", "red"]


class ModelItem(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "mlxserve"


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelItem]


class RecommendedModelItem(BaseModel):
    id: str
    label: str
    ram_min_gb: float
    ram_max_gb: float
    context: str
    notes: str


class RecommendedModelsResponse(BaseModel):
    machine_ram_gb: float
    data: list[RecommendedModelItem]


class LocalModelItem(BaseModel):
    id: str
    source: str
    local_path: str
    pulled_at: str
    size_bytes: int
    quantization: str


class LocalModelsResponse(BaseModel):
    data: list[LocalModelItem]


class PullModelRequest(BaseModel):
    model: str = Field(min_length=3)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=256, gt=0, le=4096)
    stream: bool = False
    stop: str | list[str] | None = None
    stream_options: StreamOptions | None = None


class ChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatChoice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: Literal["stop", "length"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    model: str
    choices: list[ChatChoice]
    usage: Usage
