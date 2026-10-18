from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    memory_zone: Literal["green", "yellow", "red"]
    model_loaded: bool = False
    """True when MLX (or stub) weights are in memory for ``loaded_model_id``."""
    loaded_model_id: str | None = None
    """Hub id or path of the model currently resident in RAM; ``null`` if none."""


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


class ToolCallFunction(BaseModel):
    name: str
    arguments: str
    """JSON-encoded arguments, as in the OpenAI API."""


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def _check_content(self):
        if isinstance(self.content, list):  # OpenAI content parts: keep text only
            self.content = "".join(p.get("text", "") for p in self.content if p.get("type") == "text")
        if not self.content and not (self.role == "assistant" and self.tool_calls):
            raise ValueError("content is required (assistant may send tool_calls instead)")
        return self


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
    tools: list[dict[str, Any]] | None = None
    # "none" drops tools; "auto" / "required" / a named function all behave like "auto" (not enforced).
    tool_choice: str | dict[str, Any] | None = None


class ChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None
    tool_calls: list[ToolCall] | None = None


class ChatChoice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: Literal["stop", "length", "tool_calls"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: dict[str, int] | None = None
    """OpenAI shape: ``{"cached_tokens": n}`` = prompt tokens served from the reused KV cache."""


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    model: str
    choices: list[ChatChoice]
    usage: Usage
