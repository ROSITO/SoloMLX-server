import psutil
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str = ""
    jwt_hs256_secret: str = ""
    jwt_audience: str = ""
    # None = scaled to machine RAM (14 / 15 GB on a 16 GB Mac). Compared to system-wide used RAM.
    max_memory_gb: float | None = None
    hard_memory_gb: float | None = None
    idle_unload_minutes: int = 15
    idle_unload_enabled: bool = True
    preload_default_model: bool = False
    default_model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    runtime_backend: str = "auto"
    prefill_step_size: int = 1024
    # None = full-precision KV (default). int 4 etc. can corrupt Qwen2.5 instruct on mlx-lm (garbled output).
    # Opt in with MLXSERVE_KV_BITS=4 when you accept the quality / memory tradeoff.
    kv_bits: int | None = None
    kv_group_size: int = 64
    quantized_kv_start: int = 32
    cors_allow_origins: str = "*"
    rate_limit_per_minute: int = 120
    memory_admission_tokens_per_gb: float = 4500.0
    memory_admission_cap_gb: float = 2.0
    # Optional KV-cache upper bound for pre-admission (no model load). Conservative defaults.
    memory_admission_kv_enabled: bool = True
    memory_admission_kv_max_gb: float = 4.0
    memory_admission_kv_layers: int = 40
    memory_admission_kv_heads: int = 8
    memory_admission_kv_head_dim: int = 128
    memory_admission_kv_bytes_per_element: float = 2.0
    metrics_label_chat_by_zone: bool = True
    metrics_label_chat_by_model: bool = True
    metrics_model_label_max_len: int = 64
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_num_shared_experts: int = 1
    moe_resident_experts: int = 0
    moe_resident_strategy: str = "l2"
    moe_single_expert_fastpath: bool = True
    # mlx_lm logits processor: >1.0 discourages repetition; can destabilize some quants — off by default.
    # Enable e.g. MLXSERVE_GENERATION_REPETITION_PENALTY=1.08 when needed.
    generation_repetition_penalty: float = 1.0
    generation_repetition_context_size: int = 64
    # Reasoning models (Qwen3, Qwen3.8, Gemma 4): passed as enable_thinking to the chat template.
    # Off by default: reasoning would land in the reply text. MLXSERVE_ENABLE_THINKING=true to re-enable.
    enable_thinking: bool = False
    # KV cache reused across turns (longest shared token prefix). Max GB kept between requests; 0 = off.
    prompt_cache_gb: float = 4.0
    # Requests asking for more (agent harnesses send 8k-32k) are clamped, not rejected.
    max_output_tokens: int = 8192

    model_config = SettingsConfigDict(env_prefix="MLXSERVE_", extra="ignore")

    @field_validator("kv_bits", mode="before")
    @classmethod
    def _coerce_kv_bits(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("none", "off", "false", ""):
                return None
            v = int(s)
        i = int(v)
        return None if i <= 0 else i

    @model_validator(mode="after")
    def _default_memory_limits(self):
        total_gb = psutil.virtual_memory().total / (1024**3)
        if self.max_memory_gb is None:
            self.max_memory_gb = round(total_gb * 14 / 16, 1)
        if self.hard_memory_gb is None:
            self.hard_memory_gb = round(total_gb * 15 / 16, 1)
        return self


settings = Settings()
