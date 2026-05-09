from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str = ""
    jwt_hs256_secret: str = ""
    jwt_audience: str = ""
    max_memory_gb: float = 14.0
    hard_memory_gb: float = 15.0
    idle_unload_minutes: int = 15
    idle_unload_enabled: bool = True
    default_model: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    runtime_backend: str = "auto"
    prefill_step_size: int = 1024
    kv_bits: int = 4
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

    model_config = SettingsConfigDict(env_prefix="MLXSERVE_", extra="ignore")


settings = Settings()
