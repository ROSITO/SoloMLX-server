from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str = ""
    max_memory_gb: float = 14.0
    hard_memory_gb: float = 15.0
    idle_unload_minutes: int = 15
    default_model: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    runtime_backend: str = "auto"
    prefill_step_size: int = 1024
    kv_bits: int = 4
    kv_group_size: int = 64
    quantized_kv_start: int = 32
    cors_allow_origins: str = "*"
    rate_limit_per_minute: int = 120

    model_config = SettingsConfigDict(env_prefix="MLXSERVE_", extra="ignore")


settings = Settings()
