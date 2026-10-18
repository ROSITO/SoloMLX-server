import json
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass(frozen=True)
class RecommendedModel:
    id: str
    label: str
    ram_min_gb: float
    ram_max_gb: float
    context: str
    notes: str


CATALOG: list[RecommendedModel] = [
    RecommendedModel(
        id="mlx-community/gemma-4-26B-A4B-it-qat-4bit",
        label="Gemma 4 26B-A4B IT QAT Q4 (MoE)",
        ram_min_gb=32,
        ram_max_gb=64,
        context="Best speed/quality on 32-64 GB: chat, code, long context",
        notes="~14.5 GiB weights. 36 GB Mac (thinking off): 41-68 tok/s decode, ~430 tok/s prefill, peak 16 GB at 17k-token prompt.",
    ),
    RecommendedModel(
        id="mlx-community/Qwen3.8-27B-4bit",
        label="Qwen3.8 27B Q4",
        ram_min_gb=32,
        ram_max_gb=64,
        context="Dense 27B: careful reasoning, slower decode",
        notes="~15 GiB weights. 36 GB Mac (thinking off): ~7.5 tok/s decode, ~75 tok/s prefill, peak 18.9 GB at 17k-token prompt.",
    ),
    RecommendedModel(
        id="mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
        label="Qwen3 30B-A3B Instruct 2507 Q4 (MoE)",
        ram_min_gb=32,
        ram_max_gb=64,
        context="Strongest general chat/code; 3B active params so decode stays fast",
        notes="~16 GiB weights. No thinking mode.",
    ),
    RecommendedModel(
        id="mlx-community/Qwen3-14B-4bit",
        label="Qwen3 14B Q4",
        ram_min_gb=24,
        ram_max_gb=48,
        context="High quality chat/code/reasoning",
        notes="~7.7 GiB weights. 36 GB Mac: ~12.8 tok/s, peak ~10.7 GiB at 10k-token prompt. Append /no_think to skip reasoning.",
    ),
    RecommendedModel(
        id="mlx-community/Qwen3-8B-4bit",
        label="Qwen3 8B Q4",
        ram_min_gb=16,
        ram_max_gb=24,
        context="Higher quality chat/code",
        notes="~4.3 GiB weights. Successor to Qwen2.5 7B. Append /no_think to skip reasoning.",
    ),
    RecommendedModel(
        id="mlx-community/Qwen3-4B-Instruct-2507-4bit",
        label="Qwen3 4B Instruct 2507 Q4",
        ram_min_gb=8,
        ram_max_gb=16,
        context="General purpose assistant",
        notes="~2.1 GiB weights. No thinking mode.",
    ),
    RecommendedModel(
        id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        label="Qwen2.5 7B Instruct Q4",
        ram_min_gb=16,
        ram_max_gb=24,
        context="Higher quality chat/code",
        notes="Default server model; use conservative max_tokens and low concurrency on 16 GB.",
    ),
    RecommendedModel(
        id="mlx-community/Qwen2.5-3B-Instruct-4bit",
        label="Qwen2.5 3B Instruct Q4",
        ram_min_gb=8,
        ram_max_gb=16,
        context="General purpose assistant",
        notes="Best stability baseline for 16 GB machines.",
    ),
    RecommendedModel(
        id="mlx-community/Qwen2.5-Coder-32B-Instruct-3bit",
        label="Qwen2.5 Coder 32B Instruct Q3",
        ram_min_gb=16,
        ram_max_gb=36,
        context="Large code / reasoning (32B class)",
        notes="Gate 16 GiB: mlx_moe_bench peak ~13.5 GiB (24 tok) with KV4, quantized_kv_start=32, prefill_step_size=512. Code bias.",
    ),
    RecommendedModel(
        id="mlx-community/Llama-3.2-3B-Instruct-4bit",
        label="Llama 3.2 3B Instruct Q4",
        ram_min_gb=8,
        ram_max_gb=16,
        context="Balanced reasoning for local agents",
        notes="Good fallback when Qwen family is unavailable.",
    ),
    RecommendedModel(
        id="mlx-community/SmolLM2-1.7B-Instruct-4bit",
        label="SmolLM2 1.7B Instruct Q4",
        ram_min_gb=8,
        ram_max_gb=12,
        context="Very low memory mode",
        notes="Lowest memory footprint, useful under pressure.",
    ),
]


def model_context_length(model_id: str) -> int:
    """Context window to advertise: min(settings.context_tokens, the model's max_position_embeddings)."""
    from mlxserve.config import settings

    cfg_path = Path(model_id) / "config.json"
    if not cfg_path.exists():
        try:
            from huggingface_hub import try_to_load_from_cache

            hit = try_to_load_from_cache(model_id, "config.json")
            cfg_path = Path(hit) if isinstance(hit, str) else None
        except Exception:
            cfg_path = None
    try:
        cfg = json.loads(cfg_path.read_text()) if cfg_path else {}
    except Exception:
        cfg = {}
    native = cfg.get("max_position_embeddings") or cfg.get("text_config", {}).get("max_position_embeddings")
    return min(settings.context_tokens, native) if native else settings.context_tokens


def machine_ram_gb() -> float:
    return psutil.virtual_memory().total / (1024**3)


def recommended_for_machine(total_ram_gb: float | None = None) -> list[RecommendedModel]:
    ram = total_ram_gb if total_ram_gb is not None else machine_ram_gb()
    result = [m for m in CATALOG if ram >= m.ram_min_gb]
    # Prioritize models calibrated for this RAM band.
    preferred = [m for m in result if ram <= m.ram_max_gb]
    return preferred or result
