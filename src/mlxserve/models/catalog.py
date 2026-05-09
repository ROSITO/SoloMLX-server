from dataclasses import dataclass

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


def machine_ram_gb() -> float:
    return psutil.virtual_memory().total / (1024**3)


def recommended_for_machine(total_ram_gb: float | None = None) -> list[RecommendedModel]:
    ram = total_ram_gb if total_ram_gb is not None else machine_ram_gb()
    result = [m for m in CATALOG if ram >= m.ram_min_gb]
    # Prioritize models calibrated for this RAM band.
    preferred = [m for m in result if ram <= m.ram_max_gb]
    return preferred or result
