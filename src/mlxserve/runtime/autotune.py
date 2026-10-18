import json
import time
from dataclasses import asdict, dataclass


@dataclass
class TuneResult:
    prefill_step_size: int
    kv_bits: int | None
    kv_group_size: int
    quantized_kv_start: int
    generation_tps: float
    prompt_tps: float
    ttft_s: float
    wall_s: float
    peak_memory_gb: float


def _run_once(model, tokenizer, prompt: str, max_tokens: int, *, prefill_step_size: int, kv_bits: int | None, kv_group_size: int, quantized_kv_start: int):
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.2, top_p=0.95)
    mx.reset_peak_memory()
    start = time.time()
    first_token_ts = None
    last = None
    for r in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        prefill_step_size=prefill_step_size,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
    ):
        if first_token_ts is None and r.text:
            first_token_ts = time.time()
        last = r
    end = time.time()
    if last is None:
        raise RuntimeError("No generation response received during autotune.")
    return TuneResult(
        prefill_step_size=prefill_step_size,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
        generation_tps=float(last.generation_tps),
        prompt_tps=float(last.prompt_tps),
        ttft_s=float((first_token_ts - start) if first_token_ts else (end - start)),
        wall_s=float(end - start),
        peak_memory_gb=float(mx.get_peak_memory() / 1e9),
    )


def run_autotune(model_id: str, prompt: str, max_tokens: int = 96) -> dict:
    from mlx_lm import load

    model, tokenizer = load(model_id)

    # quick warmup
    _ = _run_once(
        model,
        tokenizer,
        "warmup",
        16,
        prefill_step_size=2048,
        kv_bits=None,
        kv_group_size=64,
        quantized_kv_start=32,
    )

    # fp16 KV is the server default; quantized KV is opt-in (can corrupt Qwen2.5 instruct).
    # 512 keeps the q4 prefill peak down: each chunk dequantizes the cache for attention.
    candidates = [
        (512, None, 64, 0),
        (1024, None, 64, 0),
        (2048, None, 64, 0),
        (512, 4, 64, 32),
        (1024, 4, 64, 32),
        (2048, 4, 64, 32),
    ]

    runs: list[TuneResult] = []
    for prefill, kv_bits, kv_group, qstart in candidates:
        runs.append(
            _run_once(
                model,
                tokenizer,
                prompt,
                max_tokens,
                prefill_step_size=prefill,
                kv_bits=kv_bits,
                kv_group_size=kv_group,
                quantized_kv_start=qstart,
            )
        )

    # prefer fp16 KV, then throughput, then lower TTFT; quantized runs stay in all_runs for opt-in
    best = max(runs, key=lambda r: (r.kv_bits is None, r.generation_tps, -r.ttft_s))
    payload = {
        "model": model_id,
        "max_tokens": max_tokens,
        "best": asdict(best),
        "all_runs": [asdict(r) for r in runs],
        "recommended_env": {
            "MLXSERVE_PREFILL_STEP_SIZE": str(best.prefill_step_size),
            "MLXSERVE_KV_BITS": str(best.kv_bits or "none"),
            "MLXSERVE_KV_GROUP_SIZE": str(best.kv_group_size),
            "MLXSERVE_QUANTIZED_KV_START": str(best.quantized_kv_start),
        },
    }
    return payload


def run_autotune_json(model_id: str, prompt: str, max_tokens: int = 96) -> str:
    return json.dumps(run_autotune(model_id=model_id, prompt=prompt, max_tokens=max_tokens), indent=2)
