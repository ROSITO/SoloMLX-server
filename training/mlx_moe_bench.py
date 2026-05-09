"""Benchmark mlx-lm models on Apple Silicon: peak GPU memory, wall time, MoE metadata.

Native MoE in MLX is already implemented in mlx-lm (e.g. Mixtral `MixtralSparseMoeBlock`).
This script measures what matters for a cost-first story: time, tokens, and `mx.get_peak_memory`.

Examples::

  # MoE (Mixtral 8x7B 4-bit) — premier run = gros téléchargement
  .venv/bin/python -m training.mlx_moe_bench \\
    --model-id mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit \\
    --max-tokens 32

  # Dense baseline sur la même machine (comparaison RAM / s)
  .venv/bin/python -m training.mlx_moe_bench \\
    --model-id mlx-community/Mistral-7B-Instruct-v0.3-4bit \\
    --max-tokens 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlxserve.runtime.moe_offload import apply_moe_expert_offload, MoEOffloadReport


def _model_summary(model: Any) -> dict[str, Any]:
    args = getattr(model, "args", None)
    if args is None:
        return {"kind": "unknown"}
    out: dict[str, Any] = {
        "model_type": getattr(args, "model_type", ""),
    }
    n_loc = getattr(args, "num_local_experts", None)
    if n_loc is None:
        n_loc = getattr(args, "num_experts", None)
    n_tok = getattr(args, "num_experts_per_tok", None)
    if n_loc:
        out["num_local_experts"] = int(n_loc)
        out["num_experts_per_tok"] = int(n_tok) if n_tok is not None else None
        if n_tok and n_loc:
            out["expert_activation_ratio"] = float(n_tok) / float(n_loc)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="mlx-lm MoE/dense micro-benchmark (MLX memory + time).")
    p.add_argument(
        "--model-id",
        default="mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
        help="Hugging Face repo id for an mlx-lm compatible model.",
    )
    p.add_argument("--prompt", default="Explique en 2 phrases pourquoi un MoE réduit le coût par token.")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--out", default="bench/moe_training/mlx_moe_bench_report.json")
    p.add_argument(
        "--ram-budget-gib",
        type=float,
        default=None,
        help="If set, fail with exit code 2 when peak MLX memory (worst of load/gen/active) exceeds this GiB.",
    )
    p.add_argument(
        "--moe-resident-experts",
        type=int,
        default=0,
        help="If >0 and model is MoE, keep only this number of experts resident per layer (MLX runtime offload).",
    )
    p.add_argument(
        "--moe-resident-strategy",
        choices=("l2", "first"),
        default="l2",
        help="Expert selection strategy when applying runtime offload.",
    )
    p.add_argument(
        "--moe-single-expert-fastpath",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, replace MoE block forward with a single-expert fast path when experts_after=1.",
    )
    p.add_argument("--prefill-step-size", type=int, default=2048, help="Passed to mlx_lm.generate (prefill chunking).")
    p.add_argument("--kv-bits", type=int, default=None, help="Optional KV quant bits for mlx_lm.generate (None = default).")
    p.add_argument("--kv-group-size", type=int, default=64, help="KV group size when kv-bits is set.")
    p.add_argument("--quantized-kv-start", type=int, default=0, help="Layer index to start KV quant when kv-bits is set.")
    args = p.parse_args()

    try:
        from mlx_lm import generate as mlx_generate
        from mlx_lm import load as mlx_load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as e:
        raise SystemExit("mlx-lm is required. Install: pip install -e '.[mlx]' or pip install mlx-lm") from e

    mx.reset_peak_memory()
    t_load0 = time.perf_counter()
    model, tokenizer = mlx_load(args.model_id)
    offload_report: MoEOffloadReport | None = apply_moe_expert_offload(
        model,
        keep_experts=args.moe_resident_experts,
        strategy=args.moe_resident_strategy,
        enable_single_expert_fastpath=args.moe_single_expert_fastpath,
    )
    t_load = time.perf_counter() - t_load0
    peak_after_load = int(mx.get_peak_memory())

    mx.reset_peak_memory()
    t_gen0 = time.perf_counter()
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p)
    gen_kw: dict[str, Any] = {
        "verbose": False,
        "sampler": sampler,
        "prefill_step_size": args.prefill_step_size,
    }
    if args.kv_bits is not None:
        gen_kw["kv_bits"] = int(args.kv_bits)
        gen_kw["kv_group_size"] = int(args.kv_group_size)
        gen_kw["quantized_kv_start"] = int(args.quantized_kv_start)
    text = mlx_generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        **gen_kw,
    )
    t_gen = time.perf_counter() - t_gen0
    peak_after_generate = int(mx.get_peak_memory())
    active_after = int(mx.get_active_memory())

    summary = _model_summary(model)
    g_load = round(peak_after_load / (1024**3), 3)
    g_gen = round(peak_after_generate / (1024**3), 3)
    g_active = round(active_after / (1024**3), 3)
    worst_gib = max(g_load, g_gen, g_active)

    report = {
        "model_id": args.model_id,
        "model_summary": summary,
        "prefill_step_size": args.prefill_step_size,
        "kv_bits": args.kv_bits,
        "kv_group_size": args.kv_group_size if args.kv_bits is not None else None,
        "quantized_kv_start": args.quantized_kv_start if args.kv_bits is not None else None,
        "prompt_len_chars": len(args.prompt),
        "max_tokens": args.max_tokens,
        "output_len_chars": len(text),
        "load_wall_s": round(t_load, 4),
        "generate_wall_s": round(t_gen, 4),
        "tokens_per_s": round(args.max_tokens / t_gen, 4) if t_gen > 0 else None,
        "peak_memory_bytes_after_load": peak_after_load,
        "peak_memory_bytes_generate_phase": peak_after_generate,
        "active_memory_bytes_end": active_after,
        "peak_memory_gib_after_load": g_load,
        "peak_memory_gib_generate_phase": g_gen,
        "active_memory_gib_end": g_active,
        "peak_memory_gib_worst": worst_gib,
    }
    if offload_report is not None:
        report["moe_offload"] = {
            "keep_experts": offload_report.keep_experts,
            "strategy": offload_report.strategy,
            "layers_touched": offload_report.layers_touched,
            "experts_before": offload_report.experts_before,
            "experts_after": offload_report.experts_after,
            "single_expert_fastpath": offload_report.single_expert_fastpath,
        }

    if args.ram_budget_gib is not None:
        report["ram_budget_gib"] = float(args.ram_budget_gib)
        report["within_ram_budget"] = worst_gib <= float(args.ram_budget_gib)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwritten: {out_path}")
    if args.ram_budget_gib is not None and not report.get("within_ram_budget", True):
        print(
            f"\nRAM budget exceeded: peak_memory_gib_worst={worst_gib} > ram_budget_gib={args.ram_budget_gib}",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
