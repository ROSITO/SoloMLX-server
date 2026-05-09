from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.moe_model import (
    TopKMoELlamaMLP,
    infer_num_experts_from_moe_adapter_state,
    load_moe_state,
    replace_llama_mlp_with_moe,
)


def _load_prompts(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Prompt file must be a non-empty JSON list")
    return data


def _eval_model(model, tok, prompts: list[dict], *, device: str, max_new_tokens: int) -> dict:
    losses: list[float] = []
    lat_ms: list[float] = []
    samples: list[dict] = []
    model.eval()
    with torch.no_grad():
        for item in prompts:
            text = item["prompt"]
            enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            t0 = time.perf_counter()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = float(out.loss.item())
            losses.append(loss)
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            dt = (time.perf_counter() - t0) * 1000.0
            lat_ms.append(dt)
            dec = tok.decode(gen[0], skip_special_tokens=True)
            samples.append({"id": item.get("id", ""), "loss": loss, "latency_ms": dt, "output": dec})
    return {
        "avg_loss": sum(losses) / len(losses),
        "p95_latency_ms": max(lat_ms) if len(lat_ms) < 20 else sorted(lat_ms)[int(0.95 * len(lat_ms)) - 1],
        "samples": samples,
    }


def _release_model(model: object) -> None:
    del model
    gc.collect()
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _reclaim_device_memory(device: str) -> None:
    """Best-effort free MPS/CPU memory between heavy model loads (sequential eval legs)."""
    gc.collect()
    gc.collect()
    if device == "mps" and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _run_moe_eval_branch(
    *,
    model_id: str,
    device: str,
    dtype_kw: dict,
    low_cpu_mem_usage: bool,
    layers: list[int],
    num_experts: int,
    top_k: int,
    shared_experts: int,
    offload_unused_experts: bool,
    adapter_path: str,
    adapter_state: dict | None,
    warmup_steps: int,
    tok,
    prompts: list[dict],
    max_new_tokens: int,
) -> tuple[dict, list[int], int]:
    """Load base model, replace MLP with MoE, optionally load adapter, warmup, eval, free."""
    _reclaim_device_memory(device)
    moe = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=low_cpu_mem_usage,
        **dtype_kw,
    ).to(device)
    if device == "cpu":
        moe = moe.to(dtype=torch.float32)
    moe, replaced = replace_llama_mlp_with_moe(
        moe,
        layers,
        num_experts=num_experts,
        top_k=top_k,
        shared_experts=shared_experts,
        offload_unused_experts=offload_unused_experts,
    )
    loaded_layers = 0
    if adapter_state is not None:
        loaded_layers = load_moe_state(moe, layers, adapter_state)
        adapter_state.clear()
    elif adapter_path:
        state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        loaded_layers = load_moe_state(moe, layers, state)
        state.clear()
    _warmup_moe_model(moe, steps=warmup_steps, layers=layers, device=device)
    rep = _eval_model(moe, tok, prompts, device=device, max_new_tokens=max_new_tokens)
    _release_model(moe)
    return rep, replaced, loaded_layers


def _warmup_moe_model(model, *, steps: int, layers: list[int], device: str) -> None:
    if steps <= 0:
        return
    moe_params: list[torch.nn.Parameter] = []
    for li in layers:
        mlp = model.model.layers[li].mlp
        if isinstance(mlp, TopKMoELlamaMLP):
            moe_params.extend(list(mlp.parameters()))
    if not moe_params:
        return
    vocab = int(model.config.vocab_size)
    opt = torch.optim.AdamW(moe_params, lr=1e-4)
    model.train()
    for _ in range(steps):
        input_ids = torch.randint(0, vocab, (2, 32), device=device)
        out = model(input_ids=input_ids, labels=input_ids)
        main_loss = out.loss
        bal_terms = []
        for li in layers:
            mlp = model.model.layers[li].mlp
            if not isinstance(mlp, TopKMoELlamaMLP):
                continue
            st = mlp.last_stats
            usage = torch.tensor([st["expert_usage"][f"expert_{i}"] for i in range(mlp.num_experts)], device=device)
            bal_terms.append(((usage - usage.mean()) ** 2).mean())
        balance_loss = torch.stack(bal_terms).mean() if bal_terms else torch.tensor(0.0, device=device)
        loss = main_loss + 0.05 * balance_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(moe_params, 1.0)
        opt.step()


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate dense vs MoE-bridge model on prompt set.")
    p.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--prompts", default="bench/eval_quality_v1.json")
    p.add_argument("--layers", default="2,6,10")
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--shared-experts", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--out", default="bench/moe_training/eval_ab_report.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--adapter-path", default="", help="Optional path to a saved MoE adapter state (.pt)")
    p.add_argument(
        "--adapter-path-shrunk",
        default="",
        help="Optional second adapter (fewer experts). Runs after the main MoE eval; frees memory between runs.",
    )
    p.add_argument(
        "--num-experts-shrunk",
        type=int,
        default=0,
        help="Expert count for --adapter-path-shrunk. If 0, inferred from router.weight in the checkpoint.",
    )
    p.add_argument(
        "--offload-unused-experts",
        action="store_true",
        help="Park non-routed expert weights on CPU between MoE forwards (eval/generate only; lowers MPS peak).",
    )
    p.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="torch_dtype for from_pretrained (bf16 recommended for MPS on 7B+).",
    )
    p.add_argument(
        "--low-cpu-mem-usage",
        action="store_true",
        help="low_cpu_mem_usage=True when loading checkpoints.",
    )
    args = p.parse_args()

    device = args.device
    prompts = _load_prompts(args.prompts)
    if "mistral" in args.model_id.lower():
        try:
            tok = AutoTokenizer.from_pretrained(args.model_id, fix_mistral_regex=True)
        except TypeError:
            tok = AutoTokenizer.from_pretrained(args.model_id)
    else:
        tok = AutoTokenizer.from_pretrained(args.model_id)
    dtype_kw: dict = {}
    if args.dtype == "bf16":
        dtype_kw["dtype"] = torch.bfloat16
    elif args.dtype == "fp16":
        dtype_kw["dtype"] = torch.float16
    elif args.dtype == "fp32":
        dtype_kw["dtype"] = torch.float32
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    # Load dense only, evaluate, then free — avoids two full copies on device (OOM on MPS).
    dense = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        **dtype_kw,
    ).to(device)
    if device == "cpu":
        dense = dense.to(dtype=torch.float32)
    dense_rep = _eval_model(dense, tok, prompts, device=device, max_new_tokens=args.max_new_tokens)
    _release_model(dense)
    _reclaim_device_memory(device)

    moe_rep, replaced, loaded_layers = _run_moe_eval_branch(
        model_id=args.model_id,
        device=device,
        dtype_kw=dtype_kw,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        layers=layers,
        num_experts=args.num_experts,
        top_k=args.top_k,
        shared_experts=args.shared_experts,
        offload_unused_experts=args.offload_unused_experts,
        adapter_path=args.adapter_path,
        adapter_state=None,
        warmup_steps=args.warmup_steps,
        tok=tok,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
    )

    moe_shrunk_rep: dict | None = None
    shrunk_meta: dict[str, object] = {}
    if args.adapter_path_shrunk:
        _reclaim_device_memory(device)
        st_shrunk = torch.load(args.adapter_path_shrunk, map_location="cpu", weights_only=True)
        ne_inf = infer_num_experts_from_moe_adapter_state(st_shrunk, layers)
        if args.num_experts_shrunk and args.num_experts_shrunk != ne_inf:
            raise SystemExit(
                f"--num-experts-shrunk {args.num_experts_shrunk} does not match adapter ({ne_inf} experts)."
            )
        ne_s = args.num_experts_shrunk or ne_inf
        shrunk_meta = {"num_experts_inferred": ne_inf, "num_experts_used": ne_s}
        moe_shrunk_rep, _, loaded_shrunk = _run_moe_eval_branch(
            model_id=args.model_id,
            device=device,
            dtype_kw=dtype_kw,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            layers=layers,
            num_experts=ne_s,
            top_k=args.top_k,
            shared_experts=args.shared_experts,
            offload_unused_experts=args.offload_unused_experts,
            adapter_path="",
            adapter_state=st_shrunk,
            warmup_steps=args.warmup_steps,
            tok=tok,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
        )
        shrunk_meta["adapter_layers_loaded"] = loaded_shrunk

    eval_strategy = "sequential_dense_then_moe"
    if args.adapter_path_shrunk:
        eval_strategy += "_then_moe_shrunk"

    report: dict = {
        "model_id": args.model_id,
        "eval_strategy": eval_strategy,
        "layers": replaced,
        "moe_config": {
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "shared_experts": args.shared_experts,
            "offload_unused_experts": args.offload_unused_experts,
        },
        "adapter_path": args.adapter_path,
        "adapter_layers_loaded": loaded_layers,
        "dense": {"avg_loss": dense_rep["avg_loss"], "p95_latency_ms": dense_rep["p95_latency_ms"]},
        "moe_bridge": {"avg_loss": moe_rep["avg_loss"], "p95_latency_ms": moe_rep["p95_latency_ms"]},
        "delta": {
            "avg_loss": moe_rep["avg_loss"] - dense_rep["avg_loss"],
            "p95_latency_ms": moe_rep["p95_latency_ms"] - dense_rep["p95_latency_ms"],
        },
        "samples_dense": dense_rep["samples"],
        "samples_moe_bridge": moe_rep["samples"],
    }
    if moe_shrunk_rep is not None:
        report["adapter_path_shrunk"] = args.adapter_path_shrunk
        report["moe_shrunk_meta"] = shrunk_meta
        report["moe_bridge_shrunk"] = {
            "avg_loss": moe_shrunk_rep["avg_loss"],
            "p95_latency_ms": moe_shrunk_rep["p95_latency_ms"],
        }
        report["delta_shrunk_vs_dense"] = {
            "avg_loss": moe_shrunk_rep["avg_loss"] - dense_rep["avg_loss"],
            "p95_latency_ms": moe_shrunk_rep["p95_latency_ms"] - dense_rep["p95_latency_ms"],
        }
        report["delta_shrunk_vs_moe"] = {
            "avg_loss": moe_shrunk_rep["avg_loss"] - moe_rep["avg_loss"],
            "p95_latency_ms": moe_shrunk_rep["p95_latency_ms"] - moe_rep["p95_latency_ms"],
        }
        report["samples_moe_bridge_shrunk"] = moe_shrunk_rep["samples"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_keys = ["dense", "moe_bridge", "delta"]
    if moe_shrunk_rep is not None:
        summary_keys.extend(["moe_bridge_shrunk", "delta_shrunk_vs_dense", "delta_shrunk_vs_moe"])
    print(json.dumps({k: report[k] for k in summary_keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
