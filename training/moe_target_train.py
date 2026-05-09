from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.moe_model import export_moe_state, replace_llama_mlp_with_moe, shrink_moe_state_dict


def load_corpus(path: str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    texts: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        t = obj.get("text", "").strip()
        if t:
            texts.append(t)
    if not texts:
        raise ValueError("Empty training corpus")
    return texts


def main() -> int:
    p = argparse.ArgumentParser(description="Short target training for real MoE bridge layers.")
    p.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--corpus", default="bench/train_corpus_v1.jsonl")
    p.add_argument("--layers", default="10")
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--shared-experts", type=int, default=0)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-adapter", default="bench/moe_training/moe_adapter_v1.pt")
    p.add_argument("--out-report", default="bench/moe_training/moe_target_train_report.json")
    p.add_argument(
        "--offload-unused-experts",
        action="store_true",
        help="Keep inactive MoE experts on CPU each forward (lower device RAM; end-of-forward CPU release only in eval mode).",
    )
    p.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="Weight dtype for from_pretrained. Use bf16 on MPS for large Llama/Mistral checkpoints when RAM allows.",
    )
    p.add_argument(
        "--low-cpu-mem-usage",
        action="store_true",
        help="Pass low_cpu_mem_usage=True to from_pretrained (recommended for large checkpoints).",
    )
    p.add_argument(
        "--router-balance-weight",
        type=float,
        default=0.03,
        help="Penalty on per-layer expert usage variance (encourages load balance). Set 0 when using --router-entropy-weight.",
    )
    p.add_argument(
        "--router-entropy-weight",
        type=float,
        default=0.0,
        help="If >0, adds mean router entropy (differentiable) to the loss — minimize to sharpen routing (expert ticket / pruning prep).",
    )
    p.add_argument(
        "--shrink-to-experts",
        type=int,
        default=0,
        help="If >=2 and < --num-experts, export a second adapter shrunk with l2_router ranking (for smaller MLX resident sets).",
    )
    p.add_argument(
        "--out-adapter-shrunk",
        default="",
        help="Path for shrunk adapter when --shrink-to-experts is set (default: out-adapter with _shrunkK suffix).",
    )
    args = p.parse_args()

    device = args.device
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    texts = load_corpus(args.corpus)
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        **dtype_kw,
    ).to(device)
    if device == "cpu":
        model = model.to(dtype=torch.float32)
    model.train()

    model, replaced = replace_llama_mlp_with_moe(
        model,
        layers,
        num_experts=args.num_experts,
        top_k=args.top_k,
        shared_experts=args.shared_experts,
        offload_unused_experts=args.offload_unused_experts,
    )

    # Train only MoE layer params (adapter-like)
    for p_all in model.parameters():
        p_all.requires_grad = False
    for li in layers:
        mlp = model.model.layers[li].mlp
        for p_moe in mlp.parameters():
            p_moe.requires_grad = True

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    losses: list[float] = []

    for step in range(args.steps):
        text = texts[step % len(texts)]
        enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        main_loss = out.loss

        # Optional load-balance (from detached last_stats) vs sharp routing (differentiable entropy on _last_router_probs).
        bal_terms = []
        for li in layers:
            mlp = model.model.layers[li].mlp
            st = getattr(mlp, "last_stats", None)
            if not st:
                continue
            usage = torch.tensor(
                [st["expert_usage"][f"expert_{i}"] for i in range(mlp.num_experts)],
                device=device,
            )
            bal_terms.append(((usage - usage.mean()) ** 2).mean())
        balance_loss = torch.stack(bal_terms).mean() if bal_terms else torch.tensor(0.0, device=device)

        ent_terms = []
        for li in layers:
            mlp = model.model.layers[li].mlp
            probs = getattr(mlp, "_last_router_probs", None)
            if probs is None:
                continue
            ent = (-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)).mean()
            ent_terms.append(ent)
        entropy_loss = torch.stack(ent_terms).mean() if ent_terms else torch.tensor(0.0, device=device)

        loss = main_loss + args.router_balance_weight * balance_loss + args.router_entropy_weight * entropy_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(float(loss.item()))

    adapter_state = export_moe_state(model, layers)
    out_adapter = Path(args.out_adapter)
    out_adapter.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter_state, out_adapter)

    expert_usage_end: dict[str, dict[str, float]] = {}
    for li in layers:
        mlp = model.model.layers[li].mlp
        st = getattr(mlp, "last_stats", None) or {}
        eu = st.get("expert_usage")
        if isinstance(eu, dict):
            expert_usage_end[str(li)] = {k: float(v) for k, v in eu.items()}

    shrunk_path: str | None = None
    shrink_meta: dict[str, object] | None = None
    if args.shrink_to_experts >= 2 and args.shrink_to_experts < args.num_experts:
        shrunk_state, shrink_meta = shrink_moe_state_dict(
            adapter_state,
            new_num_experts=args.shrink_to_experts,
            rank_strategy="l2_router",
        )
        if args.out_adapter_shrunk:
            sp = Path(args.out_adapter_shrunk)
        else:
            sp = out_adapter.with_name(f"{out_adapter.stem}_shrunk{args.shrink_to_experts}{out_adapter.suffix}")
        sp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(shrunk_state, sp)
        shrunk_path = str(sp)

    report = {
        "model_id": args.model_id,
        "layers": replaced,
        "steps": args.steps,
        "loss_start": losses[0],
        "final_loss": losses[-1],
        "loss_delta": losses[-1] - losses[0],
        "loss_min": min(losses),
        "loss_max": max(losses),
        "adapter_path": str(out_adapter),
        "expert_usage_end": expert_usage_end,
        "shrunk_adapter_path": shrunk_path,
        "shrink_meta": shrink_meta,
        "config": {
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "shared_experts": args.shared_experts,
            "lr": args.lr,
            "device": device,
            "dtype": args.dtype,
            "low_cpu_mem_usage": args.low_cpu_mem_usage,
            "offload_unused_experts": args.offload_unused_experts,
            "router_balance_weight": args.router_balance_weight,
            "router_entropy_weight": args.router_entropy_weight,
            "shrink_to_experts": args.shrink_to_experts,
        },
    }
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
