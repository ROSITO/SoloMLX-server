from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from training.moe_model import TopKMoELlamaMLP, replace_llama_mlp_with_moe


def run_bridge_smoke(
    *,
    model_id: str,
    layers: list[int],
    steps: int = 20,
    batch_size: int = 2,
    seq_len: int = 32,
    num_experts: int = 4,
    top_k: int = 1,
    shared_experts: int = 1,
    lr: float = 1e-4,
    device: str = "cpu",
) -> dict:
    torch.manual_seed(7)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.train()
    model.to(device)
    if device == "cpu":
        model.to(dtype=torch.float32)
    model, replaced = replace_llama_mlp_with_moe(
        model,
        layers,
        num_experts=num_experts,
        top_k=top_k,
        shared_experts=shared_experts,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    vocab = int(model.config.vocab_size)

    losses: list[float] = []
    entropies: list[float] = []
    usage_accum = {f"expert_{i}": 0.0 for i in range(num_experts)}
    t0 = time.perf_counter()
    for _ in range(steps):
        input_ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)
        main_loss = out.loss

        # aggregate MoE stats + balancing loss over converted layers
        bal_terms = []
        step_entropy = 0.0
        n_stats = 0
        for li in layers:
            mlp = model.model.layers[li].mlp
            if not isinstance(mlp, TopKMoELlamaMLP):
                continue
            st = mlp.last_stats
            usage = torch.tensor([st["expert_usage"][f"expert_{i}"] for i in range(num_experts)], device=device)
            bal_terms.append(((usage - usage.mean()) ** 2).mean())
            step_entropy += float(st["router_entropy"])
            n_stats += 1
            for k, v in st["expert_usage"].items():
                usage_accum[k] += float(v)
        balance_loss = torch.stack(bal_terms).mean() if bal_terms else torch.tensor(0.0, device=device)
        compute_pen = torch.tensor(float(top_k + shared_experts) / float(num_experts + shared_experts), device=device)
        loss = main_loss + 0.05 * balance_loss + 0.01 * compute_pen

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(float(loss.item()))
        if n_stats:
            entropies.append(step_entropy / n_stats)

    elapsed = time.perf_counter() - t0
    usage_avg = {k: v / max(1, (steps * len(layers))) for k, v in usage_accum.items()}
    report = {
        "model_id": model_id,
        "layers_replaced": replaced,
        "steps": steps,
        "loss_start": losses[0],
        "final_loss": losses[-1],
        "loss_delta": losses[-1] - losses[0],
        "loss_min": min(losses),
        "loss_max": max(losses),
        "router_entropy_avg": (sum(entropies) / len(entropies)) if entropies else 0.0,
        "expert_usage_avg": usage_avg,
        "seconds_total": elapsed,
        "seconds_per_step": elapsed / max(1, steps),
        "config": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_experts": num_experts,
            "top_k": top_k,
            "shared_experts": shared_experts,
            "lr": lr,
            "device": device,
        },
    }
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Bridge smoke: HF dense checkpoint with MoE-replaced layers.")
    p.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--layers", default="2,6,10")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out", default="bench/moe_training/bridge_smoke_report.json")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    report = run_bridge_smoke(model_id=args.model_id, layers=layers, steps=args.steps, device=args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
