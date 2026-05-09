from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.moe_model import TopKMoEFFN


class DenseTeacherFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_stabilization(
    *,
    steps: int = 200,
    batch_size: int = 8,
    seq_len: int = 32,
    dim: int = 256,
    hidden_dim: int = 512,
    num_experts: int = 4,
    top_k: int = 1,
    shared_experts: int = 1,
    lr: float = 2e-4,
    balance_coeff: float = 0.05,
    compute_coeff: float = 0.01,
    device: str = "cpu",
) -> dict:
    torch.manual_seed(42)
    teacher = DenseTeacherFFN(dim=dim, hidden_dim=hidden_dim).to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    moe = TopKMoEFFN(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        shared_experts=shared_experts,
    ).to(device)
    opt = torch.optim.AdamW(moe.parameters(), lr=lr)

    losses: list[float] = []
    entropies: list[float] = []
    usage_accum = {f"expert_{i}": 0.0 for i in range(num_experts)}

    for _ in range(steps):
        x = torch.randn(batch_size, seq_len, dim, device=device)
        with torch.no_grad():
            y = teacher(x)

        y_hat, stats = moe(x)
        main_loss = F.mse_loss(y_hat, y)
        usage = torch.tensor([stats["expert_usage"][f"expert_{i}"] for i in range(num_experts)], device=device)
        balance_loss = ((usage - usage.mean()) ** 2).mean()
        # Proxy compute penalty: higher top-k and shared experts increase active compute.
        compute_pen = torch.tensor(float(top_k + shared_experts) / float(num_experts + shared_experts), device=device)
        loss = main_loss + balance_coeff * balance_loss + compute_coeff * compute_pen

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(moe.parameters(), 1.0)
        opt.step()

        losses.append(float(loss.item()))
        entropies.append(float(stats["router_entropy"]))
        for k, v in stats["expert_usage"].items():
            usage_accum[k] += float(v)

    usage_avg = {k: v / steps for k, v in usage_accum.items()}
    report = {
        "steps": steps,
        "final_loss": losses[-1],
        "loss_start": losses[0],
        "loss_delta": losses[-1] - losses[0],
        "loss_min": min(losses),
        "loss_max": max(losses),
        "router_entropy_avg": sum(entropies) / len(entropies),
        "expert_usage_avg": usage_avg,
        "config": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "dim": dim,
            "hidden_dim": hidden_dim,
            "num_experts": num_experts,
            "top_k": top_k,
            "shared_experts": shared_experts,
            "lr": lr,
            "balance_coeff": balance_coeff,
            "compute_coeff": compute_coeff,
            "device": device,
        },
    }
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Run MoE stabilization smoke training.")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--out", default="bench/moe_training/stabilize_report.json")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    report = run_stabilization(steps=args.steps, device=args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
