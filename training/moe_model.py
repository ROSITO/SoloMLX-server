"""MoE training scaffold for Sprint 2.

This file intentionally contains only architecture placeholders and interfaces.
Real model bindings (torch/transformers) will be added after source model lock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEModelConfig:
    num_experts: int = 4
    top_k: int = 1
    shared_experts: int = 1
    layers_to_convert: tuple[int, ...] = ()


def validate_config(cfg: MoEModelConfig) -> None:
    if cfg.num_experts < 2:
        raise ValueError("num_experts must be >= 2")
    if cfg.top_k < 1 or cfg.top_k > cfg.num_experts:
        raise ValueError("top_k must be in [1, num_experts]")
    if cfg.shared_experts < 0:
        raise ValueError("shared_experts must be >= 0")


class MoEModelScaffold:
    """Thin scaffold to freeze interfaces before heavy implementation."""

    def __init__(self, cfg: MoEModelConfig) -> None:
        validate_config(cfg)
        self.cfg = cfg

    def summary(self) -> dict[str, object]:
        return {
            "num_experts": self.cfg.num_experts,
            "top_k": self.cfg.top_k,
            "shared_experts": self.cfg.shared_experts,
            "layers_to_convert": list(self.cfg.layers_to_convert),
            "status": "scaffold",
        }


class TopKMoEFFN(nn.Module):
    """Small MoE FFN block used for training smoke/stabilization."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 1,
        shared_experts: int = 1,
        fast_top1: bool = True,
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if shared_experts < 0:
            raise ValueError("shared_experts must be >= 0")
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_experts = shared_experts
        self.fast_top1 = fast_top1

        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.shared = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, dim),
                )
                for _ in range(shared_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        # x: [batch, seq, dim]
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        if self.fast_top1 and self.top_k == 1:
            top_idx = torch.argmax(logits, dim=-1, keepdim=True)
            top_vals = torch.ones_like(top_idx, dtype=x.dtype)
        else:
            top_vals, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
            norm = top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            top_vals = top_vals / norm

        out = torch.zeros_like(x)
        flat_x = x.reshape(-1, self.dim)
        flat_out = out.reshape(-1, self.dim)
        flat_idx = top_idx.reshape(-1, self.top_k)
        flat_vals = top_vals.reshape(-1, self.top_k)
        tok_ids = torch.arange(flat_x.shape[0], device=x.device).unsqueeze(1).expand_as(flat_idx)
        pair_tok = tok_ids.reshape(-1)
        pair_exp = flat_idx.reshape(-1)
        pair_w = flat_vals.reshape(-1)

        for expert_id, expert in enumerate(self.experts):
            m = pair_exp == expert_id
            if not torch.any(m):
                continue
            selected_tok = pair_tok[m]
            selected_w = pair_w[m].unsqueeze(1)
            y = expert(flat_x[selected_tok])
            flat_out[selected_tok] += y * selected_w
        out = flat_out.view_as(x)

        if self.shared_experts:
            shared_out = torch.zeros_like(x)
            for mod in self.shared:
                shared_out = shared_out + mod(x)
            out = out + shared_out / float(self.shared_experts)

        # diagnostics
        mean_probs = probs.mean(dim=(0, 1))
        entropy = float((-(probs * (probs.clamp_min(1e-8).log())).sum(dim=-1)).mean().item())
        usage = {f"expert_{i}": float(mean_probs[i].item()) for i in range(self.num_experts)}
        stats = {"router_entropy": entropy, "expert_usage": usage}
        return out, stats


class TopKMoELlamaMLP(nn.Module):
    """Drop-in replacement for Hugging Face `LlamaMLP` with MoE routing."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        act_fn: nn.Module,
        num_experts: int = 4,
        top_k: int = 1,
        shared_experts: int = 1,
        fast_top1: bool = True,
        offload_unused_experts: bool = False,
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if shared_experts < 0:
            raise ValueError("shared_experts must be >= 0")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_experts = shared_experts
        self.fast_top1 = fast_top1
        self.offload_unused_experts = offload_unused_experts
        self.act_fn = act_fn

        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts_gate = nn.ModuleList([nn.Linear(hidden_size, intermediate_size, bias=False) for _ in range(num_experts)])
        self.experts_up = nn.ModuleList([nn.Linear(hidden_size, intermediate_size, bias=False) for _ in range(num_experts)])
        self.experts_down = nn.ModuleList([nn.Linear(intermediate_size, hidden_size, bias=False) for _ in range(num_experts)])
        self.shared_gate = nn.ModuleList([nn.Linear(hidden_size, intermediate_size, bias=False) for _ in range(shared_experts)])
        self.shared_up = nn.ModuleList([nn.Linear(hidden_size, intermediate_size, bias=False) for _ in range(shared_experts)])
        self.shared_down = nn.ModuleList([nn.Linear(intermediate_size, hidden_size, bias=False) for _ in range(shared_experts)])
        self.last_stats: dict[str, Any] = {"router_entropy": 0.0, "expert_usage": {}}
        self._last_router_probs: torch.Tensor | None = None

    def _move_expert(self, expert_id: int, device: torch.device) -> None:
        self.experts_gate[expert_id].to(device=device)
        self.experts_up[expert_id].to(device=device)
        self.experts_down[expert_id].to(device=device)

    def _experts_place_for_forward(self, needed: set[int], compute_device: torch.device) -> None:
        """Pin only routed experts on the compute device; park the rest on CPU."""
        if not self.offload_unused_experts:
            return
        for eid in range(self.num_experts):
            target = compute_device if eid in needed else torch.device("cpu")
            self._move_expert(eid, target)

    def _experts_release_to_cpu(self) -> None:
        if not self.offload_unused_experts:
            return
        cpu = torch.device("cpu")
        for eid in range(self.num_experts):
            self._move_expert(eid, cpu)

    @staticmethod
    def _ffn(gate: nn.Linear, up: nn.Linear, down: nn.Linear, act_fn: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return down(act_fn(gate(x)) * up(x))

    def warmstart_from_dense(self, dense_mlp: nn.Module) -> None:
        # Copy dense FFN weights into each expert/shared branch.
        for i in range(self.num_experts):
            self.experts_gate[i].weight.data.copy_(dense_mlp.gate_proj.weight.data)
            self.experts_up[i].weight.data.copy_(dense_mlp.up_proj.weight.data)
            self.experts_down[i].weight.data.copy_(dense_mlp.down_proj.weight.data)
        for i in range(self.shared_experts):
            self.shared_gate[i].weight.data.copy_(dense_mlp.gate_proj.weight.data)
            self.shared_up[i].weight.data.copy_(dense_mlp.up_proj.weight.data)
            self.shared_down[i].weight.data.copy_(dense_mlp.down_proj.weight.data)
        # near-uniform router init
        nn.init.zeros_(self.router.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        if self.fast_top1 and self.top_k == 1:
            top_idx = torch.argmax(logits, dim=-1, keepdim=True)
            top_vals = torch.ones_like(top_idx, dtype=x.dtype)
        else:
            top_vals, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
            norm = top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            top_vals = top_vals / norm
        out = torch.zeros_like(x)
        flat_x = x.reshape(-1, self.hidden_size)
        flat_out = out.reshape(-1, self.hidden_size)
        flat_idx = top_idx.reshape(-1, self.top_k)
        flat_vals = top_vals.reshape(-1, self.top_k)
        tok_ids = torch.arange(flat_x.shape[0], device=x.device).unsqueeze(1).expand_as(flat_idx)
        pair_tok = tok_ids.reshape(-1)
        pair_exp = flat_idx.reshape(-1)
        pair_w = flat_vals.reshape(-1)
        needed = {int(i) for i in torch.unique(pair_exp).tolist()}
        self._experts_place_for_forward(needed, x.device)
        for expert_id in range(self.num_experts):
            m = pair_exp == expert_id
            if not torch.any(m):
                continue
            selected_tok = pair_tok[m]
            selected_w = pair_w[m].unsqueeze(1)
            y = self._ffn(
                self.experts_gate[expert_id],
                self.experts_up[expert_id],
                self.experts_down[expert_id],
                self.act_fn,
                flat_x[selected_tok],
            )
            flat_out[selected_tok] += y * selected_w
        out = flat_out.view_as(x)

        if self.shared_experts:
            s = torch.zeros_like(x)
            for i in range(self.shared_experts):
                s = s + self._ffn(self.shared_gate[i], self.shared_up[i], self.shared_down[i], self.act_fn, x)
            out = out + s / float(self.shared_experts)

        mean_probs = probs.mean(dim=(0, 1))
        entropy = float((-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)).mean().item())
        usage = {f"expert_{i}": float(mean_probs[i].item()) for i in range(self.num_experts)}
        self.last_stats = {"router_entropy": entropy, "expert_usage": usage}
        # Differentiable router probs for auxiliary losses during training (e.g. entropy minimization).
        self._last_router_probs = probs if self.training else None
        # Releasing to CPU after forward breaks autograd during training.
        if self.offload_unused_experts and not self.training:
            self._experts_release_to_cpu()
        return out


def replace_llama_mlp_with_moe(
    model: nn.Module,
    layers_to_convert: list[int],
    *,
    num_experts: int = 4,
    top_k: int = 1,
    shared_experts: int = 1,
    fast_top1: bool = True,
    offload_unused_experts: bool = False,
) -> tuple[nn.Module, list[int]]:
    replaced: list[int] = []
    for i in layers_to_convert:
        layer = model.model.layers[i]
        dense_mlp = layer.mlp
        moe = TopKMoELlamaMLP(
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            act_fn=dense_mlp.act_fn,
            num_experts=num_experts,
            top_k=top_k,
            shared_experts=shared_experts,
            fast_top1=fast_top1,
            offload_unused_experts=offload_unused_experts,
        )
        ref_w = dense_mlp.gate_proj.weight
        moe = moe.to(device=ref_w.device, dtype=ref_w.dtype)
        moe.warmstart_from_dense(dense_mlp)
        if offload_unused_experts:
            moe._experts_release_to_cpu()
        layer.mlp = moe
        replaced.append(i)
    return model, replaced


def export_moe_state(model: nn.Module, layers: list[int]) -> dict[str, torch.Tensor]:
    """Export only MoE layer parameters for lightweight checkpointing."""
    out: dict[str, torch.Tensor] = {}
    for li in layers:
        mlp = model.model.layers[li].mlp
        if not isinstance(mlp, TopKMoELlamaMLP):
            continue
        prefix = f"layers.{li}."
        for name, param in mlp.state_dict().items():
            out[prefix + name] = param.detach().cpu().clone()
    return out


def load_moe_state(model: nn.Module, layers: list[int], state: dict[str, torch.Tensor]) -> int:
    """Load previously exported MoE parameters into replaced layers."""
    loaded = 0
    for li in layers:
        mlp = model.model.layers[li].mlp
        if not isinstance(mlp, TopKMoELlamaMLP):
            continue
        prefix = f"layers.{li}."
        sub = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        if not sub:
            continue
        mlp.load_state_dict(sub, strict=False)
        loaded += 1
    return loaded


def _moe_state_layer_and_suffix(key: str) -> tuple[int, str] | None:
    if not key.startswith("layers."):
        return None
    parts = key.split(".")
    if len(parts) < 3:
        return None
    try:
        li = int(parts[1])
    except ValueError:
        return None
    suffix = ".".join(parts[2:])
    return li, suffix


def infer_num_experts_from_moe_adapter_state(
    state: dict[str, torch.Tensor],
    layer_indices: list[int],
) -> int:
    """Read ``num_experts`` from ``router.weight`` shape ``[num_experts, hidden]``."""
    for li in layer_indices:
        w = state.get(f"layers.{li}.router.weight")
        if w is not None:
            return int(w.shape[0])
    raise ValueError("cannot infer num_experts: missing router.weight for the given layers")


def shrink_moe_state_dict(
    state: dict[str, torch.Tensor],
    *,
    new_num_experts: int,
    rank_strategy: str = "l2_router",
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Drop least important experts from an exported MoE adapter (per-layer router ranking).

    Shared-expert tensors are copied unchanged. Target model must be built with
    ``replace_llama_mlp_with_moe(..., num_experts=new_num_experts)`` before ``load_moe_state``.
    """
    if new_num_experts < 2:
        raise ValueError("new_num_experts must be >= 2")
    if rank_strategy != "l2_router":
        raise ValueError(f"unsupported rank_strategy: {rank_strategy!r}")

    by_layer: dict[int, dict[str, torch.Tensor]] = {}
    extra: list[tuple[str, torch.Tensor]] = []
    for k, v in state.items():
        parsed = _moe_state_layer_and_suffix(k)
        if parsed is None:
            extra.append((k, v))
            continue
        li, suffix = parsed
        by_layer.setdefault(li, {})[suffix] = v

    out: dict[str, torch.Tensor] = {k: v.clone() for k, v in extra}
    meta: dict[str, object] = {}

    for li, sub in sorted(by_layer.items()):
        prefix = f"layers.{li}."
        router_w = sub.get("router.weight")
        if router_w is None:
            for suf, v in sub.items():
                out[prefix + suf] = v.clone()
            continue
        old_e = int(router_w.shape[0])
        if new_num_experts >= old_e:
            for suf, v in sub.items():
                out[prefix + suf] = v.clone()
            meta[str(li)] = {"skipped": True, "old_num_experts": old_e}
            continue
        scores = router_w.norm(dim=1)
        _, idx = torch.sort(scores, descending=True)
        pick = idx[:new_num_experts].tolist()
        new_router = router_w[pick].clone()
        out[prefix + "router.weight"] = new_router
        for new_i, old_i in enumerate(pick):
            for stem in ("experts_gate", "experts_up", "experts_down"):
                sk = f"{stem}.{old_i}.weight"
                if sk not in sub:
                    raise KeyError(f"missing {prefix}{sk} for shrink")
                out[prefix + f"{stem}.{new_i}.weight"] = sub[sk].clone()
        for suf, v in sub.items():
            if suf.startswith("shared_"):
                out[prefix + suf] = v.clone()
        meta[str(li)] = {"old_num_experts": old_e, "new_num_experts": new_num_experts, "kept_old_indices": pick}

    return out, {"layers": meta, "rank_strategy": rank_strategy}
