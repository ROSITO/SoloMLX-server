from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn


@dataclass
class MoEOffloadReport:
    model_type: str
    keep_experts: int
    strategy: str
    layers_touched: int
    experts_before: int
    experts_after: int
    single_expert_fastpath: bool


class _MixtralSingleExpertFastPath(nn.Module):
    def __init__(self, switch_mlp: Any, expert_index: int = 0):
        super().__init__()
        self.switch_mlp = switch_mlp
        self.expert_index = int(expert_index)

    def __call__(self, x: mx.array) -> mx.array:
        inds = mx.full((*x.shape[:-1], 1), self.expert_index, dtype=mx.int32)
        y = self.switch_mlp(x, inds)
        return y.squeeze(-2)


class _Qwen2MoeSingleExpertFastPath(nn.Module):
    def __init__(
        self,
        switch_mlp: Any,
        shared_expert: Any,
        shared_expert_gate: Any,
        expert_index: int = 0,
    ):
        super().__init__()
        self.switch_mlp = switch_mlp
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        self.expert_index = int(expert_index)

    def __call__(self, x: mx.array) -> mx.array:
        inds = mx.full((*x.shape[:-1], 1), self.expert_index, dtype=mx.int32)
        y = self.switch_mlp(x, inds).squeeze(-2)
        shared = self.shared_expert(x)
        gate = mx.sigmoid(self.shared_expert_gate(x))
        return y + gate * shared


def _slice_first_dim(arr: Any, indices: mx.array) -> Any:
    if arr is None:
        return None
    return mx.take(arr, indices, axis=0)


def _slice_switch_linear_experts(mod: Any, indices: mx.array) -> None:
    for name in ("weight", "bias", "scales", "biases"):
        if hasattr(mod, name):
            cur = getattr(mod, name)
            if cur is not None:
                setattr(mod, name, _slice_first_dim(cur, indices))


def _top_expert_indices_from_gate(gate_weight: Any, keep_experts: int, strategy: str) -> mx.array:
    n_experts = int(gate_weight.shape[0])
    k = max(1, min(keep_experts, n_experts))
    if strategy == "first":
        return mx.arange(k)
    # default: keep experts with largest gate row norm (static proxy usefulness)
    norms = mx.sqrt(mx.sum(gate_weight * gate_weight, axis=1))
    order = mx.argsort(-norms)
    return order[:k]


def _apply_mixtral(
    model: Any, keep_experts: int, strategy: str, enable_single_expert_fastpath: bool
) -> MoEOffloadReport:
    layers = model.model.layers
    touched = 0
    before = int(model.args.num_local_experts)
    idx_cache: dict[int, mx.array] = {}
    for i, layer in enumerate(layers):
        moe = layer.block_sparse_moe
        idx = _top_expert_indices_from_gate(moe.gate.weight, keep_experts, strategy)
        idx_cache[i] = idx
        moe.gate.weight = _slice_first_dim(moe.gate.weight, idx)
        _slice_switch_linear_experts(moe.switch_mlp.gate_proj, idx)
        _slice_switch_linear_experts(moe.switch_mlp.up_proj, idx)
        _slice_switch_linear_experts(moe.switch_mlp.down_proj, idx)
        moe.num_experts = int(idx.shape[0])
        if enable_single_expert_fastpath and moe.num_experts == 1:
            layer.block_sparse_moe = _MixtralSingleExpertFastPath(moe.switch_mlp, expert_index=0)
        touched += 1
    after = int(idx_cache[0].shape[0]) if idx_cache else before
    model.args.num_local_experts = after
    if model.args.num_experts_per_tok > after:
        model.args.num_experts_per_tok = after
    return MoEOffloadReport(
        model_type="mixtral",
        keep_experts=keep_experts,
        strategy=strategy,
        layers_touched=touched,
        experts_before=before,
        experts_after=after,
        single_expert_fastpath=bool(enable_single_expert_fastpath and after == 1),
    )


def _apply_qwen2_moe(
    model: Any, keep_experts: int, strategy: str, enable_single_expert_fastpath: bool
) -> MoEOffloadReport:
    layers = model.model.layers
    touched = 0
    before = int(model.args.num_experts)
    idx_cache: dict[int, mx.array] = {}
    for i, layer in enumerate(layers):
        moe = layer.mlp
        idx = _top_expert_indices_from_gate(moe.gate.weight, keep_experts, strategy)
        idx_cache[i] = idx
        moe.gate.weight = _slice_first_dim(moe.gate.weight, idx)
        _slice_switch_linear_experts(moe.switch_mlp.gate_proj, idx)
        _slice_switch_linear_experts(moe.switch_mlp.up_proj, idx)
        _slice_switch_linear_experts(moe.switch_mlp.down_proj, idx)
        moe.num_experts = int(idx.shape[0])
        if moe.top_k > moe.num_experts:
            moe.top_k = moe.num_experts
        if enable_single_expert_fastpath and moe.num_experts == 1:
            layer.mlp = _Qwen2MoeSingleExpertFastPath(
                moe.switch_mlp,
                moe.shared_expert,
                moe.shared_expert_gate,
                expert_index=0,
            )
        touched += 1
    after = int(idx_cache[0].shape[0]) if idx_cache else before
    model.args.num_experts = after
    if model.args.num_experts_per_tok > after:
        model.args.num_experts_per_tok = after
    return MoEOffloadReport(
        model_type="qwen2_moe",
        keep_experts=keep_experts,
        strategy=strategy,
        layers_touched=touched,
        experts_before=before,
        experts_after=after,
        single_expert_fastpath=bool(enable_single_expert_fastpath and after == 1),
    )


def apply_moe_expert_offload(
    model: Any,
    keep_experts: int,
    strategy: str = "l2",
    enable_single_expert_fastpath: bool = True,
) -> MoEOffloadReport | None:
    if keep_experts <= 0:
        return None
    model_type = getattr(getattr(model, "args", None), "model_type", "")
    if model_type == "mixtral":
        return _apply_mixtral(model, keep_experts, strategy, enable_single_expert_fastpath)
    if model_type == "qwen2_moe":
        return _apply_qwen2_moe(model, keep_experts, strategy, enable_single_expert_fastpath)
    return None
