#!/usr/bin/env python3
"""Shrink a PyTorch MoE adapter checkpoint (per-layer l2_router ranking).

Loads ``export_moe_state`` / ``moe_target_train`` adapter .pt files, drops weak
experts, writes a new checkpoint for use with ``replace_llama_mlp_with_moe`` +
``load_moe_state`` using the smaller ``num_experts``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training.moe_model import shrink_moe_state_dict


def main() -> int:
    p = argparse.ArgumentParser(description="Shrink MoE adapter checkpoint by expert count.")
    p.add_argument("--in-adapter", required=True, help="Input .pt from export_moe_state / moe_target_train.")
    p.add_argument("--out-adapter", required=True, help="Output .pt with fewer experts.")
    p.add_argument("--new-num-experts", type=int, required=True, help="Target expert count (>=2, < current).")
    p.add_argument(
        "--rank-strategy",
        default="l2_router",
        choices=("l2_router",),
        help="Per-layer ranking of experts to keep.",
    )
    p.add_argument("--out-meta", default="", help="Optional JSON path for shrink metadata.")
    args = p.parse_args()

    state = torch.load(args.in_adapter, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise SystemExit("adapter must be a dict of tensors")
    shrunk, meta = shrink_moe_state_dict(
        state,
        new_num_experts=args.new_num_experts,
        rank_strategy=args.rank_strategy,
    )
    outp = Path(args.out_adapter)
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(shrunk, outp)
    if args.out_meta:
        Path(args.out_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"out_adapter": str(outp), "meta": meta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
