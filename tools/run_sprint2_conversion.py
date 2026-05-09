from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

from tools.convert_dense_to_moe import (
    MoEConversionConfig,
    convert_safetensors_dense_to_moe,
    download_all_safetensors,
    detect_ffn_keys_from_safetensors,
    parse_layer_id_from_key,
)


def parse_layers(raw: str) -> set[int]:
    if not raw.strip():
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def main() -> int:
    p = argparse.ArgumentParser(description="Run real checkpoint Dense->MoE bootstrap conversion.")
    p.add_argument("--repo-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--workdir", default="bench/moe_conversion")
    p.add_argument("--layers", default="2,6,10", help="Comma-separated layer ids to convert")
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--shared-experts", type=int, default=1)
    args = p.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"[moe_convert] workdir={workdir.resolve()} layers={args.layers}", flush=True)
    src_files = download_all_safetensors(args.repo_id, str(workdir))

    cfg = MoEConversionConfig(
        num_experts=args.num_experts,
        top_k=args.top_k,
        shared_experts=args.shared_experts,
    )
    layer_set = parse_layers(args.layers)
    total_ffn = 0
    total_selected = 0
    converted = 0
    total = 0
    outputs: list[str] = []
    inputs: list[str] = []
    for si, src in enumerate(src_files, 1):
        src_path = Path(src)
        out = workdir / f"{src_path.stem}.moe-bootstrap.safetensors"
        print(f"[moe_convert] shard {si}/{len(src_files)} scan FFN keys: {src_path.name}", flush=True)
        ffn_keys = detect_ffn_keys_from_safetensors(src, cfg)
        selected = [k for k in ffn_keys if parse_layer_id_from_key(k) in layer_set]
        print(
            f"[moe_convert] shard {si}/{len(src_files)} convert → {out.name} "
            f"(ffn_keys={len(ffn_keys)} selected_layer_keys={len(selected)})",
            flush=True,
        )
        c, t = convert_safetensors_dense_to_moe(
            src,
            str(out),
            cfg,
            layers_to_convert=layer_set if layer_set else None,
        )
        print(f"[moe_convert] shard {si}/{len(src_files)} done converted_tensors={c} total_keys={t}", flush=True)
        converted += c
        total += t
        total_ffn += len(ffn_keys)
        total_selected += len(selected)
        inputs.append(str(src_path))
        outputs.append(str(out))
    report = {
        "repo_id": args.repo_id,
        "inputs": inputs,
        "outputs": outputs,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "shared_experts": args.shared_experts,
        "shards": len(src_files),
        "ffn_keys_detected": total_ffn,
        "ffn_keys_selected": total_selected,
        "converted_keys": converted,
        "total_input_keys": total,
        "layers": sorted(layer_set),
    }
    rep_path = workdir / "conversion_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
