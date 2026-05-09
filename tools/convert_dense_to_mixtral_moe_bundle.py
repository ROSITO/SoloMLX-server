from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.errors import RemoteEntryNotFoundError
from safetensors import safe_open
from safetensors.torch import save_file


REQUIRED_META = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]

OPTIONAL_META = [
    "special_tokens_map.json",
    "tokenizer.model",
    "tekken.json",
]


def _parse_layer_id(key: str) -> int | None:
    # model.layers.{idx}.mlp.*
    if not key.startswith("model.layers."):
        return None
    parts = key.split(".")
    if len(parts) < 4 or parts[1] != "layers":
        return None
    if not parts[2].isdigit():
        return None
    return int(parts[2])


def _copy_meta(repo_id: str, out_dir: Path) -> None:
    for name in REQUIRED_META:
        src = Path(hf_hub_download(repo_id, name))
        shutil.copy2(src, out_dir / name)
    for name in OPTIONAL_META:
        try:
            src = Path(hf_hub_download(repo_id, name))
            shutil.copy2(src, out_dir / name)
        except RemoteEntryNotFoundError:
            continue


def _get_shards(repo_id: str) -> list[str]:
    files = sorted(f for f in list_repo_files(repo_id) if f.endswith(".safetensors"))
    shards = [f for f in files if f.startswith("model-") and "-of-" in f]
    return shards if shards else files


def _rewrite_config_as_mixtral(config_path: Path, *, num_experts: int, top_k: int) -> int:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg["model_type"] = "mixtral"
    cfg["num_local_experts"] = int(num_experts)
    cfg["num_experts_per_tok"] = int(top_k)
    cfg.setdefault("num_key_value_heads", cfg.get("num_attention_heads"))
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return int(cfg["num_hidden_layers"])


def _convert_shard(
    src_file: Path,
    dst_file: Path,
    *,
    num_experts: int,
    layers_to_convert: set[int],
    emitted_gate_layers: set[int],
) -> dict[str, str]:
    out: dict[str, torch.Tensor] = {}
    weight_map: dict[str, str] = {}
    with safe_open(str(src_file), framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            layer_id = _parse_layer_id(key)
            is_target_layer = layer_id is not None and layer_id in layers_to_convert
            is_mlp = ".mlp." in key
            if is_target_layer and is_mlp and key.endswith("gate_proj.weight"):
                for e in range(num_experts):
                    nk = key.replace(".mlp.gate_proj.weight", f".block_sparse_moe.experts.{e}.w1.weight")
                    out[nk] = t.clone()
                    weight_map[nk] = dst_file.name
                if layer_id not in emitted_gate_layers:
                    hidden = int(t.shape[1])
                    gk = f"model.layers.{layer_id}.block_sparse_moe.gate.weight"
                    out[gk] = torch.zeros((num_experts, hidden), dtype=t.dtype)
                    weight_map[gk] = dst_file.name
                    emitted_gate_layers.add(layer_id)
                continue
            if is_target_layer and is_mlp and key.endswith("up_proj.weight"):
                for e in range(num_experts):
                    nk = key.replace(".mlp.up_proj.weight", f".block_sparse_moe.experts.{e}.w3.weight")
                    out[nk] = t.clone()
                    weight_map[nk] = dst_file.name
                continue
            if is_target_layer and is_mlp and key.endswith("down_proj.weight"):
                for e in range(num_experts):
                    nk = key.replace(".mlp.down_proj.weight", f".block_sparse_moe.experts.{e}.w2.weight")
                    out[nk] = t.clone()
                    weight_map[nk] = dst_file.name
                continue

            # keep non-converted keys; drop dense mlp keys on converted layers
            if is_target_layer and is_mlp:
                continue
            out[key] = t
            weight_map[key] = dst_file.name

    save_file(out, str(dst_file))
    return weight_map


def main() -> int:
    p = argparse.ArgumentParser(description="Convert a dense Mistral-like HF repo into a Mixtral-style MoE HF bundle.")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-experts", type=int, default=2)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--layers", default="all", help="Comma-separated layer ids or 'all'.")
    p.add_argument("--smoke-load", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _copy_meta(args.repo_id, out_dir)
    n_layers = _rewrite_config_as_mixtral(out_dir / "config.json", num_experts=args.num_experts, top_k=args.top_k)

    if args.layers.strip().lower() == "all":
        layers_to_convert = set(range(n_layers))
    else:
        layers_to_convert = {int(x.strip()) for x in args.layers.split(",") if x.strip()}

    shards = _get_shards(args.repo_id)
    emitted_gate_layers: set[int] = set()
    total_weight_map: dict[str, str] = {}
    output_shards: list[Path] = []

    for i, shard_name in enumerate(shards, 1):
        src = Path(hf_hub_download(args.repo_id, shard_name))
        dst = out_dir / shard_name.replace(".safetensors", ".mixtral-moe.safetensors")
        print(f"[mixtral-moe] convert shard {i}/{len(shards)}: {src.name} -> {dst.name}", flush=True)
        wm = _convert_shard(
            src,
            dst,
            num_experts=args.num_experts,
            layers_to_convert=layers_to_convert,
            emitted_gate_layers=emitted_gate_layers,
        )
        total_weight_map.update(wm)
        output_shards.append(dst)

    total_size = sum(p.stat().st_size for p in output_shards)
    idx = {
        "metadata": {"total_size": total_size},
        "weight_map": total_weight_map,
    }
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2), encoding="utf-8")

    report: dict[str, object] = {
        "repo_id": args.repo_id,
        "out_dir": str(out_dir.resolve()),
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "layers_converted_count": len(layers_to_convert),
        "layers_with_gate_emitted_count": len(emitted_gate_layers),
        "output_shards": [p.name for p in output_shards],
    }

    if args.smoke_load:
        try:
            from mlx_lm import load

            _m, _t = load(str(out_dir))
            report["mlx_load_ok"] = True
        except Exception as e:  # noqa: BLE001
            report["mlx_load_ok"] = False
            report["mlx_load_error"] = str(e)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
