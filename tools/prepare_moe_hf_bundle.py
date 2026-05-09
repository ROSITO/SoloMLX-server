from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError


REQUIRED_META_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
]

OPTIONAL_META_FILES = [
    "special_tokens_map.json",
    "params.json",
    "tekken.json",
]


def _copy_metadata(base_repo: str, out_dir: Path) -> None:
    for name in REQUIRED_META_FILES:
        src = Path(hf_hub_download(base_repo, name))
        shutil.copy2(src, out_dir / name)
    for name in OPTIONAL_META_FILES:
        try:
            src = Path(hf_hub_download(base_repo, name))
            shutil.copy2(src, out_dir / name)
        except RemoteEntryNotFoundError:
            # Some repos (e.g. Qwen) do not expose params.json / tekken.json.
            continue


def _rewrite_index_to_moe_bootstrap(index_path: Path) -> int:
    idx = json.loads(index_path.read_text(encoding="utf-8"))
    wm = idx.get("weight_map", {})
    changed = 0
    for key, filename in list(wm.items()):
        if isinstance(filename, str) and filename.startswith("model-") and filename.endswith(".safetensors"):
            new_name = filename.replace(".safetensors", ".moe-bootstrap.safetensors")
            if new_name != filename:
                wm[key] = new_name
                changed += 1
    index_path.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    return changed


def _link_moe_shards(source_dir: Path, out_dir: Path) -> int:
    count = 0
    for shard in sorted(source_dir.glob("model-*-of-*.moe-bootstrap.safetensors")):
        dst = out_dir / shard.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(shard.resolve())
        count += 1
    return count


def _smoke_load(model_path: Path) -> tuple[bool, str]:
    try:
        from mlx_lm import load

        _model, _tok = load(str(model_path))
        return True, "mlx_lm.load() OK"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def main() -> int:
    p = argparse.ArgumentParser(description="Prepare local HF-style bundle from MoE bootstrap shards.")
    p.add_argument("--source-dir", required=True, help="Directory containing model-*-of-*.moe-bootstrap.safetensors")
    p.add_argument("--base-repo", required=True, help="HF repo id used to fetch config/tokenizer/index metadata")
    p.add_argument("--out-dir", required=True, help="Output local model directory for mlx_lm.load() smoke tests")
    p.add_argument("--smoke-load", action="store_true", help="Run mlx_lm.load() at the end and print result")
    args = p.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _copy_metadata(args.base_repo, out_dir)
    remapped = _rewrite_index_to_moe_bootstrap(out_dir / "model.safetensors.index.json")
    linked = _link_moe_shards(source_dir, out_dir)

    report: dict[str, object] = {
        "source_dir": str(source_dir.resolve()),
        "base_repo": args.base_repo,
        "out_dir": str(out_dir.resolve()),
        "index_entries_remapped": remapped,
        "moe_shards_linked": linked,
    }

    if args.smoke_load:
        ok, msg = _smoke_load(out_dir)
        report["mlx_load_ok"] = ok
        report["mlx_load_message"] = msg

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
