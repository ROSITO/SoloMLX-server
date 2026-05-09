"""Dense -> MoE conversion utilities.

Phase 1:
- framework-agnostic dict conversion for unit testing

Phase 2 bootstrap:
- safetensors checkpoint conversion on real Hugging Face dense models
  (without requiring torch for this conversion step)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, model_info
from huggingface_hub.errors import HfHubHTTPError
from safetensors import safe_open
from safetensors.torch import save_file


@dataclass
class MoEConversionConfig:
    num_experts: int = 4
    top_k: int = 1
    shared_experts: int = 1
    # Key pattern for dense FFN weights in a generic state dict.
    # Example: "model.layers.0.mlp.up_proj.weight"
    dense_ffn_key_suffixes: tuple[str, ...] = (
        "mlp.up_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.down_proj.weight",
    )
    layer_regex_prefix: str = "model.layers."


def is_dense_ffn_key(key: str, config: MoEConversionConfig) -> bool:
    return any(key.endswith(sfx) for sfx in config.dense_ffn_key_suffixes)


def dense_to_moe_key_variants(key: str, config: MoEConversionConfig) -> list[str]:
    """Create destination MoE keys for one dense FFN parameter key."""
    out: list[str] = []
    for i in range(config.num_experts):
        out.append(key.replace(".mlp.", f".moe.experts.{i}."))
    for s in range(config.shared_experts):
        out.append(key.replace(".mlp.", f".moe.shared_experts.{s}."))
    return out


def convert_state_dict_dense_to_moe(
    dense_state: dict[str, object], config: MoEConversionConfig
) -> dict[str, object]:
    """Bootstrap conversion by duplicating dense FFN tensors into experts.

    Non-FFN keys are kept as-is.
    Router weights are initialized implicitly (left to training code).
    """
    if config.num_experts < 2:
        raise ValueError("num_experts must be >= 2")
    if config.top_k < 1 or config.top_k > config.num_experts:
        raise ValueError("top_k must be in [1, num_experts]")
    if config.shared_experts < 0:
        raise ValueError("shared_experts must be >= 0")

    out: dict[str, object] = {}
    for key, value in dense_state.items():
        if is_dense_ffn_key(key, config):
            for new_key in dense_to_moe_key_variants(key, config):
                out[new_key] = value
        else:
            out[key] = value
    return out


def _model_info_with_retry(repo_id: str, *, max_attempts: int = 6) -> Any:
    """HF Hub occasionally returns 5xx; short exponential backoff."""
    delay_s = 2.0
    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return model_info(repo_id)
        except HfHubHTTPError as e:
            last = e
            code = getattr(getattr(e, "response", None), "status_code", None)
            retryable = code in (408, 429, 500, 502, 503, 504)
            if retryable and attempt < max_attempts - 1:
                time.sleep(delay_s)
                delay_s = min(delay_s * 2.0, 60.0)
                continue
            raise
    assert last is not None
    raise last


def _hf_hub_download_with_retry(
    repo_id: str, filename: str, local_dir: str, *, max_attempts: int = 6
) -> str:
    delay_s = 2.0
    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        except HfHubHTTPError as e:
            last = e
            code = getattr(getattr(e, "response", None), "status_code", None)
            retryable = code in (408, 429, 500, 502, 503, 504)
            if retryable and attempt < max_attempts - 1:
                time.sleep(delay_s)
                delay_s = min(delay_s * 2.0, 120.0)
                continue
            raise
    assert last is not None
    raise last


def hf_repo_has_single_safetensors(repo_id: str) -> bool:
    info = _model_info_with_retry(repo_id)
    safes = [s.rfilename for s in (info.siblings or []) if s.rfilename.endswith(".safetensors")]
    return len(safes) == 1


_INDEXED_SHARD_RE = re.compile(r"^model-\d+-of-\d+\.safetensors$")


def select_safetensors_for_conversion(filenames: list[str]) -> list[str]:
    """Prefer HF **shards** when both `consolidated.safetensors` and `model-*-of-*` exist.

    Converting `consolidated.safetensors` builds a full in-memory state dict (~2× model
    size peak) and is impractical on laptops; sharded checkpoints process one slice at a time.
    """
    names = sorted(filenames)
    has_shards = any(_INDEXED_SHARD_RE.match(Path(f).name) for f in names)
    if not has_shards:
        return names
    return sorted(f for f in names if Path(f).name != "consolidated.safetensors")


def list_repo_safetensors(repo_id: str) -> list[str]:
    info = _model_info_with_retry(repo_id)
    safes = [s.rfilename for s in (info.siblings or []) if s.rfilename.endswith(".safetensors")]
    return select_safetensors_for_conversion(safes)


def detect_ffn_keys_from_safetensors(local_file: str, config: MoEConversionConfig) -> list[str]:
    keys: list[str] = []
    with safe_open(local_file, framework="pt") as f:
        for key in f.keys():
            if is_dense_ffn_key(key, config):
                keys.append(key)
    return sorted(keys)


def convert_safetensors_dense_to_moe(
    input_file: str,
    output_file: str,
    config: MoEConversionConfig,
    *,
    layers_to_convert: set[int] | None = None,
) -> tuple[int, int]:
    """Convert one safetensors shard by duplicating dense FFN tensors."""
    out: dict[str, object] = {}
    n_copied = 0
    n_total = 0
    with safe_open(input_file, framework="pt") as f:
        for key in f.keys():
            n_total += 1
            value = f.get_tensor(key)
            should_convert = is_dense_ffn_key(key, config)
            if should_convert and layers_to_convert is not None:
                layer_id = parse_layer_id_from_key(key, prefix=config.layer_regex_prefix)
                should_convert = layer_id in layers_to_convert
            if should_convert:
                for new_key in dense_to_moe_key_variants(key, config):
                    out[new_key] = value.clone()
                n_copied += 1
            else:
                out[key] = value
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, output_file)
    return n_copied, n_total


def parse_layer_id_from_key(key: str, prefix: str = "model.layers.") -> int | None:
    # Expected key pattern: model.layers.{idx}.mlp....
    if not key.startswith(prefix):
        return None
    rest = key[len(prefix) :]
    chunk = rest.split(".", 1)[0]
    if chunk.isdigit():
        return int(chunk)
    return None


def download_single_safetensors(repo_id: str, local_dir: str) -> str:
    if not hf_repo_has_single_safetensors(repo_id):
        raise ValueError(
            f"{repo_id} does not expose a single safetensors file; "
            "this bootstrap converter currently supports one-file checkpoints."
        )
    info = _model_info_with_retry(repo_id)
    safe_name = next(s.rfilename for s in (info.siblings or []) if s.rfilename.endswith(".safetensors"))
    return _hf_hub_download_with_retry(repo_id, safe_name, local_dir)


def download_all_safetensors(repo_id: str, local_dir: str) -> list[str]:
    files = list_repo_safetensors(repo_id)
    if not files:
        raise ValueError(f"{repo_id} does not expose any .safetensors file")
    print(f"[moe_convert] repo={repo_id!r} safetensors_shards={len(files)} (consolidated omitted if sharded)", flush=True)
    out: list[str] = []
    for i, name in enumerate(files, 1):
        print(f"[moe_convert] fetch {i}/{len(files)} {name} …", flush=True)
        path = _hf_hub_download_with_retry(repo_id, name, local_dir)
        print(f"[moe_convert] ready {path}", flush=True)
        out.append(path)
    return out
