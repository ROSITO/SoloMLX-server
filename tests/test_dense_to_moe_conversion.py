from pathlib import Path

import torch
from safetensors.torch import save_file

from tools.convert_dense_to_moe import (
    MoEConversionConfig,
    convert_safetensors_dense_to_moe,
    convert_state_dict_dense_to_moe,
    dense_to_moe_key_variants,
    detect_ffn_keys_from_safetensors,
    parse_layer_id_from_key,
    select_safetensors_for_conversion,
)
from training.moe_model import MoEModelConfig, MoEModelScaffold


def test_dense_key_expands_to_experts_and_shared() -> None:
    cfg = MoEConversionConfig(num_experts=4, top_k=1, shared_experts=1)
    src = "model.layers.0.mlp.up_proj.weight"
    variants = dense_to_moe_key_variants(src, cfg)
    assert len(variants) == 5
    assert "model.layers.0.moe.experts.0.up_proj.weight" in variants
    assert "model.layers.0.moe.shared_experts.0.up_proj.weight" in variants


def test_convert_state_dict_copies_ffn_and_keeps_non_ffn() -> None:
    cfg = MoEConversionConfig(num_experts=3, top_k=1, shared_experts=1)
    dense_state = {
        "model.layers.0.mlp.up_proj.weight": "UP",
        "model.layers.0.self_attn.q_proj.weight": "Q",
    }
    out = convert_state_dict_dense_to_moe(dense_state, cfg)
    assert "model.layers.0.self_attn.q_proj.weight" in out
    assert out["model.layers.0.self_attn.q_proj.weight"] == "Q"
    assert "model.layers.0.moe.experts.0.up_proj.weight" in out
    assert "model.layers.0.moe.experts.1.up_proj.weight" in out
    assert "model.layers.0.moe.experts.2.up_proj.weight" in out
    assert "model.layers.0.moe.shared_experts.0.up_proj.weight" in out


def test_training_scaffold_summary() -> None:
    model = MoEModelScaffold(
        MoEModelConfig(num_experts=4, top_k=1, shared_experts=1, layers_to_convert=(2, 6, 10))
    )
    s = model.summary()
    assert s["num_experts"] == 4
    assert s["top_k"] == 1
    assert s["status"] == "scaffold"


def test_select_safetensors_drops_consolidated_when_shards_present() -> None:
    files = [
        "consolidated.safetensors",
        "model-00001-of-00010.safetensors",
        "model-00002-of-00010.safetensors",
    ]
    out = select_safetensors_for_conversion(files)
    assert "consolidated.safetensors" not in out
    assert out == sorted([f for f in files if f != "consolidated.safetensors"])


def test_select_safetensors_keeps_consolidated_when_no_shards() -> None:
    assert select_safetensors_for_conversion(["consolidated.safetensors"]) == ["consolidated.safetensors"]


def test_parse_layer_id_from_key() -> None:
    assert parse_layer_id_from_key("model.layers.12.mlp.up_proj.weight") == 12
    assert parse_layer_id_from_key("model.embed_tokens.weight") is None


def test_convert_safetensors_dense_to_moe_file(tmp_path: Path) -> None:
    src = tmp_path / "dense.safetensors"
    dst = tmp_path / "moe.safetensors"
    tensors = {
        "model.layers.2.mlp.up_proj.weight": torch.ones(2, 2),
        "model.layers.2.self_attn.q_proj.weight": torch.ones(2, 2),
    }
    save_file(tensors, str(src))
    cfg = MoEConversionConfig(num_experts=3, top_k=1, shared_experts=1)
    ffn = detect_ffn_keys_from_safetensors(str(src), cfg)
    assert len(ffn) == 1
    converted, total = convert_safetensors_dense_to_moe(
        str(src), str(dst), cfg, layers_to_convert={2}
    )
    assert total == 2
    assert converted == 1
