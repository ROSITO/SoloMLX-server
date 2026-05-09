import torch
from transformers import LlamaConfig, LlamaForCausalLM

from training.moe_model import (
    export_moe_state,
    infer_num_experts_from_moe_adapter_state,
    load_moe_state,
    replace_llama_mlp_with_moe,
    shrink_moe_state_dict,
)


def test_export_and_load_moe_state_roundtrip() -> None:
    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
    )
    src = LlamaForCausalLM(cfg)
    src, _ = replace_llama_mlp_with_moe(src, [0], num_experts=4, top_k=1, shared_experts=0)
    state = export_moe_state(src, [0])
    assert state

    dst = LlamaForCausalLM(cfg)
    dst, _ = replace_llama_mlp_with_moe(dst, [0], num_experts=4, top_k=1, shared_experts=0)
    loaded = load_moe_state(dst, [0], state)
    assert loaded == 1
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    out = dst(input_ids=x, labels=x)
    assert out.loss is not None


def test_shrink_moe_state_dict_loads_into_smaller_moe() -> None:
    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
    )
    src = LlamaForCausalLM(cfg)
    src, _ = replace_llama_mlp_with_moe(src, [0], num_experts=4, top_k=1, shared_experts=0)
    with torch.no_grad():
        src.model.layers[0].mlp.router.weight[0].fill_(2.0)
        src.model.layers[0].mlp.router.weight[1:].zero_()
    state = export_moe_state(src, [0])
    shrunk, meta = shrink_moe_state_dict(state, new_num_experts=2)
    assert meta["layers"]["0"]["new_num_experts"] == 2
    assert 0 in meta["layers"]["0"]["kept_old_indices"]

    dst = LlamaForCausalLM(cfg)
    dst, _ = replace_llama_mlp_with_moe(dst, [0], num_experts=2, top_k=1, shared_experts=0)
    assert load_moe_state(dst, [0], shrunk) == 1
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    out = dst(input_ids=x, labels=x)
    assert out.loss is not None


def test_infer_num_experts_from_moe_adapter_state() -> None:
    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
    )
    src = LlamaForCausalLM(cfg)
    src, _ = replace_llama_mlp_with_moe(src, [0], num_experts=4, top_k=1, shared_experts=0)
    state = export_moe_state(src, [0])
    assert infer_num_experts_from_moe_adapter_state(state, [0]) == 4
