import torch
from transformers import LlamaConfig, LlamaForCausalLM

from training.moe_model import TopKMoELlamaMLP, replace_llama_mlp_with_moe


def test_replace_llama_mlp_with_moe_on_tiny_model() -> None:
    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
    )
    model = LlamaForCausalLM(cfg)
    model, replaced = replace_llama_mlp_with_moe(model, [0], num_experts=4, top_k=1, shared_experts=1)
    assert replaced == [0]
    assert isinstance(model.model.layers[0].mlp, TopKMoELlamaMLP)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = model(input_ids=x, labels=x)
    assert y.loss is not None


def test_replace_llama_mlp_with_moe_offloads_experts_to_cpu_after_forward() -> None:
    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
    )
    model = LlamaForCausalLM(cfg)
    model, _ = replace_llama_mlp_with_moe(
        model,
        [0],
        num_experts=4,
        top_k=1,
        shared_experts=1,
        offload_unused_experts=True,
    )
    mlp = model.model.layers[0].mlp
    assert isinstance(mlp, TopKMoELlamaMLP)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    _ = model(input_ids=x, labels=x)
    for i in range(mlp.num_experts):
        assert mlp.experts_gate[i].weight.device.type == "cpu"
        assert mlp.experts_up[i].weight.device.type == "cpu"
        assert mlp.experts_down[i].weight.device.type == "cpu"


def test_offload_experts_train_backward_does_not_crash() -> None:
    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
    )
    model = LlamaForCausalLM(cfg)
    model.train()
    model, _ = replace_llama_mlp_with_moe(
        model,
        [0],
        num_experts=4,
        top_k=1,
        shared_experts=0,
        offload_unused_experts=True,
    )
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids=x, labels=x)
    out.loss.backward()
