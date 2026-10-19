"""Real Metal numerical parity and SSD cache eviction (small synthetic weights)."""
import pytest
mx = pytest.importorskip('mlx.core')
nn = pytest.importorskip('mlx.nn')
pytest.importorskip('mlx_lm')
from mlx.utils import tree_flatten
from mlx_lm.models.switch_layers import SwitchGLU

from mlxserve.runtime.ssd_experts import attach_ssd_experts


class TinyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = SwitchGLU(64, 64, 5, bias=True)


@pytest.mark.parametrize('quantized', [False, True])
@pytest.mark.parametrize('group_experts', [False, True])
@pytest.mark.parametrize('tokens,k', [(1, 2), (7, 3), (32, 3)])
def test_paged_experts_match_resident_with_evictions(tmp_path, quantized, tokens, k, group_experts):
    mx.random.seed(42)
    model = TinyMoE()
    if quantized:
        nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), dict(tree_flatten(model.parameters())))
    x = mx.random.normal((1, tokens, 64))
    indices = (mx.arange(tokens * k) % 5).reshape(1, tokens, k)
    expected = model.experts(x, indices)
    mx.eval(expected)
    expert_bytes = sum(v.nbytes for _, v in tree_flatten(model.experts.parameters())) // 5
    cache = attach_ssd_experts(model, tmp_path, cache_bytes=2 * expert_bytes, group_experts=group_experts)
    actual = model.experts(x, indices)
    mx.eval(actual)
    assert mx.allclose(expected, actual, atol=2e-5, rtol=2e-5).item()
    assert cache.resident_bytes <= cache.max_bytes
    assert cache.evictions > 0 or tokens == 1
    assert cache.store.bytes_read > 0
    assert model.experts.gate_proj.num_experts == 5
    # Full stacked weights are absent from the model parameter tree.
    assert tree_flatten(model.parameters()) == []


def test_zero_cache_and_repeat_hits(tmp_path):
    model = TinyMoE()
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), dict(tree_flatten(model.parameters())))
    cache = attach_ssd_experts(model, tmp_path, cache_bytes=1_000_000)
    x, indices = mx.ones((1, 1, 64)), mx.array([[[2]]])
    mx.eval(model.experts(x, indices))
    read = cache.store.bytes_read
    mx.eval(model.experts(x, indices))
    assert cache.hits == 1
    assert cache.store.bytes_read == read
    cache.clear()
    cache.max_bytes = 0
    mx.eval(model.experts(x, indices))
    assert cache.resident_bytes == 0
    assert cache.store.bytes_read > read


def test_complete_qwen3_moe_logits_match(tmp_path):
    from mlx_lm.models.qwen3_moe import Model, ModelArgs

    args = ModelArgs(model_type='qwen3_moe', hidden_size=64, num_hidden_layers=2,
        intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-6, vocab_size=128, max_position_embeddings=128,
        moe_intermediate_size=64, num_experts=4, num_experts_per_tok=2,
        head_dim=16, decoder_sparse_step=1, mlp_only_layers=[], rope_theta=10000,
        tie_word_embeddings=False, norm_topk_prob=True)
    model = Model(args)
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), dict(tree_flatten(model.parameters())))
    tokens = mx.array([[1, 20, 7, 42]])
    expected = model(tokens)
    mx.eval(expected)
    cache = attach_ssd_experts(model, tmp_path, cache_bytes=16_000)
    actual = model(tokens)
    mx.eval(actual)
    assert mx.allclose(expected, actual, atol=1e-4, rtol=1e-4).item()
    assert mx.argmax(expected, axis=-1).tolist() == mx.argmax(actual, axis=-1).tolist()
    assert cache.evictions > 0


def test_qwen3_next_prefill_and_cached_decode_match(tmp_path):
    from mlx_lm.models.qwen3_next import Model, ModelArgs
    from mlx_lm.models.cache import make_prompt_cache

    args = ModelArgs(model_type='qwen3_next', hidden_size=64, num_hidden_layers=2,
        intermediate_size=128, num_attention_heads=4, linear_num_value_heads=2,
        linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, num_experts=5, num_experts_per_tok=2,
        decoder_sparse_step=1, shared_expert_intermediate_size=64,
        mlp_only_layers=[], moe_intermediate_size=64, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, rope_theta=10000,
        partial_rotary_factor=0.5, max_position_embeddings=128, head_dim=16,
        norm_topk_prob=True, full_attention_interval=2)
    model = Model(args)
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), dict(tree_flatten(model.parameters())))
    resident_kv = make_prompt_cache(model)
    prompt, decode = mx.array([[1, 20, 7, 42]]), mx.array([[11]])
    expected_prefill = model(prompt, cache=resident_kv)
    mx.eval(expected_prefill)
    expected_decode = model(decode, cache=resident_kv)
    mx.eval(expected_decode)
    cache = attach_ssd_experts(model, tmp_path, cache_bytes=16_000)
    # Force several bounded expert batches within a projection.
    for _, module in model.named_modules():
        if hasattr(module, '_batch_bytes'):
            module._batch_bytes = 1
    paged_kv = make_prompt_cache(model)
    actual_prefill = model(prompt, cache=paged_kv)
    mx.eval(actual_prefill)
    actual_decode = model(decode, cache=paged_kv)
    mx.eval(actual_decode)
    for expected, actual in [(expected_prefill, actual_prefill), (expected_decode, actual_decode)]:
        assert mx.allclose(expected, actual, atol=1e-4, rtol=1e-4).item()
        assert mx.argmax(expected, axis=-1).tolist() == mx.argmax(actual, axis=-1).tolist()
    assert cache.resident_bytes <= cache.max_bytes
    assert args.num_experts == 5 and args.num_experts_per_tok == 2


def test_backend_load_generate_release_and_unload_with_native_checkpoint(tmp_path, monkeypatch):
    import asyncio
    import json
    from dataclasses import asdict
    from tokenizers import Tokenizer, models, pre_tokenizers
    from mlx_lm.models.qwen3_moe import Model, ModelArgs
    from mlxserve.config import settings
    from mlxserve.runtime.backends import MLXLMBackend

    args = ModelArgs(model_type='qwen3_moe', hidden_size=64, num_hidden_layers=2,
        intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-6, vocab_size=128, max_position_embeddings=128,
        moe_intermediate_size=64, num_experts=4, num_experts_per_tok=2,
        head_dim=16, decoder_sparse_step=1, mlp_only_layers=[], rope_theta=10000,
        tie_word_embeddings=False, norm_topk_prob=True)
    model = Model(args)
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), dict(tree_flatten(model.parameters())))
    config = dict(asdict(args), quantization={'group_size': 32, 'bits': 4, 'mode': 'affine'})
    (tmp_path / 'config.json').write_text(json.dumps(config))
    tokenizer = Tokenizer(models.WordLevel({str(i): i for i in range(128)}, unk_token='0'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / 'tokenizer.json'))
    (tmp_path / 'tokenizer_config.json').write_text(json.dumps({
        'tokenizer_class': 'PreTrainedTokenizerFast', 'unk_token': '0', 'eos_token': '127'}))
    monkeypatch.setattr(settings, 'ssd_experts_enabled', True)
    monkeypatch.setattr(settings, 'ssd_experts_cache_gb', 16_000 / 1024**3)

    async def scenario():
        paged = MLXLMBackend(prefill_step_size=2)
        resident = MLXLMBackend(prefill_step_size=2)
        try:
            await paged.load(str(tmp_path))
            cache = paged._ssd_expert_cache
            assert cache is not None
            monkeypatch.setattr(settings, 'ssd_experts_enabled', False)
            await resident.load(str(tmp_path))
            expected = await resident.generate('1 20 7', max_tokens=4, temperature=0)
            actual = await paged.generate('1 20 7', max_tokens=4, temperature=0)
            assert actual == expected
            assert cache.misses > 0
            # A failed switch keeps the previous paging owner and model usable.
            monkeypatch.setattr(settings, 'ssd_experts_enabled', True)
            bad_checkpoint = tmp_path / 'incomplete'
            bad_checkpoint.mkdir()
            with pytest.raises(FileNotFoundError):
                await paged.load(str(bad_checkpoint))
            assert paged._ssd_expert_cache is cache
            assert await paged.generate('1 20 7', max_tokens=4, temperature=0) == expected
            await paged.release_memory()
            assert cache.resident_bytes == 0
            paged.unload()
            assert cache.store._fds == {}
        finally:
            paged.unload()
            resident.unload()
            paged._mlx_thread.shutdown(wait=True)
            resident._mlx_thread.shutdown(wait=True)
    asyncio.run(scenario())


@pytest.mark.parametrize('tokens', [1, 2])
@pytest.mark.parametrize('fused', [False, True])
@pytest.mark.parametrize('dtype,tolerance', [(mx.float32, 2e-5), (mx.float16, 0.003), (mx.bfloat16, 0.02)])
def test_direct_packed_matvec_matches_native_quantized_glu(tmp_path, dtype, tolerance, fused, tokens):
    model = TinyMoE()
    model.experts = SwitchGLU(64, 128, 12, bias=False)
    for name in ('up_proj', 'gate_proj', 'down_proj'):
        layer = getattr(model.experts, name)
        layer.weight = layer.weight.astype(dtype)
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), dict(tree_flatten(model.parameters())))
    x = mx.random.normal((1, tokens, 64)).astype(dtype)
    indices = (mx.arange(tokens * 10) * 5 % 12).reshape(1, tokens, 10)
    expected = model.experts(x, indices)
    mx.eval(expected)
    cache = attach_ssd_experts(model, tmp_path, cache_bytes=1_000_000)
    model.experts._fused_matvec = fused
    actual = model.experts(x, indices)
    mx.eval(actual)
    assert cache.direct_q4_calls == 3
    assert cache.fused_q4_calls == int(fused)
    assert mx.allclose(expected, actual, atol=tolerance, rtol=tolerance).item()
    assert cache.resident_bytes <= cache.max_bytes


def test_direct_kernel_falls_back_before_metal_buffer_limit(tmp_path):
    model = TinyMoE()
    model.experts = SwitchGLU(64, 64, 31, bias=False)
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), dict(tree_flatten(model.parameters())))
    x = mx.random.normal((1, 4, 64))
    indices = (mx.arange(32) % 31).reshape(1, 4, 8)
    expected = model.experts(x, indices)
    mx.eval(expected)
    cache = attach_ssd_experts(model, tmp_path, cache_bytes=1_000_000)
    actual = model.experts(x, indices)
    mx.eval(actual)
    assert cache.direct_q4_calls == 0
    assert mx.allclose(expected, actual, atol=2e-5, rtol=2e-5).item()


def test_lfu_keeps_hot_experts_through_scan_and_admits_new_hot_expert(tmp_path):
    from mlxserve.runtime.ssd_experts import ExpertStore, ExpertCache
    weights = mx.arange(5 * 32).reshape(5, 4, 8).astype(mx.float32)
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), {'experts.weight': weights})
    cache = ExpertCache(ExpertStore(tmp_path), max_bytes=2 * 32 * 4, policy='lfu')
    for _ in range(6):
        mx.eval(cache.get('experts', ('weight',), 0)['weight'])
    cache.get('experts', ('weight',), 1)
    for expert in (2, 3, 4):
        loaded = cache.get('experts', ('weight',), expert)
        assert mx.array_equal(loaded['weight'], weights[expert]).item()
    assert ('experts', 0) in cache.entries
    assert cache.admission_bypasses == 3
    cache.get_many('experts', ('weight',), [3], counts={3: 10})
    assert ('experts', 3) in cache.entries
    assert cache.resident_bytes <= cache.max_bytes
    for _ in range(5000):
        cache.get('experts', ('weight',), 0)
    assert len(cache._heap) <= 4 * len(cache.entries) + 1024
    cache.clear()
    assert cache.resident_bytes == 0
    mx.eval(cache.get('experts', ('weight',), 0)['weight'])
    cache.store.close()


def test_lfu_frequency_cap_limits_prefill_credit(tmp_path):
    from mlxserve.runtime.ssd_experts import ExpertStore, ExpertCache
    weights = mx.arange(3 * 32).reshape(3, 4, 8).astype(mx.float32)
    mx.save_safetensors(str(tmp_path / 'model.safetensors'), {'experts.weight': weights})
    cache = ExpertCache(ExpertStore(tmp_path), max_bytes=128, policy='lfu', frequency_cap=1)
    cache.get_many('experts', ('weight',), [0], counts={0: 1000})
    assert cache._frequencies[('experts', 0)] == 1
    cache.get('experts', ('weight',), 1)
    cache.get('experts', ('weight',), 1)
    assert ('experts', 1) in cache.entries
    assert ('experts', 0) not in cache.entries
    assert mx.array_equal(cache.get('experts', ('weight',), 1)['weight'], weights[1]).item()
    cache.store.close()
