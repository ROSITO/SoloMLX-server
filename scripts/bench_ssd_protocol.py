"""Fixed-token, greedy SSD benchmark with independent prefill/decode timers.

Like janas-bench: prefill N-1 tokens, then G single-token steps, ignore EOS,
retain the expert cache across a warm-up and measured rounds, reset KV each time.
Benchmark input text is supplied separately and recorded by hash, not embedded.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--prompt-file', type=Path, required=True)
    p.add_argument('--prompt-tokens', type=int, default=256)
    p.add_argument('--gen-tokens', type=int, default=64)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--prefill-step', type=int, default=256)
    p.add_argument('--cache-policy', choices=['lru','lfu'], default='lru')
    p.add_argument('--fused-swiglu', action='store_true', help='Experimental fused native Q4 SwiGLU kernel')
    p.add_argument('--frequency-cap', type=int, default=None, help='Cap LFU credit per expert fetch (experimental)')
    p.add_argument('--cache-gb', type=float, default=8)
    p.add_argument('--memory-limit-gb', type=float, default=12)
    p.add_argument('--packed-dir', type=Path, default=None, help='Directory with ssd_experts*.safetensors (one read per expert)')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if min(args.prompt_tokens - 1, args.gen_tokens, args.rounds, args.prefill_step) <= 0 or args.warmup < 0:
        p.error('Invalid token counts, rounds or warm-up')
    import mlx.core as mx
    import importlib.metadata
    import psutil
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from mlxserve.runtime.ssd_experts import attach_ssd_experts
    from mlxserve.runtime.ssd_telemetry import MemoryMonitor

    mx.set_memory_limit(int(args.memory_limit_gb * 1024**3))
    mx.set_cache_limit(128 * 1024**2)
    text = args.prompt_file.read_text()
    report = dict(status='running', model=str(args.model), prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
        prompt_tokens=args.prompt_tokens, gen_tokens=args.gen_tokens, warmup=args.warmup,
        cache_limit_bytes=int(args.cache_gb * 1024**3), rounds=[],
        physical_ram_bytes=psutil.virtual_memory().total,
        checkpoint_bytes=sum(f.stat().st_size for f in args.model.glob('model*.safetensors')),
        protocol='prefill N-1 then G forward+greedy steps; EOS ignored; KV reset; expert cache retained',
        cache_policy=args.cache_policy, frequency_cap=args.frequency_cap, fused_swiglu=args.fused_swiglu, prefill_step=args.prefill_step, memory_limit_gb=args.memory_limit_gb, packed_dir=str(args.packed_dir) if args.packed_dir else None,
        runtime_versions={n: importlib.metadata.version(n) for n in ('mlx','mlx-lm')},
        source_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [Path('src/mlxserve/runtime/ssd_experts.py'), Path('src/mlxserve/runtime/ssd_qmm.py'), Path(__file__)]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_snapshot = {name: Path(name).read_text() for name in report["source_sha256"]}
    args.output.with_suffix(".sources.json").write_text(json.dumps(source_snapshot, indent=2))
    monitor = MemoryMonitor()
    cache = None
    previous_wired_limit = None

    def save():
        report.update(memory=monitor.report(), peak_mlx_bytes=mx.get_peak_memory(),
                      cache=cache.stats() if cache else None)
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(args.output)

    try:
        with monitor:
            monitor.phase = 'load'
            model, tokenizer = load(str(args.model), lazy=True)
            cache = attach_ssd_experts(model, args.model, report['cache_limit_bytes'], cache_policy=args.cache_policy,
                                       frequency_cap=args.frequency_cap, packed_dir=args.packed_dir)
            if args.fused_swiglu:
                from mlxserve.runtime.ssd_experts import SSDGroupedSwitch
                for _, module in model.named_modules():
                    if type(module) is SSDGroupedSwitch:
                        module._fused_matvec = True
            recommended = mx.device_info().get('max_recommended_working_set_size')
            if recommended is not None:
                previous_wired_limit = mx.set_wired_limit(recommended)
                report['wired_limit_bytes'] = recommended
            once = tokenizer.encode(text, add_special_tokens=False)
            if not once:
                raise ValueError('Empty benchmark prompt')
            report['unrepeated_prompt_tokens'] = len(once)
            # Janas repeats its prose only when the requested token count exceeds it.
            repeated = text * ((args.prompt_tokens + len(once) - 1) // len(once))
            prompt = tokenizer.encode(repeated, add_special_tokens=False)[:args.prompt_tokens]
            if len(prompt) != args.prompt_tokens:
                raise ValueError('Prompt corpus did not produce the requested length')
            report['prompt_token_sha256'] = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
            save()
            for index in range(-args.warmup, args.rounds):
                kv = make_prompt_cache(model)
                monitor.phase = f'round_{index}_prefill'
                run = dict(round=index, warmup=index < 0, tokens=[], decode_step_seconds=[])
                report['rounds'].append(run)
                start = time.perf_counter()
                for offset in range(0, len(prompt) - 1, args.prefill_step):
                    block = mx.array(prompt[offset:min(offset + args.prefill_step, len(prompt) - 1)])[None]
                    model(block, cache=kv)
                    mx.eval([c.state for c in kv])
                prefilled = time.perf_counter()
                run.update(prefill_seconds=prefilled - start,
                           prefill_tokens_per_second=(len(prompt) - 1) / (prefilled - start),
                           cache_after_prefill=cache.stats())
                print(json.dumps(dict(round=index, phase='prefill', tps=run['prefill_tokens_per_second'])), flush=True)
                monitor.phase = f'round_{index}_decode'
                token = mx.array([[prompt[-1]]])
                for _ in range(args.gen_tokens):
                    logits = model(token, cache=kv)
                    token = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
                    mx.eval(token)
                    run['tokens'].append(token.item())
                    run['decode_step_seconds'].append(time.perf_counter() - prefilled)
                ended = time.perf_counter()
                run.update(decode_seconds=ended - prefilled,
                           decode_tokens_per_second=args.gen_tokens / (ended - prefilled),
                           cache_after_decode=cache.stats())
                save()
                print(json.dumps({k:v for k,v in run.items() if k not in ('tokens','decode_step_seconds','cache_after_prefill','cache_after_decode')}), flush=True)
                del kv, logits, token
                mx.clear_cache()
            measured = [r for r in report['rounds'] if not r['warmup']]
            report['mean_prefill_tokens_per_second'] = sum(r['prefill_tokens_per_second'] for r in measured) / len(measured)
            report['mean_decode_tokens_per_second'] = sum(r['decode_tokens_per_second'] for r in measured) / len(measured)
            report['tokens_match_across_rounds'] = all(r['tokens'] == measured[0]['tokens'] for r in report['rounds'])
            report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
        if previous_wired_limit is not None:
            mx.synchronize()
            mx.set_wired_limit(previous_wired_limit)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rounds','memory')}, indent=2))


if __name__ == '__main__':
    main()
