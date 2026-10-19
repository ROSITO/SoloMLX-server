"""Standalone SSD-expert validation with physical footprint and swap sampling.

Run resident and SSD modes as separate processes. Use an idle Mac for timing.
"""
import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Existing local MLX checkpoint directory')
    parser.add_argument('--cache-gb', type=float, default=1.0)
    parser.add_argument('--memory-limit-gb', type=float, default=8.0)
    parser.add_argument('--prompt', default='The capital of France is')
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=16)
    parser.add_argument('--prefill-step-size', type=int, default=8)
    parser.add_argument('--resident', action='store_true', help='Reference without SSD paging')
    parser.add_argument('--projection-mode', action='store_true', help='Control: page each projection separately')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.cache_gb < 0 or args.memory_limit_gb <= 0 or min(args.max_tokens, args.prefill_step_size, args.rounds) <= 0:
        parser.error('Invalid cache size, memory limit, rounds or token count')
    directory = Path(args.model)
    if not directory.is_dir():
        parser.error('--model must be an existing local checkpoint (no automatic download)')

    import mlx.core as mx
    import psutil
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlxserve.runtime.ssd_experts import attach_ssd_experts
    from mlxserve.runtime.ssd_telemetry import MemoryMonitor

    mx.set_memory_limit(int(args.memory_limit_gb * 1024**3))
    mx.set_cache_limit(128 * 1024**2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_bytes = sum(f.stat().st_size for f in directory.glob('model*.safetensors'))
    physical_ram = psutil.virtual_memory().total
    report = dict(model=str(directory.resolve()), resident=args.resident,
        checkpoint_bytes=checkpoint_bytes, physical_ram_bytes=physical_ram,
        checkpoint_exceeds_ram=checkpoint_bytes > physical_ram,
        cache_limit_bytes=int(args.cache_gb * 1024**3),
        mlx_limit_bytes=int(args.memory_limit_gb * 1024**3),
        prompt=args.prompt, chat=args.chat, group_experts=not args.projection_mode, rounds=[], status='running')
    cache = None
    monitor = MemoryMonitor()

    def save():
        report['memory'] = monitor.report()
        report['cache'] = cache.stats() if cache else None
        report['peak_mlx_bytes'] = mx.get_peak_memory()
        report['active_mlx_bytes'] = mx.get_active_memory()
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(args.output)

    try:
        with monitor:
            monitor.phase = 'load'
            save()
            start = time.perf_counter()
            model, tokenizer = load(str(directory), lazy=not args.resident)
            cache = None if args.resident else attach_ssd_experts(model, directory, report['cache_limit_bytes'], group_experts=not args.projection_mode)
            report['load_seconds'] = time.perf_counter() - start
            report['loaded_memory'] = monitor.sample()
            prompt = args.prompt
            if args.chat:
                prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
            save()
            print(json.dumps(dict(phase='loaded', seconds=report['load_seconds'],
                                  active_mlx_bytes=mx.get_active_memory())), flush=True)
            for index in range(args.rounds):
                monitor.phase = f'round_{index + 1}_prefill'
                start = time.perf_counter()
                run = dict(text='', tokens=[], token_seconds=[])
                report['rounds'].append(run)
                for response in stream_generate(model, tokenizer, prompt=prompt,
                        sampler=make_sampler(temp=0), max_tokens=args.max_tokens,
                        prefill_step_size=args.prefill_step_size):
                    run['text'] += response.text
                    run['tokens'].append(response.token)
                    run['token_seconds'].append(time.perf_counter() - start)
                    if len(run['tokens']) == 1:
                        run['cache_after_first_token'] = cache.stats() if cache else None
                        run['prompt_tokens'] = response.prompt_tokens
                    monitor.phase = f'round_{index + 1}_decode'
                    monitor.sample()
                    save()
                    print(json.dumps(dict(round=index + 1, token=len(run['tokens']),
                                          text=response.text)), flush=True)
                run['inference_seconds'] = time.perf_counter() - start
                if len(run['token_seconds']) > 1:
                    run['decode_tokens_per_second'] = (len(run['tokens']) - 1) / max(
                        1e-9, run['token_seconds'][-1] - run['token_seconds'][0])
                run['cache'] = cache.stats() if cache else None
                save()
            report['status'] = 'complete'
    except BaseException as exc:
        report['status'] = 'error'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save()
    print(json.dumps({k: v for k, v in report.items() if k != 'memory'}, indent=2))


if __name__ == '__main__':
    main()
