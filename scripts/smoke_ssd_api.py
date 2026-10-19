"""Validate local SSD-paged serving through FastAPI, without starting a listener."""
import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--cache-gb', type=float, default=4)
    parser.add_argument('--memory-limit-gb', type=float, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.model.is_dir() or args.cache_gb < 0 or args.memory_limit_gb <= 0:
        parser.error('An existing local checkpoint and valid memory budgets are required')
    os.environ.update(MLXSERVE_RUNTIME_BACKEND='mlx', MLXSERVE_SSD_EXPERTS_ENABLED='true',
        MLXSERVE_SSD_EXPERTS_CACHE_GB=str(args.cache_gb), MLXSERVE_PREFILL_STEP_SIZE='8',
        MLXSERVE_PROMPT_CACHE_GB='0.125', MLXSERVE_MTP='false',
        MLXSERVE_PRELOAD_DEFAULT_MODEL='false', MLXSERVE_CONTEXT_TOKENS='4096',
        MLXSERVE_DEFAULT_MODEL=str(args.model.resolve()))
    import mlx.core as mx
    import psutil
    from fastapi.testclient import TestClient
    from mlxserve.runtime.ssd_telemetry import MemoryMonitor
    from mlxserve.api.app import app
    from mlxserve.api.deps import engine
    from mlxserve.config import settings

    mx.set_memory_limit(int(args.memory_limit_gb * 1024**3))
    report = dict(status='running', requests=[],
        checkpoint_bytes=sum(f.stat().st_size for f in args.model.glob('model*.safetensors')),
        physical_ram_bytes=psutil.virtual_memory().total)
    report['checkpoint_exceeds_ram'] = report['checkpoint_bytes'] > report['physical_ram_bytes']
    headers = {'Authorization': f'Bearer {settings.api_key}'} if settings.api_key else {}
    monitor = MemoryMonitor()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with monitor:
            with TestClient(app) as client:
                for stream in (False, True):
                    start = time.perf_counter()
                    body = dict(messages=[{'role': 'user', 'content': 'Capitale de la France ? Un seul mot.'}],
                                temperature=0, max_tokens=24, stream=stream)
                    if stream:
                        body['stream_options'] = {'include_usage': True}
                    response = client.post('/v1/chat/completions', json=body, headers=headers)
                    result = dict(stream=stream, status_code=response.status_code,
                                  seconds=time.perf_counter() - start)
                    assert response.status_code == 200, response.text
                    if stream:
                        events = [json.loads(line[6:]) for line in response.text.splitlines()
                                  if line.startswith('data: ') and line != 'data: [DONE]']
                        result['text'] = ''.join(event['choices'][0]['delta'].get('content', '') for event in events)
                        result['done'] = 'data: [DONE]' in response.text
                        result['usage'] = next((event['usage'] for event in reversed(events) if 'usage' in event), None)
                        result['finish_reason'] = events[-1]['choices'][0]['finish_reason']
                        assert result['done'] and result['usage']
                    else:
                        payload = response.json()
                        result.update(text=payload['choices'][0]['message']['content'],
                                      usage=payload['usage'], finish_reason=payload['choices'][0]['finish_reason'])
                    assert 'Paris' in result['text'], result
                    report['requests'].append(result)
                    print(json.dumps(result), flush=True)
                assert report['requests'][0]['text'] == report['requests'][1]['text']
                report['health'] = client.get('/health').json()
                report['cache'] = engine.backend._ssd_expert_cache.stats()
                report['model_num_experts'] = getattr(engine.backend.model.args, 'num_experts', None)
                report['model_num_experts_per_tok'] = getattr(engine.backend.model.args, 'num_experts_per_tok', None)
                engine.unload_model()
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['memory'] = monitor.report()
        report['peak_mlx_bytes'] = mx.get_peak_memory()
        args.output.write_text(json.dumps(report, indent=2))
        engine.unload_model()
        engine.backend._mlx_thread.shutdown(wait=True)
    print(json.dumps({k: v for k, v in report.items() if k != 'memory'}, indent=2))


if __name__ == '__main__':
    main()
