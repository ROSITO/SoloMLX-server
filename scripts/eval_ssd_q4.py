"""Teacher-forced logits comparison: native gather_qmm vs direct cached Q4 kernel."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--prompt-file', type=Path, required=True)
    p.add_argument('--fused-swiglu', action='store_true')
    p.add_argument('--tokens', type=int, default=64)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import wired_limit
    from mlx_lm.models.cache import make_prompt_cache
    from mlxserve.runtime.ssd_experts import attach_ssd_experts, SSDGroupedSwitch

    mx.set_memory_limit(8 * 1024**3)
    mx.set_cache_limit(128 * 1024**2)
    model, tokenizer = load(str(args.model), lazy=True)
    cache = attach_ssd_experts(model, args.model, 4 * 1024**3)
    blocks = [m for _, m in model.named_modules() if isinstance(m, SSDGroupedSwitch)]
    prompt = tokenizer.encode(args.prompt_file.read_text(), add_special_tokens=False)[:256]
    references, generated, rows = [], [], []

    def native_prefill():
        for b in blocks:
            b._direct_matvec = False
        kv = make_prompt_cache(model)
        model(mx.array(prompt[:-1])[None], cache=kv)
        mx.eval([c.state for c in kv])
        return kv

    with wired_limit(model):
        kv = native_prefill()
        token = mx.array([[prompt[-1]]])
        for i in range(args.tokens):
            logits = model(token, cache=kv)[:, -1, :].astype(mx.float32)
            token = mx.argmax(logits, axis=-1)[:, None]
            mx.eval(logits, token)
            references.append(logits)
            generated.append(token.item())
        print('Native references ready', flush=True)
        kv = native_prefill()
        for b in blocks:
            b._direct_matvec = True
            b._fused_matvec = args.fused_swiglu
        for i in range(args.tokens):
            previous = prompt[-1] if i == 0 else generated[i - 1]
            actual = model(mx.array([[previous]]), cache=kv)[:, -1, :].astype(mx.float32)
            expected = references[i]
            a = actual - mx.logsumexp(actual, axis=-1, keepdims=True)
            b = expected - mx.logsumexp(expected, axis=-1, keepdims=True)
            kl = mx.sum(mx.exp(b) * (b - a))
            maximum = mx.max(mx.abs(actual - expected))
            predicted = mx.argmax(actual, axis=-1)
            mx.eval(kl, maximum, predicted)
            rows.append(dict(index=i, native=generated[i], direct=predicted.item(),
                             kl=kl.item(), max_abs_logit_error=maximum.item()))
    result = dict(tokens=args.tokens, fused_swiglu=args.fused_swiglu, argmax_agreement=sum(r['native']==r['direct'] for r in rows)/len(rows),
                  mean_kl=sum(r['kl'] for r in rows)/len(rows), max_kl=max(r['kl'] for r in rows),
                  model=str(args.model), comparisons=rows, cache=cache.stats())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('comparisons','cache')},indent=2))


if __name__ == '__main__':
    main()
