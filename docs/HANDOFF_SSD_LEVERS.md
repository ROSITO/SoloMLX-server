# Handoff: SSD expert paging, next optimization levers

State on 19 October 2026, branch `feat/ssd-expert-paging` (PR #12, stacked on PR #11). Background: [SSD expert paging](SSD_EXPERT_PAGING.md), experiment log: [SSD performance log](SSD_PERFORMANCE_PLAN.md).

## Where things stand

- PR #11 and PR #12 are open. Merge #11 first, then #12, which then targets `main`. Merge only when the owner asks.
- PR #12 contains paging (Codex), packed experts (one `pread` per expert), `MLXSERVE_SSD_EXPERTS_PACKED_DIR`, and the fix for a model configured by path being served under its advertised `org/name` id (`_resolve_model` in `src/mlxserve/api/app.py`).
- `python -m pytest -q`: 165 passed.
- **To do:** the PR #12 description doesn't include the results below yet (end-to-end packed run, Mac mini, OpenCode session).

### Machine and data (M3 Pro, 36 GB)

| What | Where |
|---|---|
| Qwen3-Next-80B-A3B-Instruct-4bit checkpoint (41.76 GiB) | `/Volumes/T7 Shield/huggingface/hub/models--mlx-community--Qwen3-Next-80B-A3B-Instruct-4bit/snapshots/d8a069bfa8ae87d3d468412e1034acae19b5892b` (USB, slow, used only for dense weights and the router) |
| Packed experts (internal SSD, ~5 GB/s) | `~/.mlxserve/ssd-bench/Qwen3-Next-packed/ssd_experts.safetensors` |
| Raw evidence (git-ignored) | `artifacts/`, including `artifacts/mini_m4/` |

The T7 cache has no `refs/` directory. Configure the model by its snapshot **path** with `HF_HUB_OFFLINE=1`, not by repo id.

## Measured results

Fixed protocol (`scripts/bench_ssd_protocol.py`): 1 warm-up + 3 rounds, 12 GiB LFU cache, frequency cap 1.

| Experts | Prompt / gen | Prefill tok/s | Decode tok/s |
|---|---|---|---|
| 9 reads per expert | 1024/16 | 84.3 | ~21-23 |
| packed, 1 read per expert | 1024/16 | **121.9** (+45 %) | unchanged |
| 9 reads per expert | 256/64 | 47.9 | 20.84 |
| packed | 256/64 | **69.3** | unchanged |

Decode doesn't move: at 12 GiB the hit rate is ~98.6 %, and ~80 % of the ~44 ms per token is routing synchronization, not I/O.

### Real OpenCode session

6 successive tasks, packed experts, 12 GiB cache, prompt cache 1 GB, 24k context. Harness: `scripts/opencode_ssd_session.sh`, logged by `scripts/peek_proxy.py`.

- **Tasks:** 6/6 succeeded (tests green each time), 33 requests, all HTTP 200, 31 min wall time.
- **Prefix cache:** 343k of 397k prompt tokens were served from the prefix cache.
- **Decode in session: ~8.5-9 tok/s.** For example 887 tokens in 101 s, 1229 in 138 s, 1093 in 125 s, at 10-16k context. That is **less than half the 21 tok/s of the fixed bench.**
- **Cold prefills:** 8.1k tokens in ~107 s, 12.7k tokens in ~135 s (~75-95 tok/s).
- **Compaction and title requests (no tools) never hit the prefix cache:** a 4.6k-token request took 114 s, a 5.5k one 156 s, and after each compaction the agent re-prefills 12.7k tokens cold (202 s).

### Mac mini M4, 16 GB (Nemotron Lightning 30B-A3B, paged)

- **4 GiB cache:** 7.6-8.4 tok/s, footprint 6.2-7.4 GiB.
- **7 GiB cache:** 13.9 tok/s (256/64), footprint 9.3-10.4 GiB. No swap.
- **Mini SSD:** 2.18 GB/s sequential, 2.97 GB/s parallel.
- **Not tested:** the packed format (not enough free disk).
- **Memory budget:** ASIMOV services (~9 GB) and a 7 GiB cache don't fit together.
- **Procedure:** stop ollama, qwen3_server, mlxserve and agents; test in a separate directory; delete the model; restart everything.

## Levers, in recommended order

### 1. Explain the 9 vs 21 tok/s gap in a real session (measure first)

Nothing is instrumented per request yet. Log per request: expert-cache hit rate, MB read per token, `read_wall_seconds`, routing-sync time, and attention time vs context length. Candidate causes:

- Attention cost at 10-16k tokens (the bench uses 256-1024).
- A larger expert working set for code and tool calls than for the bench prose.
- Prompt-cache and KV memory competing with the expert cache.

Re-run with `MLXSERVE_SSD_EXPERTS_CACHE_GB=16` to separate cache misses from attention.

### 2. Routing synchronization in decode (biggest known cost)

Each of the 48 MoE layers forces a GPU→CPU sync to learn its top-10 experts before reading them. At 98.6 % hits this is ~35 of 44 ms per token. Options:

- **GPU-side slot table:** keep cached experts in one preallocated buffer and map expert id → slot on the GPU, so hits need no host round trip. Sync only to detect misses: let the layer run with the hits, or block only if `any(miss)`.
- **Speculative routing:** apply layer l+1's router to layer l's output to predict the next experts. Prefetch them during layer l, then verify. Every routed expert must still be computed; no pruning (unlike Janas's cache-conditional routing).
- **Fewer evals per layer:** batch the host work (fewer `mx.eval`, no `.tolist()` per projection).

### 3. Compaction and secondary requests miss the prefix cache

Keep a second prompt-cache entry for the no-tools prefix (title and compaction prompts), or make sure eviction doesn't drop the main agent prefix. After a compaction, the new conversation prefix is always cold (12.7k tokens, ~135-200 s). Faster prefill (packed experts helped by +45 %) is the only cure there.

### 4. Prefetch and overlap I/O with compute in prefill

Prefill still spends roughly half of each 1024-token chunk reading experts. Read layer l+1's experts (routing known after its router runs on the chunk) while layer l computes. Note: the earlier experiment overlapping shared-expert computation was rejected (19.28 vs 21.84 tok/s), see the performance log.

### 5. Promote the fused Q4 SwiGLU kernel to a server setting

`--fused-swiglu` in the bench: 23.16 vs 20.84 tok/s. It needs broader quality validation first: teacher-forced KL over more prompts, code and French. It is in `src/mlxserve/runtime/ssd_qmm.py`.

### 6. Mac mini: packed format

Free ~20 GB on the mini, pack Lightning (`scripts/pack_ssd_experts.py --src ... --dst ...`), then re-run the 4 GiB and 7 GiB protocols. Estimate (not measured): ~11-12 and ~18-19 tok/s.

## Benchmark pitfalls (already hit)

- **macOS page cache:** `F_NOCACHE` doesn't keep pages out of the page cache. A just-written or just-read file reads at 8-9 GB/s from RAM. Evict first (`msync(MS_INVALIDATE)`, see `scripts/bench_ssd_reads.py`).
- **Other GPU jobs:** check with `ps` before any measurement. A concurrent job (a MuseTalk run) halved decode once.
- **Memory pressure:** the full 80B run needs ~15 GiB free. Watch swap (`sysctl vm.swapusage`); runs that swapped are invalid.

## Commands

```bash
# fixed protocol, packed experts
.venv/bin/python scripts/bench_ssd_protocol.py --model "$CHECKPOINT" --packed-dir "$PACKED_DIR" \
  --prompt-file bench/ssd_prompts/<file> --cache-gb 12 --cache-policy lfu --frequency-cap 1 \
  --memory-limit-gb 16 --prompt-tokens 1024 --gen-tokens 16 --prefill-step 1024 --output artifacts/<name>.json

# disk only, no model load
.venv/bin/python scripts/bench_ssd_reads.py --checkpoint "$CHECKPOINT" --packed-dir "$PACKED_DIR"

# real OpenCode session (6 tasks), logs in $OUT
OUT=artifacts/opencode_session CHECKPOINT=... PACKED_DIR=... scripts/opencode_ssd_session.sh run1
```
