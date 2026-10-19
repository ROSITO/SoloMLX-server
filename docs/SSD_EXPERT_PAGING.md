# SSD expert paging — validated experimental implementation

Experimental, off by default (`MLXSERVE_SSD_EXPERTS_ENABLED=false`). A MoE checkpoint larger than the Mac's RAM runs through SoloMLX while its routed experts are read from SSD into a bounded cache. No experts are pruned and every routed expert is computed.

## Results at a glance

Qwen3-Next-80B-A3B-Instruct 4-bit (41.76 GiB) on an M3 Pro, 36 GiB RAM, checkpoint on the internal SSD. One warm-up plus three measured rounds, greedy, EOS ignored (protocol in [SSD performance plan](SSD_PERFORMANCE_PLAN.md)).

| Configuration | Path | Prompt/gen | Prefill tok/s | Decode tok/s | Peak footprint |
| --- | --- | --- | --- | --- | --- |
| 12 GiB LFU, frequency cap 1 | server default kernels | 256/64 | 47.9 | **20.84** | 14.37 GiB |
| 12 GiB LFU, cap 1, fused Q4 SwiGLU | benchmark only (`--fused-swiglu`) | 1024/16 | **84.31** | **23.16** | 14.88 GiB |
| 8 GiB LFU, cap 1 | server default kernels | 256/64 | 35.8 | 12.51 | 10.34 GiB |
| 4 GiB LRU, checkpoint on external T7 (USB) | first implementation | short chat | – | 1.2-1.6 | 6.06 GiB |

No swap growth in the internal-SSD runs. For reference, Janas publishes 24 decode / 62 prefill tok/s for the same model on different hardware and quantization (Q4_K_M, ~19 GB expert cache); the 24 tok/s decode mark is not reached. Storage dominates: the internal SSD reads ~5 GB/s, a USB SSD 0.6-0.9 GB/s (a slow cable negotiated USB 2.0 at ~40 MB/s).

## How it works

`ssd_experts.py` indexes safetensors headers and replaces native MLX SwitchLinear/QuantizedSwitchLinear modules after `mlx_lm.load(lazy=True)`, before shared parameters are evaluated. The model's router and shared experts remain unchanged. Every selected expert is computed, including cache misses.

Each miss reads only that expert's packed weight, scale and bias rows with `pread`, using eight I/O workers for raw bytes. Native SwitchGLU/SwitchMLP blocks fetch their selected experts across all projections together. Small affine-Q4 decode batches can use the installed MLX native matvec helper directly on packed cached buffers; other paths use native gather operations. File descriptors are reused. On macOS, `F_NOCACHE` avoids retaining a second copy of evicted expert data in the filesystem cache. Other operating systems currently use ordinary buffered reads. Counters report bytes returned by file reads, not independent measurements of physical device traffic.

A global byte-bounded cache defaults to LRU; `MLXSERVE_SSD_EXPERTS_CACHE_POLICY=lfu` selects frequency-based admission and eviction. Setting `MLXSERVE_SSD_EXPERTS_FREQUENCY_CAP=1` prevents a single long prefill from receiving more frequency credit per expert than one decode fetch. Cache bypass affects storage residency only: every routed expert is still computed. Entries larger than the budget are used transiently without caching. Selected experts are gathered in batches of up to 64 MiB of packed projection data (or one expert if larger); each batch completes before the next. This limits lazy graph retention and reduces GPU synchronizations compared with processing every expert separately. Stacked copies and active batch operands add temporary memory beyond the LRU budget.

The server also bounds retained free Metal buffers, defaulting to 128 MiB while paging is enabled, and restores the previous allocator cache limit on unload. Shared weights, activations, KV caches, Python and Metal overhead remain additional to the expert cache. Shared weights must fit RAM. A failed checkpoint switch preserves ownership of the previous cache and leaves that model usable.

Only checkpoints already converted to native stacked MLX expert layout are supported. Missing source names or mismatching shapes are rejected. Native SwitchGLU and SwitchMLP projections are supported; custom expert modules are not. Models whose sanitization changes expert values without changing their shapes have not been validated.

## Historical memory validation — 19 October 2026

Machine: Apple M3 Pro Mac (Mac15,6), 36 GiB physical RAM, external T7 Shield SSD. Runtime: MLX 0.32.4, mlx-lm 0.32.2. Other applications were running, so these are functional validation measurements, not controlled comparisons with another engine.

Checkpoint: `mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit`, revision `d8a069bfa8ae87d3d468412e1034acae19b5892b`.

| Requirement | Evidence |
| --- | --- |
| Weights exceed physical RAM | 44,844,286,500 checkpoint bytes = 41.76 GiB, versus 36 GiB physical RAM |
| Expert weights remain on SSD | 40.50 GiB of routed projections paged; about 1.26 GiB shared weights materialized at load |
| Expert cache is bounded | 4 GiB configured; 4,294,508,544 bytes retained, below 4,294,967,296 bytes |
| All experts remain available | 144 projections across 48 layers; 512 experts per layer, top-10 routing unchanged |
| Real inference completes | Two passes of 24 generated tokens; coherent French text; identical token sequences across passes |
| Physical memory remains below model size and RAM | Sampled peak process physical footprint 6.06 GiB; peak MLX allocations 5.38 GiB under an 8 GiB MLX limit |
| No growing swap during the measured two-pass run | System swap decreased by 0.94 GiB; sampled process compression peak 448.7 MiB |
| Actual SSD misses are served | 35,354 projection misses, 28,073 evictions, 20.85 GB of returned file data; uncached reads enabled |
| SoloMLX API works | Normal JSON and SSE both return `Paris`, HTTP 200, finish reason `stop`; SSE ends with `[DONE]` and usage |
| Prompt cache works alongside paging | Repeated API request serves 16 of 19 prompt tokens from KV cache |
| Numerical checks | Complete small Qwen3-MoE logits and Qwen3-Next prefill/cached-decode match resident references; real 80B gate/up/down projections are bit-identical for ten sampled expert IDs spanning 0–511 |
| Regression suite | 140 tests passed at the time (161 with the later kernel and cache tests) |

On the two-pass run, first-token latency was 65.14 s cold and 28.63 s on the second pass. Decode throughput, excluding first-token latency, was 1.21 and 1.27 tokens/s. The 24-token answers were intentionally capped, rather than claimed to be complete sentences. The API smoke uses a separate short question and produces the complete answer `Paris`; its repeated, warm response is not a general throughput measurement.

Raw evidence (JSON written by the scripts below: requirement checks, two-pass inference with routing counts and memory timeline, API smoke via FastAPI TestClient, sampled projection parity on the real 80B weights, Lightning 30B-A3B runs) is kept locally under `artifacts/`, which is git-ignored. The Lightning runs' swap growth happened under concurrent workloads and is not evidence of a swap-free run.

The 80B model cannot be loaded as a full resident reference within this Mac's RAM. Full-model resident parity is therefore established on small complete models, supplemented by sampled projection parity on the real checkpoint; a complete 80B reference comparison has not been performed.

## Activation

Use a local checkpoint directory, ideally on the internal SSD (a USB SSD is 5-8x slower here). Clients should send the same identifier, or omit `model` when calling this server directly. The settings below are the best measured server configuration (20.84 decode tok/s on 256/64, ~14.4 GiB peak footprint); lower `MLXSERVE_SSD_EXPERTS_CACHE_GB` if the machine has less free memory (8 GiB: 12.5 tok/s).

```sh
MLXSERVE_RUNTIME_BACKEND=mlx \
MLXSERVE_DEFAULT_MODEL="/absolute/path/to/native-stacked-MoE-checkpoint" \
MLXSERVE_SSD_EXPERTS_ENABLED=true \
MLXSERVE_SSD_EXPERTS_CACHE_GB=12 \
MLXSERVE_SSD_EXPERTS_CACHE_POLICY=lfu \
MLXSERVE_SSD_EXPERTS_FREQUENCY_CAP=1 \
MLXSERVE_SSD_EXPERTS_METAL_CACHE_MB=128 \
MLXSERVE_MOE_RESIDENT_EXPERTS=0 \
MLXSERVE_PREFILL_STEP_SIZE=1024 \
MLXSERVE_PROMPT_CACHE_GB=0.5 \
MLXSERVE_MTP=false \
.venv/bin/mlxserve serve
```

The fused Q4 SwiGLU kernel behind the 23.16 tok/s row is not exposed as a server setting yet (benchmark flag only, pending broader quality checks).

Alternatively configure a Hugging Face repo ID and set `HF_HUB_CACHE` to the SSD cache directory containing its downloaded snapshot. Keep the client and server identifiers consistent to avoid unnecessary model reloads. An uncached repo ID can trigger a large download.

Paging and static expert pruning cannot be combined. MTP interaction, long agent contexts (OpenCode-style 8k+ prompts), cancellation and multiple concurrent model identities have not been validated for this mode.

## Reproduce

Checkpoint: `mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit`, revision `d8a069bfa8ae87d3d468412e1034acae19b5892b` (download it first; the scripts never download).

Performance protocol (results table above):

```sh
.venv/bin/python scripts/bench_ssd_protocol.py \
  --model /absolute/path/to/checkpoint --prompt-file bench/ssd_prompts/english_systems.txt \
  --prompt-tokens 1024 --gen-tokens 16 --prefill-step 1024 \
  --cache-gb 12 --cache-policy lfu --frequency-cap 1 --memory-limit-gb 16 \
  --output artifacts/ssd_protocol_1024_12g.json
```

Historical memory validation and API smoke:

```sh
.venv/bin/python scripts/bench_ssd_experts.py \
  --model /absolute/path/to/checkpoint \
  --cache-gb 4 --memory-limit-gb 8 \
  --chat --prompt "Explique en deux phrases pourquoi le ciel est bleu." \
  --max-tokens 24 --rounds 2 --prefill-step-size 8 \
  --output artifacts/ssd_qwen80b_validation.json

.venv/bin/python scripts/smoke_ssd_api.py \
  --model /absolute/path/to/checkpoint \
  --output artifacts/ssd_qwen80b_api_validation.json

.venv/bin/python -m pytest -q
```

Both scripts require a local checkpoint and never automatically download one. The benchmark supports `--resident` for a separate reference process, only when the checkpoint fits the chosen memory limit. The MLX allocation limit is not a total process memory cap; Darwin `phys_footprint`, compression and system swap are sampled separately. Sampling can miss very short peaks. Retain operating-system headroom and use an idle machine for performance comparisons.

## Further work

Pack all projections of each expert contiguously, prefetch and overlap reads with GPU computation, expose the fused kernel once validated more broadly, and validate long agent contexts, cancellation, multiple concurrent model identities and MTP.

Janas is an architectural reference; this implementation does not copy its GPL source. Its documented cache-conditional routing can skip low-weight experts at small cache sizes, whereas this implementation computes every selected expert.
