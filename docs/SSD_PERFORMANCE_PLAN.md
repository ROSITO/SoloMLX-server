# SSD MoE performance log

Experiment log behind [SSD expert paging](SSD_EXPERT_PAGING.md). The `artifacts/*.json` files named below are the raw evidence; they are kept locally (git-ignored), not in the repository.

The initial memory feasibility milestone is insufficient for the requested result. The active objective is to reach at least Janas's published performance on an out-of-RAM model, with an explicit comparison of hardware, RAM budget, routing, precision and measurement protocol.

Reference (checked 19 October 2026): https://github.com/prabanta-dev/janas

- Qwen3-Next-80B: published headline 24 tokens/s without drafts and 62 prompt tokens/s, 1,024-token prompt and 16 generated tokens. The README also reports a 256/64 protocol over three rounds (about 22 generation tokens/s), and 37 tokens/s with drafts on a chat reply.
- CPU Core Ultra 9 185H, 32 GB RAM, integrated Arc GPU, approximately 5.7 GB/s internal NVMe, approximately 19 GB expert cache for the main out-of-RAM comparison.
- Small-cache results must distinguish exact top-10 routing from cache-conditional expert omission. With 4 GiB, features.md reports 20.7 tokens/s with all ten and 27.7 with its approximate routing. SoloMLX currently preserves all selected experts.
- The MLX checkpoint (44.84 GB) and Janas Q4_K_M checkpoint (48.4 GB) use different quantization layouts; a speed comparison is not a numerical equivalence claim.

Targets remain pending: at least the published non-draft generation/prefill rates on comparable token counts and warm-up, then the speculative-decoding rate with a separately validated drafting path. A short warm request or a memory-only test does not meet this performance objective.

## Measurement method

`scripts/bench_ssd_protocol.py` accepts benchmark prose as an external input, records hashes and exact token counts, prefills N-1 tokens, then times G single-token greedy forwards while ignoring EOS. KV is reset each round; the bounded expert cache is retained. One warm-up precedes three measured rounds. This follows the timing structure in Janas's `src/bench/main.c`; its source is not incorporated into SoloMLX.

Use 256/64 and 1024/16 cases. Record prefill, decode, I/O bytes and wall time, cache hit/miss counts, sampled physical footprint, compression and swap. Compare several original prose/code tasks as well as the reference benchmark input, and keep cold-start/first-token latency separate.

## Current experiments

1. Baseline: 4 GiB LRU, synchronous per-projection expert reads on T7: approximately 1.2 tokens/s on the short French chat sample.
2. Parallel reads on T7, same cache/sample: approximately 1.6 tokens/s; first-token latency falls substantially. `read_seconds` sums worker durations; `read_wall_seconds` measures the parallel interval.
3. Same projection implementation and same cache/sample on internal SSD: approximately 5.5 tokens/s. This isolates the storage change from the grouped-MLP change.
4. Grouped SwitchGLU/SwitchMLP on internal SSD: 5.50/6.39 generation tokens/s on the same short sample. These measurements differ from the fixed-token protocol below.
5. Native MLX Q4 matvec helper reading packed expert buffers directly, with an 8 GiB LFU cache and a wired-memory ceiling: fixed 256/64 protocol, one warm-up then three rounds, mean prefill **35.51 tokens/s**, mean decode **6.78 tokens/s** (6.49, 6.88, 6.97). Peak sampled physical footprint 10.32 GiB; system swap growth zero. Evidence: `artifacts/ssd_qwen80b_protocol_8g_lfu_native_q4.json`. This remains well below the performance objective.

On the final measured round of that run, decode cache hit rate was 77.1%, with about 194 MB returned by expert reads per generated token. Parallel read wall time was 4.83 seconds over 64 tokens; routing synchronization time was 3.03 seconds and includes preceding deferred GPU work. This identifies both I/O and computation/synchronization as remaining limits.

Teacher-forced native-gather versus packed-buffer Q4 evaluation agreed on 63/64 greedy choices, mean KL divergence 0.00168, maximum 0.01742 (`artifacts/ssd_native_q4_quality_64.json`). This is a preliminary numerical check, not full-model equivalence. LFU is selectable in the server with `MLXSERVE_SSD_EXPERTS_CACHE_POLICY=lfu`; the default remains LRU. An optional frequency cap (`MLXSERVE_SSD_EXPERTS_FREQUENCY_CAP=1`) can prevent long prefills from dominating admission scores.

Storage screening used 128 MiB temporary files, uncached reads, and removed only its own scratch files. UGREEN with the fast cable: about 890 MB/s sequential. T7 with the slow cable: about 40 MB/s, negotiating USB 2.0. Internal SSD: about 5 GB/s sequential in this short test. These are not sustained whole-model bandwidth certifications.

For the internal-SSD runs the native checkpoint was copied to `~/.mlxserve/ssd-bench/Qwen3-Next-80B-A3B-Instruct-4bit` (about 42 GiB of internal storage) using bounded uncached I/O.

The 1024/16 protocol with 8 GiB LFU and prefill step 1024 achieved mean prefill **72.72 tokens/s** and decode **6.15 tokens/s**, one warm-up plus three measured rounds. All rounds generated the same tokens. Evidence: `artifacts/ssd_qwen80b_protocol_1024_lfu.json`. Prefill exceeds the published headline in this configuration; generation does not. Different hardware, quantization and expert-cache budgets still apply.

With frequency credit capped at one per expert fetch and unnecessary stack graphs removed from the direct Q4 path, the 256/64 benchmark at 8 GiB cache improved to **12.51 generation tokens/s** (11.99, 12.83, 12.72) and 35.79 prefill tokens/s. Output tokens match the preceding native-helper run in every round. The final decode round hit 92.8% of expert requests, reading about 61.5 MB/token instead of 194 MB/token. Peak physical footprint was 10.34 GiB, and system swap decreased by 8 MiB. Evidence: `artifacts/ssd_qwen80b_protocol_8g_lfu_cap1.json`. These results combine the admission adjustment and host graph simplification; their individual contributions are not isolated.

At 12 GiB LFU cache with frequency cap one, the same 256/64 protocol reached **20.84 generation tokens/s** (20.84, 20.92, 20.75), with identical tokens to the 8 GiB cap-one run, peak physical footprint 14.37 GiB and zero swap growth. Decode hit rate reached 98.61%, reading 11.81 MB/token. Evidence: `artifacts/ssd_qwen80b_protocol_12g_lfu_cap1.json`.

An experimental fused native Q4 up/gate/SwiGLU kernel reached **22.28 generation tokens/s** (22.57, 22.83, 21.43) at the same cache budget. It is disabled by default, pending broader quality validation. Its generated trajectory differs from the non-fused run, so this is not a fixed-trajectory kernel speed comparison. Evidence: `artifacts/ssd_qwen80b_protocol_12g_fused.json`. The 24-token/s generation objective remains unmet.

The fused kernel's teacher-forced 64-position check agreed on 63/64 greedy choices with native gather, mean KL 0.00174 and maximum KL 0.02098 (`artifacts/ssd_fused_q4_quality_64.json`). This is preliminary validation and the kernel remains opt-in. Subsequent direct kernels can broadcast a single decode input vector across experts instead of materializing repeated input rows; numerical tests cover one-token and batched inputs for all three tested dtypes.

Broadcasting the one-token input removed repeated input materialization but did not establish a speed gain: 21.84 generation tokens/s versus the previous 22.28, with identical generated tokens (`artifacts/ssd_qwen80b_protocol_12g_broadcast.json`). Treat this as within-run/environment variation, not a throughput improvement. An opt-in Qwen3-Next shared-expert scheduling experiment is being measured separately; it starts shared GPU computation before paged-expert CPU reads and preserves logits in the reduced complete-model tests.

The shared-expert overlap experiment was rejected: 19.28 generation tokens/s, below the preceding 21.84, with identical generated tokens. Its wrapper and benchmark flag were removed; the measured source is preserved alongside `artifacts/ssd_qwen80b_protocol_12g_overlap.json` in the `.sources.json` snapshot. Its internal routing timer excludes the extra synchronization outside the paged MLP and must not be compared directly with earlier routing counters. Future protocol runs save their complete local benchmark/runtime source snapshots along with hashes.

On 1024/16, the 12 GiB capped LFU plus fused kernel measured **84.84 prefill tokens/s** and **22.23 generation tokens/s** (21.80, 21.86, 23.02), with one warm-up and three rounds, zero swap growth and 14.86 GiB peak physical footprint. Decode returned only 2.43 MB/token from SSD on each measured round. Evidence and complete measured sources: `artifacts/ssd_qwen80b_protocol_1024_12g_fused.json` and its `.sources.json` companion. Generation remains below the 24-token/s target even though prefill exceeds 62.

Avoiding the redundant scatter for unique top-k experts of a single token measured **23.16 generation tokens/s** (23.51, 23.40, 22.56) and **84.31 prefill tokens/s** on 1024/16, with identical tokens to the preceding fused run. Evidence and measured source snapshot: `artifacts/ssd_qwen80b_protocol_1024_direct_output.json` and `.sources.json`. This modest gain still does not meet 24 tokens/s on the three-round mean. The current full regression suite passes 161 tests; the fused kernel remains benchmark opt-in.

## Status

The interleaved dual-projection experiment measured 22.84 generation tokens/s on 1024/16, below the preceding 23.16. It was removed from the runtime; its results and exact source are retained in `artifacts/ssd_qwen80b_protocol_1024_dual.json` and its `.sources.json` companion. The retained runtime uses the earlier fused helper.

The 24 tok/s objective is **not achieved**: the best measured three-round mean is 23.16 generation tokens/s (benchmark-only fused kernel); the server's default kernels reach 20.84 at the same cache budget. Prefill reached 84.31 tokens/s. The production backend supports capped LFU; the fused helper is still a benchmark experiment, disabled by default. Broader quality validation, the drafting target, and performance on additional tasks remain outstanding. Three original workloads were prepared under `bench/ssd_prompts/`; they have not been benchmarked. The optimized non-fused API smoke passed JSON and SSE, including prompt-cache reuse, in `artifacts/ssd_qwen80b_api_optimized.json`.

## Packed experts (one read per expert)

Disk-only, Qwen3-Next layers 4-7, internal SSD, page cache evicted: 9 field reads per expert 2.86-3.00 GB/s, one packed read per expert 4.40-5.36 GB/s (x1.56 decode batches, x1.86 prefill batches, x1.88 whole layer). Earlier "per-field" figures of 8-9 GB/s were served from the page cache. See [SSD expert paging](SSD_EXPERT_PAGING.md#packed-experts-one-read-per-expert). End-to-end effect on prefill and decode is pending a run with the full packed checkpoint.
