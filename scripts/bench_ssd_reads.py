"""Disk-only benchmark: per-field expert reads vs one read per packed expert. Loads no model.

    python scripts/bench_ssd_reads.py --checkpoint /path/to/checkpoint --packed-dir /path/to/packed-dir

Both methods read the same randomly chosen experts with the server's ExpertStore (8 I/O workers,
F_NOCACHE on macOS). F_NOCACHE does not keep pages out of the cache completely, so on macOS every
file is evicted from the page cache (msync MS_INVALIDATE) before each method's run, and no expert
is read twice within a run.
"""
import argparse
import ctypes
import mmap
import os
import random
import sys
import statistics
import time
from pathlib import Path

from mlxserve.runtime.ssd_experts import ExpertStore


def evict(paths) -> None:
    """Drop the files' pages from the macOS unified buffer cache (no sudo needed)."""
    if sys.platform != "darwin":
        return
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        size = os.fstat(fd).st_size
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        os.close(fd)
        libc.msync(ctypes.c_void_p(addr), size, 0x0002)  # MS_INVALIDATE
        libc.munmap(ctypes.c_void_p(addr), size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--packed-dir", type=Path, required=True)
    ap.add_argument("--batches", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    store = ExpertStore(a.checkpoint, a.packed_dir)
    prefixes = sorted(store.packed)
    rng = random.Random(a.seed)
    print(f"{len(prefixes)} packed modules, {store.packed[prefixes[0]][2] / 2**20:.2f} MiB per expert, "
          f"{len(store.packed[prefixes[0]][3])} fields per expert, uncached reads: {store.uncached_reads}")

    def run(method, batch):
        prefix, experts = batch
        if method == "per-field":
            fields = [f for f, _, _ in store.packed[prefix][3]]
            store.read_raw_many([(f"{prefix}.{f}", e) for e in experts for f in fields])
        else:
            store.read_packed_many(prefix, experts)

    files = sorted({str(store.tensors[n][0]) for n in store.tensors} | {str(v[0]) for v in store.packed.values()})
    for label, size in (("decode: 10 experts / layer", 10), ("prefill batch: 37 experts (64 MiB)", 37),
                        ("whole layer: 512 experts", 512)):
        # Unique (layer, expert) pairs: nothing is read twice in a run.
        if size == 512:
            batches = [(p, list(range(512))) for p in prefixes]
        else:
            per_layer = {p: rng.sample(range(512), 512) for p in prefixes}  # shuffled, consumed without reuse
            batches = []
            while len(batches) < a.batches and any(len(v) >= size for v in per_layer.values()):
                p = rng.choice([p for p, v in per_layer.items() if len(v) >= size])
                batches.append((p, [per_layer[p].pop() for _ in range(size)]))
        count = len(batches)
        results = {"per-field": [], "packed": []}
        for method in ("per-field", "packed"):
            evict(files)
            for batch in batches:
                t = time.perf_counter()
                run(method, batch)
                results[method].append(time.perf_counter() - t)
        row = store.packed[prefixes[0]][2]
        print(f"\n{label} ({count} batches)")
        for method, times in results.items():
            total = sum(times)
            print(f"  {method:9}  median {1000 * statistics.median(times):7.2f} ms/batch   "
                  f"{size * row * count / total / 1e9:5.2f} GB/s")
        print(f"  speedup x{statistics.median(results['per-field']) / statistics.median(results['packed']):.2f}")
    store.close()


if __name__ == "__main__":
    main()
