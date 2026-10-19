"""Experimental exact MoE paging: safetensors rows -> bounded MLX expert cache.

Only native stacked SwitchLinear checkpoints are supported. No experts are pruned.
Active experts are gathered in bounded batches; this remains an experimental kernel.
"""
from __future__ import annotations

import json
import heapq
import math
import os
import struct
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import mlx.core as mx
import mlx.nn as nn

_DTYPES = {"F32": mx.float32, "F16": mx.float16, "BF16": mx.bfloat16,
           "U32": mx.uint32, "I32": mx.int32, "U16": mx.uint16,
           "I16": mx.int16, "U8": mx.uint8, "I8": mx.int8}


PACKED_GLOB = "ssd_experts*.safetensors"  # written by scripts/pack_ssd_experts.py
PACKED_SUFFIX = ".ssd_packed"


class ExpertStore:
    """Index headers only; pread copies just one expert's contiguous tensor row.

    With a packed file (one U8 row per expert holding all its projections' weight/scales/biases back to
    back), a miss costs one read instead of one per field (9 for a quantized GLU).
    """

    def __init__(self, directory: str | Path, packed_dir: str | Path | None = None):
        self.tensors = {}
        self.packed = {}  # switch-module prefix -> (path, file offset, row bytes, [(field, offset, nbytes)])
        self.bytes_read = 0
        self.read_seconds = 0.0
        self._fds = {}
        self._lock = Lock()
        self._io_pool = None
        self.read_wall_seconds = 0.0
        self.read_calls = 0
        self.uncached_reads = sys.platform == 'darwin'
        for path in sorted(Path(directory).glob("model*.safetensors")):
            with path.open("rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                if header_size > 100_000_000:
                    raise ValueError(f"Invalid safetensors header: {path}")
                header = json.loads(f.read(header_size))
            for name, meta in header.items():
                if name != "__metadata__":
                    if name in self.tensors:
                        raise ValueError(f"Duplicate tensor: {name}")
                    self.tensors[name] = (path, 8 + header_size, meta)
        self._index_packed(Path(packed_dir) if packed_dir else Path(directory))

    def _index_packed(self, directory: Path):
        for path in sorted(directory.glob(PACKED_GLOB)):
            with path.open("rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_size))
            fields = json.loads((header.get("__metadata__") or {}).get("ssd_fields", "{}"))
            for name, meta in header.items():
                if not name.endswith(PACKED_SUFFIX):
                    continue
                prefix = name[: -len(PACKED_SUFFIX)]
                begin, _ = meta["data_offsets"]
                experts, row = meta["shape"]
                layout = [tuple(x) for x in fields[prefix]]
                for field, offset, nbytes in layout:  # must describe the original checkpoint exactly
                    original = self.tensors.get(prefix + "." + field)
                    if original is None:
                        raise ValueError(f"Packed field {prefix}.{field} not in checkpoint")
                    o_begin, o_end = original[2]["data_offsets"]
                    if original[2]["shape"][0] != experts or (o_end - o_begin) // experts != nbytes:
                        raise ValueError(f"Packed field {prefix}.{field} does not match checkpoint")
                if sum(n for _, _, n in layout) != row:
                    raise ValueError(f"Packed row size mismatch: {prefix}")
                self.packed[prefix] = (path, 8 + header_size + begin, row, layout)

    def _read_packed(self, prefix: str, expert: int):
        path, base, row_bytes, _ = self.packed[prefix]
        start = time.perf_counter()
        data = os.pread(self._fd(path), row_bytes, base + expert * row_bytes)
        if len(data) != row_bytes:
            raise ValueError(f"Truncated packed experts: {prefix}")
        with self._lock:
            self.bytes_read += len(data)
            self.read_seconds += time.perf_counter() - start
            self.read_calls += 1
        return data

    def read_packed_many(self, prefix: str, experts):
        """One read per expert, in parallel; returns raw rows."""
        if self._io_pool is None:
            self._io_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix='expert-io')
        start = time.perf_counter()
        rows = list(self._io_pool.map(lambda e: self._read_packed(prefix, e), experts))
        self.read_wall_seconds += time.perf_counter() - start
        return rows

    def _fd(self, path):
        with self._lock:
            if path not in self._fds:
                fd = os.open(path, os.O_RDONLY)
                try:
                    if self.uncached_reads:
                        import fcntl
                        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                except BaseException:
                    os.close(fd)
                    raise
                self._fds[path] = fd
            return self._fds[path]

    def _read_raw(self, name: str, expert: int):
        path, base, meta = self.tensors[name]
        shape = meta["shape"]
        if not shape or not 0 <= expert < shape[0]:
            raise ValueError(f"Invalid expert {expert} for {name}")
        dtype = _DTYPES.get(meta["dtype"])
        if dtype is None:
            raise ValueError(f"Unsupported SSD tensor dtype: {meta['dtype']}")
        begin, end = meta["data_offsets"]
        row_bytes = math.prod(shape[1:]) * dtype.size
        if end - begin != shape[0] * row_bytes:
            raise ValueError(f"Invalid tensor size: {name}")
        start = time.perf_counter()
        data = os.pread(self._fd(path), row_bytes, base + begin + expert * row_bytes)
        if len(data) != row_bytes:
            raise ValueError(f"Truncated tensor: {name}")
        with self._lock:
            self.bytes_read += len(data)
            self.read_seconds += time.perf_counter() - start
            self.read_calls += 1
        return data, dtype, shape[1:]

    @staticmethod
    def _array(raw):
        data, dtype, shape = raw
        return mx.array(memoryview(data), dtype=mx.uint8).view(dtype).reshape(shape)

    def read(self, name: str, expert: int) -> mx.array:
        return self._array(self._read_raw(name, expert))

    def read_raw_many(self, requests):
        """Parallel file I/O only; all MLX arrays are created on the caller's GPU thread."""
        if self._io_pool is None:
            self._io_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix='expert-io')
        start = time.perf_counter()
        futures = [self._io_pool.submit(self._read_raw, name, expert) for name, expert in requests]
        raw, error = [], None
        for future in futures:
            try:
                raw.append(future.result())
            except BaseException as exc:
                error = error or exc
        self.read_wall_seconds += time.perf_counter() - start
        if error is not None:
            raise error
        return raw

    def read_many(self, requests):
        return [self._array(item) for item in self.read_raw_many(requests)]

    def close(self):
        if self._io_pool is not None:
            self._io_pool.shutdown(wait=True)
            self._io_pool = None
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __del__(self):
        self.close()


class ExpertCache:
    """Global byte-bounded LRU. In-flight operands and Metal buffers are additional."""

    def __init__(self, store: ExpertStore, max_bytes: int, policy: str = 'lru', frequency_cap: int | None = None):
        if max_bytes < 0:
            raise ValueError("SSD expert cache size must be nonnegative")
        self.store = store
        if policy not in ('lru', 'lfu'):
            raise ValueError('Expert cache policy must be lru or lfu')
        if frequency_cap is not None and frequency_cap < 1:
            raise ValueError("Frequency cap must be positive")
        self.frequency_cap = frequency_cap
        self.policy = policy
        self._frequencies = {}
        self._heap = []
        self._sequence = self._accesses = 0
        self.admission_bypasses = 0
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.resident_bytes = 0
        self.hits = self.misses = self.evictions = 0
        self.paged_projections = 0
        self.paged_weight_bytes = 0
        self.experts_per_projection = []
        self.routing_seconds = 0.0
        self.fetch_seconds = 0.0
        self.expert_compute_seconds = 0.0
        self.grouped_calls = 0
        self.direct_q4_calls = 0
        self.fused_q4_calls = 0

    def get(self, prefix: str, names: tuple[str, ...], expert: int):
        return self.get_many(prefix, names, [expert])[0]

    def _rebuild_heap(self):
        self._heap = [(self._frequencies[k], i, k) for i, k in enumerate(self.entries)]
        heapq.heapify(self._heap)
        self._sequence = len(self._heap)

    def _touch(self, key, count):
        if self.policy != 'lfu':
            return
        if self.frequency_cap is not None:
            count = min(count, self.frequency_cap)
        self._frequencies[key] = self._frequencies.get(key, 0) + count
        self._accesses += count
        if key in self.entries:
            self._sequence += 1
            heapq.heappush(self._heap, (self._frequencies[key], self._sequence, key))
        if self._accesses >= 1_000_000:
            self._frequencies = {k: max(1, v // 2) for k, v in self._frequencies.items()}
            self._accesses = 0
            self._rebuild_heap()
        elif len(self._heap) > 4 * len(self.entries) + 1024:
            self._rebuild_heap()

    def _victim(self):
        if self.policy == 'lru':
            return next(iter(self.entries))
        while self._heap:
            score, _, key = self._heap[0]
            if key in self.entries and self._frequencies[key] == score:
                return key
            heapq.heappop(self._heap)
        self._rebuild_heap()
        return self._heap[0][2]

    def get_many(self, prefix: str, names: tuple[str, ...], experts, counts=None):
        loaded, missing = {}, []
        for expert in experts:
            key = (prefix, expert)
            self._touch(key, counts.get(expert, 1) if counts else 1)
            if key in self.entries:
                self.hits += 1
                self.entries.move_to_end(key)
                loaded[expert] = self.entries[key]
            else:
                self.misses += 1
                missing.append(expert)
        size = sum((self.store.tensors[prefix + '.' + n][2]['data_offsets'][1]
                    - self.store.tensors[prefix + '.' + n][2]['data_offsets'][0])
                   // self.store.tensors[prefix + '.' + n][2]['shape'][0] for n in names)
        groups = {}
        for name in names:
            groups.setdefault(name.rsplit('.', 1)[0] if '.' in name else '', []).append(name)
        packed = self.store.packed.get(prefix)
        if packed is not None and [f for f, _, _ in packed[3]] != list(names):
            packed = None  # layout differs from what the caller computes with: read field by field
        # Missing rows are bounded by the caller's batch budget. One GPU evaluation
        # for the complete batch replaces a synchronization per expert.
        if packed is not None and missing:
            rows = self.store.read_packed_many(prefix, missing)
        else:
            rows = self.store.read_raw_many([(prefix + '.' + n, e) for e in missing for n in names]) if missing else []
        for index, expert in enumerate(missing):
            value = {}
            if packed is not None:
                # One allocation for the whole expert; projections and fields are zero-copy views of it.
                blob = mx.array(memoryview(rows[index]), dtype=mx.uint8)
                spans = {f: (o, n) for f, o, n in packed[3]}
                for group, fields in groups.items():
                    start, _ = spans[fields[0]]
                    end = spans[fields[-1]][0] + spans[fields[-1]][1]
                    value['_packed_' + group] = blob[start:end]
                    for n in fields:
                        meta = self.store.tensors[prefix + '.' + n][2]
                        o, nb = spans[n]
                        value[n] = blob[o:o + nb].view(_DTYPES[meta['dtype']]).reshape(meta['shape'][1:])
            else:
                raw = dict(zip(names, rows[index * len(names):(index + 1) * len(names)]))
                for group, fields in groups.items():
                    # Keep a single contiguous allocation per projection in the cache.
                    # Field arrays are zero-copy views of the same packed allocation.
                    blob = mx.array(memoryview(b''.join(raw[n][0] for n in fields)), dtype=mx.uint8)
                    value['_packed_' + group] = blob
                    offset = 0
                    for n in fields:
                        data, dtype, shape = raw[n]
                        value[n] = blob[offset:offset + len(data)].view(dtype).reshape(shape)
                        offset += len(data)
            key = (prefix, expert)
            admit = size <= self.max_bytes
            if admit and self.policy == 'lfu' and self.entries and self.resident_bytes + size > self.max_bytes:
                admit = self._frequencies[key] > self._frequencies[self._victim()]
            if admit:
                while self.entries and self.resident_bytes + size > self.max_bytes:
                    old = self.entries.pop(self._victim())
                    self.resident_bytes -= sum(v.nbytes for n, v in old.items() if not n.startswith('_'))
                    self.evictions += 1
                self.entries[key] = value
                self.resident_bytes += size
                if self.policy == 'lfu':
                    self._sequence += 1
                    heapq.heappush(self._heap, (self._frequencies[key], self._sequence, key))
            else:
                self.admission_bypasses += 1
            loaded[expert] = value
        return [loaded[e] for e in experts]

    def clear(self):
        self.entries.clear()
        self.resident_bytes = 0
        self._heap.clear()

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, evictions=self.evictions,
                    policy=self.policy, frequency_cap=self.frequency_cap, admission_bypasses=self.admission_bypasses,
                    resident_bytes=self.resident_bytes, max_bytes=self.max_bytes,
                    bytes_read=self.store.bytes_read, read_seconds=self.store.read_seconds,
                    read_wall_seconds=self.store.read_wall_seconds,
                    read_calls=self.store.read_calls,
                    routing_seconds=self.routing_seconds, fetch_seconds=self.fetch_seconds,
                    expert_compute_seconds=self.expert_compute_seconds, grouped_calls=self.grouped_calls,
                    direct_q4_calls=self.direct_q4_calls, fused_q4_calls=self.fused_q4_calls,
                    uncached_reads=self.store.uncached_reads, packed_modules=len(self.store.packed),
                    paged_projections=self.paged_projections,
                    paged_weight_bytes=self.paged_weight_bytes,
                    min_experts_per_projection=min(self.experts_per_projection, default=0),
                    max_experts_per_projection=max(self.experts_per_projection, default=0))


class SSDLinear(nn.Module):
    def __init__(self, original, prefix: str, cache: ExpertCache):
        super().__init__()
        # Keep no reference to original or its full expert tensors.
        self._cache = cache
        self._prefix = prefix
        self._names = tuple(n for n in ('weight', 'scales', 'biases', 'bias', 'global_scale')
                            if n in original and original[n] is not None)
        self._quantized = 'scales' in self._names
        self._output_dims = original.weight.shape[1]
        self._num_experts = original.weight.shape[0]
        self._input_dims = original.input_dims
        self._batch_bytes = 64 * 1024**2
        if self._quantized:
            self._bits, self._group_size, self._mode = original.bits, original.group_size, original.mode
        for name in self._names:
            key = prefix + '.' + name
            if key not in cache.store.tensors:
                raise ValueError(f"SSD paging requires native stacked checkpoint tensor: {key}")
            if cache.store.tensors[key][2]['shape'] != list(original[name].shape):
                raise ValueError(f"Checkpoint tensor was transformed during load: {key}")
        self._packed_layout = {name: cache.store.tensors[prefix + '.' + name][2] for name in self._names}
        self._expert_bytes = sum(
            cache.store.tensors[prefix + '.' + name][2]['data_offsets'][1]
            - cache.store.tensors[prefix + '.' + name][2]['data_offsets'][0]
            for name in self._names
        ) // self._num_experts

    @property
    def input_dims(self):
        return self._input_dims

    @property
    def output_dims(self):
        return self._output_dims

    @property
    def num_experts(self):
        return self._num_experts

    def __call__(self, x, indices, sorted_indices=False):
        # Resolve routing before reading any weights; batch work for each unique expert.
        mx.eval(indices)
        ids = indices.reshape(-1).tolist()
        positions = {}
        for i, expert in enumerate(ids):
            positions.setdefault(expert, []).append(i)
        shape = (*indices.shape, 1, x.shape[-1])
        inputs = mx.broadcast_to(x, shape).reshape(-1, 1, x.shape[-1])
        result = mx.zeros((len(ids), 1, self._output_dims), dtype=x.dtype)
        # Bound staging independently of prompt length and expert count. Each batch
        # contains packed expert rows plus their stacked copy, in addition to LRU.
        per_batch = max(1, self._batch_bytes // max(1, self._expert_bytes))
        experts = list(positions)
        for start in range(0, len(experts), per_batch):
            batch = experts[start:start + per_batch]
            loaded = self._cache.get_many(self._prefix, self._names, batch,
                                           counts={e: len(positions[e]) for e in batch})
            tensors = {name: mx.stack([item[name] for item in loaded]) for name in self._names}
            rows = [r for e in batch for r in positions[e]]
            local = mx.array([i for i, e in enumerate(batch) for _ in positions[e]], dtype=mx.int32)
            row_indices = mx.array(rows, dtype=mx.int32)
            selected = inputs[row_indices]
            if self._quantized:
                y = mx.gather_qmm(selected, tensors['weight'], tensors['scales'],
                    tensors.get('biases'), rhs_indices=local, transpose=True,
                    group_size=self._group_size, bits=self._bits, mode=self._mode,
                    sorted_indices=True, **({'global_scale': tensors['global_scale']}
                    if 'global_scale' in tensors else {}))
            else:
                y = mx.gather_mm(selected, tensors['weight'].swapaxes(-1, -2),
                    rhs_indices=local, sorted_indices=True)
            if 'bias' in tensors:
                y = y + mx.expand_dims(tensors['bias'][local], -2)
            result = result.at[row_indices].add(y.astype(x.dtype))
            # Finish operations before the next eviction; bound lazy graph retention.
            mx.eval(result)
            del loaded, tensors, selected, y
        return result.reshape(*indices.shape, 1, self._output_dims)


class SSDGroupedSwitch(nn.Module):
    """One routing sync, one expert fetch and one graph evaluation for the full MLP."""

    def __init__(self, original, prefix, cache, glu=True):
        super().__init__()
        self._cache, self._prefix, self._glu = cache, prefix, glu
        self._projections = ('up_proj', 'gate_proj', 'down_proj') if glu else ('fc1', 'fc2')
        self.activation = original.activation
        for name in self._projections:
            setattr(self, name, SSDLinear(getattr(original, name), prefix + '.' + name, cache))
        first, last = getattr(self, self._projections[0]), getattr(self, self._projections[-1])
        self._num_experts = first.num_experts
        self._output_dims = last.output_dims
        self._names = tuple(p + '.' + n for p in self._projections for n in getattr(self, p)._names)
        self._expert_bytes = sum(getattr(self, p)._expert_bytes for p in self._projections)
        self._batch_bytes = 64 * 1024**2
        self._direct_matvec = True
        self._fused_matvec = False

    def _project(self, projection, x, indices, loaded):
        meta = getattr(self, projection)
        if (self._direct_matvec and meta._quantized and meta._bits == 4 and meta._mode == 'affine'
                and meta._names == ('weight', 'scales', 'biases') and indices.size <= 32 and len(loaded) <= 24
                and meta._packed_layout['scales']['dtype'] == meta._packed_layout['biases']['dtype']
                and x.dtype == _DTYPES[meta._packed_layout['scales']['dtype']]):
            from mlxserve.runtime.ssd_qmm import packed_q4_matvec
            types = {'F32': 'float', 'F16': 'half', 'BF16': 'bfloat16_t'}
            scale_type = types.get(meta._packed_layout['scales']['dtype'])
            if scale_type is not None:
                sizes = {n: (v['data_offsets'][1] - v['data_offsets'][0]) // meta.num_experts
                         for n, v in meta._packed_layout.items()}
                direct = packed_q4_matvec(x, indices, [v['_packed_' + projection] for v in loaded],
                    input_dims=meta.input_dims, output_dims=meta.output_dims,
                    group_size=meta._group_size, scale_type=scale_type,
                    scale_offset=sizes['weight'], bias_offset=sizes['weight'] + sizes['scales'])
                if direct is not None:
                    self._cache.direct_q4_calls += 1
                    return direct
        values = {n: mx.stack([v[projection + '.' + n] for v in loaded]) for n in meta._names}
        if meta._quantized:
            out = mx.gather_qmm(x, values['weight'], values['scales'], values.get('biases'),
                rhs_indices=indices, transpose=True, group_size=meta._group_size,
                bits=meta._bits, mode=meta._mode, sorted_indices=True,
                **({'global_scale': values['global_scale']} if 'global_scale' in values else {}))
        else:
            out = mx.gather_mm(x, values['weight'].swapaxes(-1, -2),
                              rhs_indices=indices, sorted_indices=True)
        if 'bias' in values:
            out = out + mx.expand_dims(values['bias'][indices], -2)
        return out

    def _fused_swiglu(self, x, indices, loaded):
        from mlx_lm.models.switch_layers import SwiGLU
        if not (self._fused_matvec and self._direct_matvec and self._glu
                and type(self.activation) is SwiGLU and indices.size <= 32 and len(loaded) <= 12):
            return None
        up, gate = self.up_proj, self.gate_proj
        if not (up._quantized and gate._quantized and up._bits == gate._bits == 4
                and up._mode == gate._mode == 'affine' and up._group_size == gate._group_size
                and up._names == gate._names == ('weight', 'scales', 'biases')
                and all(up._packed_layout[n]['shape'] == gate._packed_layout[n]['shape']
                        and up._packed_layout[n]['dtype'] == gate._packed_layout[n]['dtype']
                        and (up._packed_layout[n]['data_offsets'][1] - up._packed_layout[n]['data_offsets'][0])
                        == (gate._packed_layout[n]['data_offsets'][1] - gate._packed_layout[n]['data_offsets'][0])
                        for n in up._names)
                and x.dtype == _DTYPES[up._packed_layout['scales']['dtype']]
                and up._packed_layout['scales']['dtype'] == up._packed_layout['biases']['dtype']):
            return None
        from mlxserve.runtime.ssd_qmm import packed_q4_swiglu
        sizes = {n: (v['data_offsets'][1] - v['data_offsets'][0]) // up.num_experts
                 for n, v in up._packed_layout.items()}
        hidden = packed_q4_swiglu(x, indices, [v['_packed_up_proj'] for v in loaded],
            [v['_packed_gate_proj'] for v in loaded], input_dims=up.input_dims,
            output_dims=up.output_dims, group_size=up._group_size,
            scale_offset=sizes['weight'], bias_offset=sizes['weight'] + sizes['scales'])
        if hidden is not None:
            self._cache.direct_q4_calls += 2
            self._cache.fused_q4_calls += 1
        return hidden

    def __call__(self, x, indices):
        started = time.perf_counter()
        mx.eval(indices)
        positions = {}
        for i, expert in enumerate(indices.reshape(-1).tolist()):
            positions.setdefault(expert, []).append(i)
        self._cache.routing_seconds += time.perf_counter() - started
        self._cache.grouped_calls += 1
        inputs = mx.broadcast_to(mx.expand_dims(x, (-2, -3)),
            (*indices.shape, 1, x.shape[-1])).reshape(-1, 1, x.shape[-1])
        result = mx.zeros((indices.size, 1, self._output_dims), dtype=x.dtype)
        per_batch = max(1, self._batch_bytes // max(1, self._expert_bytes))
        experts = list(positions)
        for start in range(0, len(experts), per_batch):
            batch = experts[start:start + per_batch]
            fetch_started = time.perf_counter()
            loaded = self._cache.get_many(self._prefix, self._names, batch,
                                           counts={e: len(positions[e]) for e in batch})
            self._cache.fetch_seconds += time.perf_counter() - fetch_started
            compute_started = time.perf_counter()
            rows = mx.array([r for e in batch for r in positions[e]], dtype=mx.int32)
            local = mx.array([i for i, e in enumerate(batch) for _ in positions[e]], dtype=mx.int32)
            # All experts of one decode token consume the same input vector.
            # Direct kernels broadcast it without a gather or repeated copies.
            selected = x.reshape(1, 1, x.shape[-1]) if x.size == x.shape[-1] else inputs[rows]
            if self._glu:
                hidden = self._fused_swiglu(selected, local, loaded)
                if hidden is None:
                    up = self._project('up_proj', selected, local, loaded)
                    gate = self._project('gate_proj', selected, local, loaded)
                    hidden = self.activation(up, gate)
                out = self._project('down_proj', hidden, local, loaded)
            else:
                hidden = self.activation(self._project('fc1', selected, local, loaded))
                out = self._project('fc2', hidden, local, loaded)
            if (x.size == x.shape[-1] and start == 0 and len(batch) == len(experts)
                    and len(experts) == indices.size):
                # Unique top-k experts of one token already follow routing order.
                result = out.astype(x.dtype)
            else:
                result = result.at[rows].add(out.astype(x.dtype))
            # The next layer's routing synchronization consumes the final batch.
            # Only intermediate batches need an explicit barrier to bound staging.
            if start + per_batch < len(experts):
                mx.eval(result)
            self._cache.expert_compute_seconds += time.perf_counter() - compute_started
            del loaded, selected, hidden, out
        return result.reshape(*indices.shape, self._output_dims)


def attach_ssd_experts(model, directory: str | Path, cache_bytes: int, *, group_experts: bool = True, cache_policy: str = 'lru', frequency_cap: int | None = None, packed_dir: str | Path | None = None) -> ExpertCache:
    from mlx_lm.models.switch_layers import SwitchLinear, QuantizedSwitchLinear, SwitchGLU, SwitchMLP

    cache = ExpertCache(ExpertStore(directory, packed_dir), cache_bytes, policy=cache_policy, frequency_cap=frequency_cap)
    replacements = []
    grouped = []
    for path, module in model.named_modules():
        if any(path.startswith(parent + '.') for parent in grouped):
            continue
        if group_experts and type(module) in (SwitchGLU, SwitchMLP):
            replacements.append((path, SSDGroupedSwitch(module, path, cache, isinstance(module, SwitchGLU))))
            grouped.append(path)
        elif isinstance(module, (SwitchLinear, QuantizedSwitchLinear)):
            replacements.append((path, SSDLinear(module, path, cache)))
    if not replacements:
        raise ValueError("No supported SwitchLinear experts found; SSD paging was not enabled")
    cache.paged_projections = sum(len(m._projections) if isinstance(m, SSDGroupedSwitch) else 1 for _, m in replacements)
    cache.experts_per_projection = [m._num_experts for _, m in replacements
                                   for _ in range(len(m._projections) if isinstance(m, SSDGroupedSwitch) else 1)]
    cache.paged_weight_bytes = sum(m._expert_bytes * m._num_experts for _, m in replacements)
    # Validate every module before mutating the model.
    from mlx.utils import tree_unflatten
    model.update_modules(tree_unflatten(replacements))
    mx.eval(model.parameters())  # only shared/dense parameters remain
    mx.clear_cache()
    return cache

