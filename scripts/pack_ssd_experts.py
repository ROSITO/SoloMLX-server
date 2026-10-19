"""Write a packed copy of a native MLX MoE checkpoint's experts: one contiguous row per expert.

SSD expert paging then reads a missing expert with one pread instead of one per field (9 for a
quantized GLU expert: up/gate/down x weight/scales/biases). Streams the checkpoint by blocks of
experts; never loads the model. The output can live on another disk than the checkpoint
(MLXSERVE_SSD_EXPERTS_PACKED_DIR), e.g. packed experts on the internal SSD, checkpoint elsewhere.

    python scripts/pack_ssd_experts.py --src /path/to/checkpoint --dst /path/to/packed-dir
"""
import argparse
import fcntl
import json
import os
import struct
import sys
import time
from pathlib import Path

PROJECTIONS = (("up_proj", "gate_proj", "down_proj"), ("fc1", "fc2"))
FIELDS = ("weight", "scales", "biases", "bias", "global_scale")  # SSDLinear's order
OUT_NAME = "ssd_experts.safetensors"
SUFFIX = ".ssd_packed"


def read_headers(src: Path) -> dict:
    tensors = {}
    for path in sorted(src.glob("model*.safetensors")):
        with path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for name, meta in header.items():
            if name != "__metadata__":
                tensors[name] = (path, 8 + n, meta)
    return tensors


def expert_modules(tensors: dict) -> dict:
    """prefix -> [(field name relative to prefix, tensor name)] in the order the server reads them."""
    out = {}
    for name, (_, _, meta) in tensors.items():
        for projections in PROJECTIONS:
            suffix = f".{projections[0]}.weight"
            if not name.endswith(suffix) or len(meta["shape"]) < 2:
                continue
            prefix = name[: -len(suffix)]
            fields = [f"{p}.{f}" for p in projections for f in FIELDS if f"{prefix}.{p}.{f}" in tensors]
            if all(f"{prefix}.{p}.weight" in tensors for p in projections):
                experts = {tensors[f"{prefix}.{f}"][2]["shape"][0] for f in fields}
                if len(experts) == 1:
                    out[prefix] = fields
    return out


def row_bytes(meta) -> int:
    begin, end = meta["data_offsets"]
    return (end - begin) // meta["shape"][0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--first-module", type=int, default=0, help="skip the first N expert modules (benchmarks)")
    ap.add_argument("--max-modules", type=int, default=0, help="pack only N expert modules (benchmarks)")
    ap.add_argument("--block", type=int, default=64, help="experts read per block")
    a = ap.parse_args()

    tensors = read_headers(a.src)
    modules = expert_modules(tensors)
    prefixes = sorted(modules, key=lambda p: [int(t) if t.isdigit() else t for t in p.split(".")])
    prefixes = prefixes[a.first_module :]
    if a.max_modules:
        prefixes = prefixes[: a.max_modules]
    if not prefixes:
        print("No stacked expert modules found", file=sys.stderr)
        return 1

    header, layout, offset = {}, {}, 0
    for prefix in prefixes:
        fields, pos = [], 0
        for field in modules[prefix]:
            nbytes = row_bytes(tensors[f"{prefix}.{field}"][2])
            fields.append([field, pos, nbytes])
            pos += nbytes
        experts = tensors[f"{prefix}.{modules[prefix][0]}"][2]["shape"][0]
        header[prefix + SUFFIX] = {"dtype": "U8", "shape": [experts, pos], "data_offsets": [offset, offset + experts * pos]}
        layout[prefix] = fields
        offset += experts * pos
    header["__metadata__"] = {"ssd_fields": json.dumps(layout)}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)

    a.dst.mkdir(parents=True, exist_ok=True)
    out_path = a.dst / OUT_NAME
    tmp = out_path.with_suffix(".tmp")
    fds, started, written = {}, time.time(), 0
    with tmp.open("wb") as out:
        if hasattr(fcntl, "F_NOCACHE"):  # macOS: don't fill RAM with 40 GB of page cache
            fcntl.fcntl(out.fileno(), fcntl.F_NOCACHE, 1)
        out.write(struct.pack("<Q", len(blob)) + blob)
        for i, prefix in enumerate(prefixes):
            experts, row = header[prefix + SUFFIX]["shape"]
            for first in range(0, experts, a.block):
                count = min(a.block, experts - first)
                parts = []
                for field, _, nbytes in layout[prefix]:
                    path, base, meta = tensors[f"{prefix}.{field}"]
                    if path not in fds:
                        fds[path] = os.open(path, os.O_RDONLY)
                        if hasattr(fcntl, "F_NOCACHE"):
                            fcntl.fcntl(fds[path], fcntl.F_NOCACHE, 1)
                    data = os.pread(fds[path], count * nbytes, base + meta["data_offsets"][0] + first * nbytes)
                    if len(data) != count * nbytes:
                        raise ValueError(f"Truncated tensor {prefix}.{field}")
                    parts.append((data, nbytes))
                out.write(b"".join(d[e * n : (e + 1) * n] for e in range(count) for d, n in parts))
                written += count * row
            print(f"[{i + 1}/{len(prefixes)}] {prefix}  {written / 1e9:.1f} GB  {written / 1e9 / (time.time() - started):.2f} GB/s", flush=True)
    for fd in fds.values():
        os.close(fd)
    os.replace(tmp, out_path)
    print(f"Wrote {out_path} ({written / 2**30:.2f} GiB, {len(prefixes)} expert modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
