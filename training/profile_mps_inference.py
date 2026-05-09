"""Profile RAM / CPU / MPS memory during a short Torch+MPS inference (no sudo).

GPU "utilization %" is not exposed by PyTorch; use Activity Monitor or
`sudo powermetrics` for power / load curves.

Run from repo root::

  .venv/bin/python -m training.profile_mps_inference --adapter-path bench/moe_training/mistral7b_moe_adapter_v2_balanced.pt
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.moe_model import load_moe_state, replace_llama_mlp_with_moe


@dataclass
class Sample:
    t_wall: float
    rss_bytes: int
    cpu_proc_pct: float
    cpu_system_pct: float
    mps_current_bytes: int | None
    mps_driver_bytes: int | None
    vm_percent: float
    swap_used_bytes: int


@dataclass
class ProfileReport:
    model_id: str
    adapter_path: str | None
    layers: list[int]
    offload_unused_experts: bool
    max_new_tokens: int
    wall_s: float
    ru_maxrss_bytes: int
    rss_peak_bytes: int
    rss_avg_bytes: float
    cpu_proc_avg: float
    cpu_proc_max: float
    cpu_system_avg: float
    mps_current_peak_bytes: int | None
    mps_driver_peak_bytes: int | None
    samples: list[dict[str, Any]] = field(default_factory=list)


def _mps_mem() -> tuple[int | None, int | None]:
    if not torch.backends.mps.is_available():
        return None, None
    try:
        cur = int(torch.mps.current_allocated_memory())
        drv = int(torch.mps.driver_allocated_memory())
        return cur, drv
    except Exception:
        return None, None


def _ru_maxrss_bytes() -> int:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    v = int(ru.ru_maxrss)
    if sys.platform == "darwin":
        return v
    return v * 1024


def main() -> int:
    p = argparse.ArgumentParser(description="Profile Torch MPS inference (RAM/CPU/MPS memory).")
    p.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--adapter-path", default="")
    p.add_argument("--layers", default="10")
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--shared-experts", type=int, default=0)
    p.add_argument("--prompt", default="Resume en 3 points le compromis latence/qualite pour un MoE local.")
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--sample-ms", type=int, default=100)
    p.add_argument("--out", default="bench/moe_training/profile_mps_run.json")
    p.add_argument(
        "--offload-unused-experts",
        action="store_true",
        help="CPU-park MoE experts not selected on this forward (reduces MPS footprint; more host transfers).",
    )
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    adapter_path = args.adapter_path.strip() or None

    proc = psutil.Process()
    stop = threading.Event()
    samples: list[Sample] = []

    interval_s = max(args.sample_ms, 50) / 1000.0

    def sampler() -> None:
        while not stop.wait(timeout=interval_s):
            cur, drv = _mps_mem()
            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()
            samples.append(
                Sample(
                    t_wall=time.perf_counter(),
                    rss_bytes=proc.memory_info().rss,
                    cpu_proc_pct=proc.cpu_percent(interval=None),
                    cpu_system_pct=psutil.cpu_percent(interval=None),
                    mps_current_bytes=cur,
                    mps_driver_bytes=drv,
                    vm_percent=vm.percent,
                    swap_used_bytes=int(sw.used),
                )
            )

    proc.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None)

    t0 = time.perf_counter()
    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id).to("mps")
    model, _ = replace_llama_mlp_with_moe(
        model,
        layers,
        num_experts=args.num_experts,
        top_k=args.top_k,
        shared_experts=args.shared_experts,
        fast_top1=True,
        offload_unused_experts=args.offload_unused_experts,
    )
    if adapter_path:
        state = torch.load(adapter_path, map_location="cpu")
        load_moe_state(model, layers, state)
    model.eval()

    enc = tok(args.prompt, return_tensors="pt")
    enc = {k: v.to("mps") for k, v in enc.items()}
    with torch.inference_mode():
        _ = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
    torch.mps.synchronize()

    stop.set()
    th.join(timeout=2.0)
    wall_s = time.perf_counter() - t0

    ru_maxrss = _ru_maxrss_bytes()
    rss_vals = [s.rss_bytes for s in samples]
    cpu_p = [s.cpu_proc_pct for s in samples if s.cpu_proc_pct >= 0]
    cpu_sys = [s.cpu_system_pct for s in samples if s.cpu_system_pct >= 0]
    mps_c = [s.mps_current_bytes for s in samples if s.mps_current_bytes is not None]
    mps_d = [s.mps_driver_bytes for s in samples if s.mps_driver_bytes is not None]

    report = ProfileReport(
        model_id=args.model_id,
        adapter_path=adapter_path,
        layers=layers,
        offload_unused_experts=args.offload_unused_experts,
        max_new_tokens=args.max_new_tokens,
        wall_s=round(wall_s, 3),
        ru_maxrss_bytes=ru_maxrss,
        rss_peak_bytes=max(rss_vals) if rss_vals else 0,
        rss_avg_bytes=round(sum(rss_vals) / len(rss_vals), 1) if rss_vals else 0.0,
        cpu_proc_avg=round(sum(cpu_p) / len(cpu_p), 2) if cpu_p else 0.0,
        cpu_proc_max=round(max(cpu_p), 2) if cpu_p else 0.0,
        cpu_system_avg=round(sum(cpu_sys) / len(cpu_sys), 2) if cpu_sys else 0.0,
        mps_current_peak_bytes=max(mps_c) if mps_c else None,
        mps_driver_peak_bytes=max(mps_d) if mps_d else None,
        samples=[asdict(s) for s in samples[:2000]],
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in asdict(report).items() if k != "samples"}
    payload["samples_count"] = len(samples)
    payload["samples_preview"] = report.samples[:80]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def gb(x: int | float) -> str:
        return f"{float(x) / (1024**3):.2f} GiB"

    md = report.mps_driver_peak_bytes
    print(json.dumps(payload, indent=2))
    print(
        "\n---\n"
        f"wall_s={report.wall_s}  ru_maxrss~={gb(report.ru_maxrss_bytes)}  "
        f"rss_peak~={gb(report.rss_peak_bytes)}  "
        f"mps_driver_peak~={gb(md) if md is not None else 'n/a'}\n"
        f"cpu_proc avg/max %={report.cpu_proc_avg}/{report.cpu_proc_max}  "
        f"cpu_system avg %={report.cpu_system_avg}\n"
        f"full_json={out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
