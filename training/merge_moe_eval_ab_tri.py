"""Merge two `moe_eval_ab` JSON reports into one tri-branch report (dense / MoE full / MoE shrunk).

Run A: `moe_eval_ab` with full adapter only (no `--adapter-path-shrunk`).
Run B: same flags except `--num-experts` matching shrunk adapter and `--adapter-path` pointing
to the shrunk `.pt` only. Then merge A + B into the same shape as a single tri-branch run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def merge_tri(full_run: dict[str, Any], shrunk_run: dict[str, Any]) -> dict[str, Any]:
    d = full_run["dense"]
    m_full = full_run["moe_bridge"]
    m_sh = shrunk_run["moe_bridge"]

    out = {
        "model_id": full_run.get("model_id"),
        "eval_strategy": "subprocess_split_merged_tri",
        "layers": full_run.get("layers"),
        "moe_config": full_run.get("moe_config"),
        "adapter_path": full_run.get("adapter_path"),
        "adapter_layers_loaded": full_run.get("adapter_layers_loaded"),
        "adapter_path_shrunk": shrunk_run.get("adapter_path"),
        "moe_shrunk_meta": {
            "num_experts_inferred": shrunk_run.get("moe_config", {}).get("num_experts"),
            "num_experts_used": shrunk_run.get("moe_config", {}).get("num_experts"),
            "adapter_layers_loaded": shrunk_run.get("adapter_layers_loaded"),
            "source_report": "shrunk_leg",
        },
        "dense": dict(d),
        "moe_bridge": dict(m_full),
        "delta": dict(full_run["delta"]),
        "moe_bridge_shrunk": {
            "avg_loss": m_sh["avg_loss"],
            "p95_latency_ms": m_sh["p95_latency_ms"],
        },
        "delta_shrunk_vs_dense": {
            "avg_loss": m_sh["avg_loss"] - d["avg_loss"],
            "p95_latency_ms": m_sh["p95_latency_ms"] - d["p95_latency_ms"],
        },
        "delta_shrunk_vs_moe": {
            "avg_loss": m_sh["avg_loss"] - m_full["avg_loss"],
            "p95_latency_ms": m_sh["p95_latency_ms"] - m_full["p95_latency_ms"],
        },
        "samples_dense": full_run.get("samples_dense"),
        "samples_moe_bridge": full_run.get("samples_moe_bridge"),
        "samples_moe_bridge_shrunk": shrunk_run.get("samples_moe_bridge"),
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Merge full + shrunk-only moe_eval_ab JSON outputs.")
    p.add_argument("--full-run", required=True, help="JSON from moe_eval_ab without shrunk adapter.")
    p.add_argument("--shrunk-run", required=True, help="JSON from moe_eval_ab with shrunk adapter as sole MoE.")
    p.add_argument("--out", required=True, help="Merged tri-branch report path.")
    args = p.parse_args()
    full_run = json.loads(Path(args.full_run).read_text(encoding="utf-8"))
    shrunk_run = json.loads(Path(args.shrunk_run).read_text(encoding="utf-8"))
    merged = merge_tri(full_run, shrunk_run)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(json.dumps(merged, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
