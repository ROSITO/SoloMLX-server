from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import model_info


def main() -> int:
    p = argparse.ArgumentParser(description="Prepare scale-up gate decision (SmolLM -> 7B).")
    p.add_argument("--target-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--source-report", default="bench/moe_training/eval_ab_report_adapter_v2_mps.json")
    p.add_argument("--out", default="bench/moe_training/scaleup_gate.json")
    args = p.parse_args()

    info = model_info(args.target_model)
    eval_report = json.loads(Path(args.source_report).read_text(encoding="utf-8"))
    delta_loss = float(eval_report["delta"]["avg_loss"])
    delta_p95 = float(eval_report["delta"]["p95_latency_ms"])

    decision = {
        "target_model": args.target_model,
        "target_license": (info.cardData or {}).get("license"),
        "source_eval": args.source_report,
        "delta_loss": delta_loss,
        "delta_p95_latency_ms": delta_p95,
        "gate_quality_ok": delta_loss <= 0.0,
        "gate_latency_ok": delta_p95 <= 0.0,
    }
    decision["go_scaleup"] = bool(decision["gate_quality_ok"] and decision["gate_latency_ok"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
