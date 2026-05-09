from __future__ import annotations

import argparse
import json
from pathlib import Path


def estimate_proxy(report: dict) -> dict:
    total = float(report["total_input_keys"])
    converted = float(report["converted_keys"])
    top_k = float(report["top_k"])
    shared = float(report["shared_experts"])

    dense_active_proxy = total
    moe_active_proxy = (total - converted) + converted * (top_k + shared)
    ratio = (moe_active_proxy / dense_active_proxy) if dense_active_proxy else 0.0
    return {
        "dense_active_proxy": dense_active_proxy,
        "moe_active_proxy": moe_active_proxy,
        "moe_vs_dense_ratio": ratio,
        "delta_percent": (ratio - 1.0) * 100.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Estimate active-params proxy from conversion report.")
    p.add_argument(
        "--report",
        default="bench/moe_conversion/smollm2_135m/conversion_report.json",
    )
    p.add_argument(
        "--out",
        default="bench/moe_conversion/smollm2_135m/active_params_proxy.json",
    )
    args = p.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    out = estimate_proxy(report)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
