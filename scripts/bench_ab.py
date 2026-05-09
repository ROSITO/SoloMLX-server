import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_bench_for_backend(backend: str, out_dir: Path) -> dict:
    out_json = out_dir / f"results_{backend}.json"
    out_csv = out_dir / f"results_{backend}.csv"
    cmd = [
        sys.executable,
        "scripts/bench_chat.py",
        "--mock",
        "--out-json",
        str(out_json),
        "--out-csv",
        str(out_csv),
    ]
    env = os.environ.copy()
    env["MLXSERVE_RUNTIME_BACKEND"] = backend
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if proc.returncode not in (0, 2):
        raise RuntimeError(f"Benchmark failed for backend={backend}: {proc.stdout}\n{proc.stderr}")
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B benchmark between stub and moe_stub backends.")
    parser.add_argument("--out-json", default="bench/ab_report.json")
    args = parser.parse_args()

    out_dir = Path("bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    stub = run_bench_for_backend("stub", out_dir)
    moe = run_bench_for_backend("moe_stub", out_dir)

    report = {
        "stub": stub["summary"],
        "moe_stub": moe["summary"],
        "delta": {
            "latency_p95_ms": moe["summary"]["latency_p95_ms"] - stub["summary"]["latency_p95_ms"],
            "tokens_per_second_avg": moe["summary"]["tokens_per_second_avg"] - stub["summary"]["tokens_per_second_avg"],
            "error_rate": moe["summary"]["error_rate"] - stub["summary"]["error_rate"],
        },
        "gates": {"stub": stub["gates"], "moe_stub": moe["gates"]},
    }
    Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
