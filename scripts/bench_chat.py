import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


@dataclass
class BenchResult:
    case_id: str
    ok: bool
    status_code: int
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_per_second: float
    error: str = ""


def load_prompts(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("prompts.json must be a non-empty JSON list")
    return raw


def load_simple_yaml_gates(path: Path) -> dict[str, float]:
    gates: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid line in gates file: {line}")
        key, value = line.split(":", 1)
        gates[key.strip()] = float(value.strip())
    return gates


def _post_json(base_url: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any]]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return int(resp.status), body
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8") if exc.fp else ""
        body = {"error": {"message": raw or str(exc)}}
        return int(exc.code), body
    except Exception as exc:
        return 0, {"error": {"message": str(exc)}}


def run_case_http(base_url: str, case: dict[str, Any], timeout_s: float) -> BenchResult:
    payload = {
        "messages": case["messages"],
        "stream": False,
        "max_tokens": int(case.get("max_tokens", 128)),
        "temperature": float(case.get("temperature", 0.1)),
        "top_p": float(case.get("top_p", 0.9)),
    }
    if "model" in case:
        payload["model"] = case["model"]

    started = time.perf_counter()
    status_code, body = _post_json(base_url, payload, timeout_s)
    elapsed_s = max(time.perf_counter() - started, 1e-6)
    latency_ms = elapsed_s * 1000.0

    if status_code != 200:
        msg = body.get("error", {}).get("message", f"HTTP {status_code}")
        return BenchResult(
            case_id=case["id"],
            ok=False,
            status_code=status_code,
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            tokens_per_second=0.0,
            error=str(msg),
        )

    usage = body.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
    tps = completion_tokens / elapsed_s if completion_tokens > 0 else 0.0
    return BenchResult(
        case_id=case["id"],
        ok=True,
        status_code=status_code,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        tokens_per_second=tps,
    )


def run_case_mock(client: Any, case: dict[str, Any]) -> BenchResult:
    payload = {
        "messages": case["messages"],
        "stream": False,
        "max_tokens": int(case.get("max_tokens", 128)),
        "temperature": float(case.get("temperature", 0.1)),
        "top_p": float(case.get("top_p", 0.9)),
    }
    if "model" in case:
        payload["model"] = case["model"]

    started = time.perf_counter()
    resp = client.post("/v1/chat/completions", json=payload)
    elapsed_s = max(time.perf_counter() - started, 1e-6)
    latency_ms = elapsed_s * 1000.0

    if resp.status_code != 200:
        try:
            body = resp.json()
        except Exception:
            body = {"error": {"message": resp.text}}
        msg = body.get("error", {}).get("message", f"HTTP {resp.status_code}")
        return BenchResult(
            case_id=case["id"],
            ok=False,
            status_code=resp.status_code,
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            tokens_per_second=0.0,
            error=str(msg),
        )

    body = resp.json()
    usage = body.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
    tps = completion_tokens / elapsed_s if completion_tokens > 0 else 0.0
    return BenchResult(
        case_id=case["id"],
        ok=True,
        status_code=resp.status_code,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        tokens_per_second=tps,
    )


def summarize(results: list[BenchResult]) -> dict[str, float]:
    latencies = [r.latency_ms for r in results]
    tps_values = [r.tokens_per_second for r in results if r.ok]
    errors = [r for r in results if not r.ok]
    denials = sum(1 for r in results if r.status_code == 503)
    return {
        "cases_total": float(len(results)),
        "cases_ok": float(len(results) - len(errors)),
        "error_rate": (len(errors) / len(results)) if results else 1.0,
        "latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
        "latency_p95_ms": max(latencies) if len(latencies) < 20 else statistics.quantiles(latencies, n=100)[94],
        "tokens_per_second_avg": (sum(tps_values) / len(tps_values)) if tps_values else 0.0,
        "memory_denials": float(denials),
    }


def evaluate_gates(summary: dict[str, float], gates: dict[str, float]) -> dict[str, bool]:
    return {
        "error_rate_ok": summary["error_rate"] <= gates.get("max_error_rate", 0.05),
        "p95_ok": summary["latency_p95_ms"] <= gates.get("max_p95_ms", 12000.0),
        "tps_ok": summary["tokens_per_second_avg"] >= gates.get("min_tokens_per_second", 1.0),
        "memory_denials_ok": summary["memory_denials"] <= gates.get("max_memory_denials", 0.0),
    }


def write_csv(path: Path, results: list[BenchResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "case_id",
                "ok",
                "status_code",
                "latency_ms",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "tokens_per_second",
                "error",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark MLXServe chat endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="MLXServe base URL")
    parser.add_argument("--prompts", default="bench/prompts.json", help="Prompt cases JSON file")
    parser.add_argument("--gates", default="bench/gates.yaml", help="Gate thresholds YAML file")
    parser.add_argument("--out-json", default="bench/results.json", help="Summary output JSON")
    parser.add_argument("--out-csv", default="bench/results.csv", help="Detailed output CSV")
    parser.add_argument("--timeout-s", type=float, default=30.0, help="HTTP timeout per case")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run benchmark against in-process FastAPI app (no external server required).",
    )
    args = parser.parse_args()

    prompts = load_prompts(Path(args.prompts))
    gates = load_simple_yaml_gates(Path(args.gates))

    if args.mock:
        from fastapi.testclient import TestClient

        from mlxserve.api.app import app
        from mlxserve.api.deps import engine, guardian
        from mlxserve.runtime.backends import StubBackend

        original = (guardian.soft_limit_gb, guardian.hard_limit_gb, engine.backend, engine.loaded_model)
        guardian.soft_limit_gb = 10_000
        guardian.hard_limit_gb = 20_000
        engine.backend = StubBackend()
        engine.loaded_model = None
        try:
            client = TestClient(app)
            results = [run_case_mock(client, case) for case in prompts]
        finally:
            guardian.soft_limit_gb, guardian.hard_limit_gb, engine.backend, engine.loaded_model = original
    else:
        results = [run_case_http(args.base_url, case, args.timeout_s) for case in prompts]
    summary = summarize(results)
    gate_status = evaluate_gates(summary, gates)
    payload = {"summary": summary, "gates": gate_status, "thresholds": gates}

    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(Path(args.out_csv), results)

    print(json.dumps(payload, indent=2))
    return 0 if all(gate_status.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
