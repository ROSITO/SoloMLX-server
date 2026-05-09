import json
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.bench_chat import (
    evaluate_gates,
    load_prompts,
    load_simple_yaml_gates,
    run_case_http,
    summarize,
)


def test_load_prompts_and_gates() -> None:
    prompts = load_prompts(Path("bench/prompts.json"))
    gates = load_simple_yaml_gates(Path("bench/gates.yaml"))
    assert isinstance(prompts, list)
    assert len(prompts) >= 3
    assert "max_error_rate" in gates


def test_summarize_and_gates_decision() -> None:
    # Minimal synthetic result payload serialized through JSON to emulate script outputs.
    synthetic = [
        {
            "case_id": "a",
            "ok": True,
            "status_code": 200,
            "latency_ms": 200.0,
            "prompt_tokens": 20,
            "completion_tokens": 40,
            "total_tokens": 60,
            "tokens_per_second": 20.0,
            "error": "",
        },
        {
            "case_id": "b",
            "ok": True,
            "status_code": 200,
            "latency_ms": 300.0,
            "prompt_tokens": 22,
            "completion_tokens": 35,
            "total_tokens": 57,
            "tokens_per_second": 14.0,
            "error": "",
        },
    ]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    try:
        Path(tmp.name).write_text(json.dumps(synthetic), encoding="utf-8")
        data = json.loads(Path(tmp.name).read_text(encoding="utf-8"))
        from scripts.bench_chat import BenchResult

        results = [BenchResult(**row) for row in data]
    finally:
        tmp.close()

    s = summarize(results)
    g = evaluate_gates(
        s,
        {
            "max_error_rate": 0.5,
            "max_p95_ms": 1000.0,
            "min_tokens_per_second": 5.0,
            "max_memory_denials": 0.0,
        },
    )
    assert s["cases_ok"] == 2
    assert all(g.values())


def test_bench_script_mock_mode_writes_outputs() -> None:
    with tempfile.TemporaryDirectory() as td:
        out_json = Path(td) / "results.json"
        out_csv = Path(td) / "results.csv"
        cmd = [
            sys.executable,
            "scripts/bench_chat.py",
            "--mock",
            "--out-json",
            str(out_json),
            "--out-csv",
            str(out_csv),
        ]
        proc = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True, check=False)
        assert proc.returncode in (0, 2), proc.stdout + proc.stderr
        assert out_json.exists()
        assert out_csv.exists()
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert "summary" in payload
        assert "gates" in payload


def test_run_case_http_handles_connection_failures() -> None:
    case = {
        "id": "conn_fail",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 8,
        "temperature": 0.1,
        "top_p": 0.9,
    }
    result = run_case_http("http://127.0.0.1:9", case, timeout_s=0.1)
    assert result.ok is False
    assert result.status_code == 0
