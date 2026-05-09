#!/usr/bin/env bash
# Reproducible micro-grid: mlx_moe_bench over KV + MoE resident settings (P0 roadmap).
# Usage: MODEL_ID=... RAM_BUDGET_GIB=16 bash scripts/run_mlx_inference_grid.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
MODEL_ID="${MODEL_ID:-mlx-community/Mistral-7B-Instruct-v0.3-4bit}"
OUT_DIR="${OUT_DIR:-bench/moe_training/mlx_grid}"
MAX_TOKENS="${MAX_TOKENS:-24}"
RAM_BUDGET_GIB="${RAM_BUDGET_GIB:-}"
mkdir -p "$OUT_DIR"

run_one() {
  local name="$1"
  shift
  echo "=== $name ==="
  local out="$OUT_DIR/${name}.json"
  local cmd=( "$PY" -m training.mlx_moe_bench --model-id "$MODEL_ID" --max-tokens "$MAX_TOKENS" --out "$out" "$@" )
  if [[ -n "$RAM_BUDGET_GIB" ]]; then
    cmd+=( --ram-budget-gib "$RAM_BUDGET_GIB" )
  fi
  "${cmd[@]}" || echo "[warn] $name exit=$?"
}

run_one "kv_default_moe0" --moe-resident-experts 0
run_one "kv4_q32_moe0" --kv-bits 4 --quantized-kv-start 32 --moe-resident-experts 0
run_one "kv_default_moe2" --moe-resident-experts 2 --moe-resident-strategy l2
run_one "kv4_q32_moe2" --kv-bits 4 --quantized-kv-start 32 --moe-resident-experts 2 --moe-resident-strategy l2

echo "Grid done. Reports under $OUT_DIR"
