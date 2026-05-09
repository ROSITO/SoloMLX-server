#!/usr/bin/env bash
# Relance uniquement la conversion 32B (ex. après erreur HF 5xx sur la chaîne 24B+32B).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
LOG="${ROOT}/bench/moe_conversion/scaleup_moe_convert_24b_32b.log"
mkdir -p "$(dirname "$LOG")"
{
  echo ""
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) Qwen2.5-32B only (retry) ==="
  .venv/bin/python -m tools.run_sprint2_conversion \
    --repo-id Qwen/Qwen2.5-32B-Instruct \
    --workdir bench/moe_conversion/qwen25_32b \
    --layers 32 --num-experts 4 --top-k 1 --shared-experts 0
  echo "=== 32B OK $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
} 2>&1 | tee -a "$LOG"
