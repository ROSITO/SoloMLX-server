#!/usr/bin/env bash
# Dense→MoE safetensors (24B puis 32B), sortie visible dans ce terminal + log append.
# Usage : depuis la racine du repo —  bash scripts/run_scaleup_moe_convert_24b_32b.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
LOG="${ROOT}/bench/moe_conversion/scaleup_moe_convert_24b_32b.log"
mkdir -p "$(dirname "$LOG")"
{
  echo ""
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) run_scaleup_moe_convert_24b_32b.sh (terminal + ${LOG}) ==="
  .venv/bin/python -m tools.run_sprint2_conversion \
    --repo-id mistralai/Mistral-Small-24B-Instruct-2501 \
    --workdir bench/moe_conversion/mistral_small_24b \
    --layers 20 --num-experts 4 --top-k 1 --shared-experts 0
  echo "=== 24B OK $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  .venv/bin/python -m tools.run_sprint2_conversion \
    --repo-id Qwen/Qwen2.5-32B-Instruct \
    --workdir bench/moe_conversion/qwen25_32b \
    --layers 32 --num-experts 4 --top-k 1 --shared-experts 0
  echo "=== 32B OK $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
} 2>&1 | tee -a "$LOG"
