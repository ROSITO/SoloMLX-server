#!/usr/bin/env bash
# Two separate Python processes: dense+MoE(full) then dense+MoE(shrunk), then merge JSON.
# Avoids OOM on Apple MPS when a single process loads two 7B checkpoints back-to-back.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
MODEL_ID="${MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.2}"
LAYERS="${LAYERS:-10}"
DEVICE="${DEVICE:-mps}"
NUM_EXPERTS_FULL="${NUM_EXPERTS_FULL:-8}"
NUM_EXPERTS_SHRUNK="${NUM_EXPERTS_SHRUNK:-4}"
ADAPTER_FULL="${ADAPTER_FULL:-bench/moe_training/moe_expert_ticket_mistral7b_mps_full.pt}"
ADAPTER_SHRUNK="${ADAPTER_SHRUNK:-bench/moe_training/moe_expert_ticket_mistral7b_mps_full_shrunk4.pt}"
OUT_MERGED="${OUT_MERGED:-bench/moe_training/moe_expert_ticket_mistral7b_mps_eval_ab.json}"
PART_FULL="${PART_FULL:-bench/moe_training/_part_moe_eval_full.json}"
PART_SHRUNK="${PART_SHRUNK:-bench/moe_training/_part_moe_eval_shrunk.json}"
WARMUP="${WARMUP:-0}"
TOP_K="${TOP_K:-1}"
# shellcheck disable=SC2206
EVAL_EXTRA=( ${EXTRA_EVAL_ARGS:-} )
if [[ ${#EVAL_EXTRA[@]} -eq 0 && "$DEVICE" == "mps" ]]; then
  EVAL_EXTRA=(--dtype bf16 --low-cpu-mem-usage --offload-unused-experts)
fi
COMMON=(--model-id "$MODEL_ID" --prompts bench/eval_quality_v1.json --layers "$LAYERS" --device "$DEVICE"
  --top-k "$TOP_K" --shared-experts 0 --warmup-steps "$WARMUP")

echo "[1/3] eval dense + MoE full (${NUM_EXPERTS_FULL} experts)"
"$PY" -m training.moe_eval_ab "${COMMON[@]}" "${EVAL_EXTRA[@]}" \
  --num-experts "$NUM_EXPERTS_FULL" \
  --adapter-path "$ADAPTER_FULL" \
  --out "$PART_FULL"

echo "[2/3] eval dense + MoE shrunk (${NUM_EXPERTS_SHRUNK} experts, new process)"
"$PY" -m training.moe_eval_ab "${COMMON[@]}" "${EVAL_EXTRA[@]}" \
  --num-experts "$NUM_EXPERTS_SHRUNK" \
  --adapter-path "$ADAPTER_SHRUNK" \
  --out "$PART_SHRUNK"

echo "[3/3] merge reports -> $OUT_MERGED"
"$PY" -m training.merge_moe_eval_ab_tri --full-run "$PART_FULL" --shrunk-run "$PART_SHRUNK" --out "$OUT_MERGED"
echo "Done."
