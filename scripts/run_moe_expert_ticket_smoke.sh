#!/usr/bin/env bash
# Pipeline: MoE adapter train (entropy + shrink) then eval dense / full MoE / shrunk MoE.
# Defaults: SmolLM2-135M + CPU. Override MODEL_ID, DEVICE, STEPS, EXTRA_TRAIN_ARGS, etc.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
MODEL_ID="${MODEL_ID:-HuggingFaceTB/SmolLM2-135M-Instruct}"
LAYERS="${LAYERS:-10}"
NUM_EXPERTS="${NUM_EXPERTS:-8}"
TOP_K="${TOP_K:-1}"
SHRINK_TO="${SHRINK_TO:-4}"
STEPS="${STEPS:-32}"
DEVICE="${DEVICE:-cpu}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
ROUTER_ENTROPY_WEIGHT="${ROUTER_ENTROPY_WEIGHT:-0.02}"
ROUTER_BALANCE_WEIGHT="${ROUTER_BALANCE_WEIGHT:-0}"
OUT_FULL="${OUT_FULL:-bench/moe_training/moe_expert_ticket_smoke_full.pt}"
OUT_TRAIN_REPORT="${OUT_TRAIN_REPORT:-bench/moe_training/moe_expert_ticket_smoke_train.json}"
OUT_SHRUNK="${OUT_SHRUNK:-bench/moe_training/moe_expert_ticket_smoke_full_shrunk${SHRINK_TO}.pt}"
OUT_EVAL="${OUT_EVAL:-bench/moe_training/moe_expert_ticket_smoke_eval_ab.json}"
# shellcheck disable=SC2206
TRAIN_EXTRA=( ${EXTRA_TRAIN_ARGS:-} )
EVAL_EXTRA=( ${EXTRA_EVAL_ARGS:-} )

echo "[1/2] moe_target_train (entropy + shrink-to ${SHRINK_TO})"
"$PY" -m training.moe_target_train \
  --model-id "$MODEL_ID" \
  --corpus bench/train_corpus_v1.jsonl \
  --layers "$LAYERS" \
  --num-experts "$NUM_EXPERTS" \
  --top-k "$TOP_K" \
  --shared-experts 0 \
  --steps "$STEPS" \
  --router-balance-weight "$ROUTER_BALANCE_WEIGHT" \
  --router-entropy-weight "$ROUTER_ENTROPY_WEIGHT" \
  --shrink-to-experts "$SHRINK_TO" \
  --device "$DEVICE" \
  --out-adapter "$OUT_FULL" \
  --out-report "$OUT_TRAIN_REPORT" \
  "${TRAIN_EXTRA[@]}"

if [[ ! -f "$OUT_SHRUNK" ]]; then
  echo "Expected shrunk adapter at $OUT_SHRUNK (from --shrink-to-experts); aborting."
  exit 1
fi

echo "[2/2] eval (tri-branch)"
if [[ "${EVAL_TRI_SPLIT:-0}" == "1" ]]; then
  export ADAPTER_FULL="$OUT_FULL"
  export ADAPTER_SHRUNK="$OUT_SHRUNK"
  export OUT_MERGED="$OUT_EVAL"
  export NUM_EXPERTS_FULL="$NUM_EXPERTS"
  export NUM_EXPERTS_SHRUNK="$SHRINK_TO"
  export TOP_K="${TOP_K}"
  export WARMUP="${WARMUP_STEPS}"
  export PART_FULL="${PART_FULL:-bench/moe_training/_part_moe_eval_full.json}"
  export PART_SHRUNK="${PART_SHRUNK:-bench/moe_training/_part_moe_eval_shrunk.json}"
  bash "$ROOT/scripts/run_moe_eval_ab_tri_split.sh"
else
  "$PY" -m training.moe_eval_ab \
    --model-id "$MODEL_ID" \
    --prompts bench/eval_quality_v1.json \
    --layers "$LAYERS" \
    --num-experts "$NUM_EXPERTS" \
    --top-k "$TOP_K" \
    --shared-experts 0 \
    --warmup-steps "$WARMUP_STEPS" \
    --adapter-path "$OUT_FULL" \
    --adapter-path-shrunk "$OUT_SHRUNK" \
    --device "$DEVICE" \
    --out "$OUT_EVAL" \
    "${EVAL_EXTRA[@]}"
fi

echo "Done. Train report: $OUT_TRAIN_REPORT  Eval report: $OUT_EVAL"
