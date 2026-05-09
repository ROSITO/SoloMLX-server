#!/usr/bin/env bash
# Mistral-7B on Apple MPS: expert-ticket train (entropy + shrink) + dense/MoE/shrunk eval.
# Override STEPS (default 120), SHRINK_TO, LAYERS, OUT_* as needed.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_ID="${MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.2}"
export DEVICE="${DEVICE:-mps}"
export STEPS="${STEPS:-120}"
export LAYERS="${LAYERS:-10}"
export NUM_EXPERTS="${NUM_EXPERTS:-8}"
export TOP_K="${TOP_K:-1}"
export SHRINK_TO="${SHRINK_TO:-4}"
export WARMUP_STEPS="${WARMUP_STEPS:-2}"
export OUT_FULL="${OUT_FULL:-bench/moe_training/moe_expert_ticket_mistral7b_mps_full.pt}"
export OUT_TRAIN_REPORT="${OUT_TRAIN_REPORT:-bench/moe_training/moe_expert_ticket_mistral7b_mps_train.json}"
export OUT_SHRUNK="${OUT_SHRUNK:-bench/moe_training/moe_expert_ticket_mistral7b_mps_full_shrunk${SHRINK_TO}.pt}"
export OUT_EVAL="${OUT_EVAL:-bench/moe_training/moe_expert_ticket_mistral7b_mps_eval_ab.json}"
: "${EXTRA_TRAIN_ARGS:=--dtype bf16 --low-cpu-mem-usage --offload-unused-experts}"
: "${EXTRA_EVAL_ARGS:=--dtype bf16 --low-cpu-mem-usage --offload-unused-experts}"
export EXTRA_TRAIN_ARGS
export EXTRA_EVAL_ARGS
# Deux processus Python pour l'eval (evite OOM MPS entre MoE 8 et MoE 4).
export EVAL_TRI_SPLIT=1
exec bash "$DIR/run_moe_expert_ticket_smoke.sh"
