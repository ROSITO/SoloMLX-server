#!/usr/bin/env bash
# Lance un échantillonnage GPU/CPU via powermetrics (macOS).
# À exécuter dans Terminal.app ou un terminal où tu peux saisir ton mot de passe sudo.
#
# Terminal 1 — démarre l’échantillonnage (ex. 30 s, toutes les 1 s) :
#   ./scripts/powermetrics_gpu_sample.sh
#
# Terminal 2 — pendant ce temps, lance l’inférence (ex.) :
#   .venv/bin/python -m training.profile_mps_inference \
#     --adapter-path bench/moe_training/mistral7b_moe_adapter_v2_balanced.pt

set -euo pipefail
exec sudo powermetrics \
  --samplers smc,gpu_power,cpu_power \
  -i 1000 \
  -n 30
