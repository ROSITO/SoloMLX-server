# Sprint 5 - Entrainement cible court (MoE adapter)

Objectif: recuperer la qualite du bridge MoE sans full fine-tuning du modele complet.

## Scripts

- `training/moe_target_train.py`
  - charge checkpoint dense
  - remplace couches cible par MoE
  - entraine seulement les params MoE (adapter-like)
  - sauvegarde:
    - adapter MoE (`.pt`)
    - rapport training (`.json`)

- `training/moe_eval_ab.py`
  - peut charger `--adapter-path` pour eval A/B avec l'adapter entraine
  - `--adapter-path-shrunk` + inference `num_experts` depuis le `.pt` reduit: eval **tri-banches** (dense / MoE plein / MoE shrunk)

- `training/moe_model.py`
  - `shrink_moe_state_dict`, `infer_num_experts_from_moe_adapter_state`

- `tools/shrink_moe_adapter_checkpoint.py`
  - shrink offline d'un `.pt` exporte

- Scripts pipeline
  - `scripts/run_moe_expert_ticket_smoke.sh` (defaut SmolLM + CPU, variables d'env)
  - `scripts/run_moe_expert_ticket_mistral7b_mps.sh` (Mistral-7B + MPS + bf16/offload ; eval via **split**)
  - `scripts/run_moe_eval_ab_tri_split.sh` : deux `moe_eval_ab` + `training.merge_moe_eval_ab_tri` (contourne OOM MPS)
  - `EVAL_TRI_SPLIT=1` avec le smoke script force ce mode d'eval

## Expert ticket (concentration routeur + shrunk)

Objectif: preparer un adaptateur **plus petit** (moins d'experts) tout en gardant la qualite, avant MLX.

Train avec perte d'entropie du routeur (pas de balance contradictoire) + export automatique shrunk:

```bash
.venv/bin/python -m training.moe_target_train \
  --model-id mistralai/Mistral-7B-Instruct-v0.2 \
  --layers 10 \
  --num-experts 8 \
  --router-balance-weight 0 \
  --router-entropy-weight 0.02 \
  --shrink-to-experts 4 \
  --device mps \
  --dtype bf16 \
  --low-cpu-mem-usage \
  --offload-unused-experts \
  --out-adapter bench/moe_training/moe_ticket_full.pt \
  --out-report bench/moe_training/moe_ticket_train.json
```

Eval dense vs MoE 8-experts vs MoE 4-experts (fichier `*_shrunk4.pt` genere par `--shrink-to-experts`):

```bash
.venv/bin/python -m training.moe_eval_ab \
  --model-id mistralai/Mistral-7B-Instruct-v0.2 \
  --layers 10 \
  --num-experts 8 \
  --adapter-path bench/moe_training/moe_ticket_full.pt \
  --adapter-path-shrunk bench/moe_training/moe_ticket_full_shrunk4.pt \
  --device mps \
  --dtype bf16 \
  --low-cpu-mem-usage \
  --offload-unused-experts \
  --out bench/moe_training/moe_ticket_eval_ab.json
```

Pipeline d'un coup (memes variables d'env que le script smoke):

```bash
bash scripts/run_moe_expert_ticket_mistral7b_mps.sh
```

## Commandes

Train adapter:

```bash
.venv/bin/python -m training.moe_target_train \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --layers 10 \
  --steps 120 \
  --shared-experts 0 \
  --out-adapter bench/moe_training/moe_adapter_v1.pt \
  --out-report bench/moe_training/moe_target_train_report.json
```

Eval A/B avec adapter:

```bash
.venv/bin/python -m training.moe_eval_ab \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompts bench/eval_quality_v1.json \
  --layers 10 \
  --shared-experts 0 \
  --warmup-steps 0 \
  --adapter-path bench/moe_training/moe_adapter_v1.pt \
  --out bench/moe_training/eval_ab_report_adapter_v1.json
```

## Run v1 (etat courant)

- training:
  - `loss_start`: 6.95
  - `final_loss`: 5.17
  - `loss_delta`: -1.78
  - report: `bench/moe_training/moe_target_train_report.json`

- eval A/B:
  - dense avg_loss: 5.18
  - moe_bridge avg_loss: 5.07  (mieux)
  - delta avg_loss: -0.109
  - delta p95 latency: +37.41 ms (moins bon)
  - report: `bench/moe_training/eval_ab_report_adapter_v1.json`

Interpretation:
- l'adapter MoE recupere (et depasse legerement) la qualite proxy.
- la latence est encore en regression; optimisation runtime supplementaire requise avant gate production.

## Run v2 GPU Apple (MPS)

Training (300 steps, `device=mps`):
- `loss_start`: 6.97
- `final_loss`: 5.57
- `loss_delta`: -1.40
- adapter: `bench/moe_training/moe_adapter_v2_mps.pt`
- report: `bench/moe_training/moe_target_train_report_v2_mps.json`

Eval A/B (MPS, layers=10):
- dense avg_loss: 5.178
- moe avg_loss: 5.166 (mieux)
- delta avg_loss: -0.012
- dense p95: 2516.9 ms
- moe p95: 2344.4 ms
- delta p95: -172.5 ms (mieux)
- report: `bench/moe_training/eval_ab_report_adapter_v2_mps.json`

Conclusion v2:
- gate qualite: atteint (>= dense)
- gate latence: atteint (meilleur p95)
- gate adapter export/reload: atteint
