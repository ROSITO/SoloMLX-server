# Sprint 3 - Stabilisation MoE (smoke training)

Objectif: valider rapidement la stabilite forward/backward d'un bloc MoE top-k avec regularisation router avant entrainement complet.

## Scripts

- `training/moe_stabilize.py`:
  - teacher dense synthetique (frozen)
  - student MoE (`TopKMoEFFN`)
  - loss = reconstruction MSE + balancing + penalite compute proxy
  - sortie JSON de diagnostics

## Commande

```bash
.venv/bin/python -m training.moe_stabilize --steps 200 \
  --out bench/moe_training/stabilize_report.json
```

## Signaux de validation

- `loss_delta` negatif (idealement)
- absence de NaN/OOM
- `expert_usage_avg` non completement collapse
- `router_entropy_avg` > 0 (router actif)

## Limites

- Bridge checkpoint reel ajoute via `training/moe_bridge_smoke.py`.
- Prochaine etape: run plus long et eval qualite sur set v1.

## Resultat run actuel (200 steps)

- `loss_start`: 0.0924
- `final_loss`: 0.0192
- `loss_delta`: -0.0732
- `router_entropy_avg`: 1.31
- `expert_usage_avg`: quasi uniforme (environ 25% par expert)

Artefact:
- `bench/moe_training/stabilize_report.json`

## Bridge reel (dense checkpoint -> couches MoE)

Commande:

```bash
.venv/bin/python -m training.moe_bridge_smoke \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --layers 2,6,10 \
  --steps 20 \
  --out bench/moe_training/bridge_smoke_report.json
```

Resultat courant:
- `loss_start`: 13.94
- `final_loss`: 11.52
- `loss_delta`: -2.42
- `router_entropy_avg`: 1.386
- `expert_usage_avg`: ~25% par expert
- `seconds_per_step`: ~0.39s CPU
