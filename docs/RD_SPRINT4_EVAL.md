# Sprint 4 - Eval A/B Dense vs MoE Bridge

Objectif: comparer rapidement un modele dense vs sa variante MoE-bridge sur un set fixe.

## Script

- `training/moe_eval_ab.py`
  - charge modele dense de reference
  - construit variante MoE bridge (couches remplacees)
  - warmup court MoE (optionnel)
  - evalue:
    - avg loss
    - p95 latency
    - sorties texte (samples)

## Commande

```bash
.venv/bin/python -m training.moe_eval_ab \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompts bench/eval_quality_v1.json \
  --layers 2,6,10 \
  --warmup-steps 10 \
  --out bench/moe_training/eval_ab_report.json
```

## Sorties

- `bench/moe_training/eval_ab_report.json`
- champs principaux:
  - `dense.avg_loss`, `dense.p95_latency_ms`
  - `moe_bridge.avg_loss`, `moe_bridge.p95_latency_ms`
  - `delta`

## Run actuel (v1)

- dense:
  - `avg_loss`: 5.18
  - `p95_latency_ms`: 632.81
- moe_bridge:
  - `avg_loss`: 5.93
  - `p95_latency_ms`: 664.13
- delta:
  - `avg_loss`: +0.75 (moins bon)
  - `p95_latency_ms`: +31.33 ms (moins bon)

Interpretation:
- le bridge MoE v1 n'est pas encore au niveau dense.
- decision actuelle: **No-Go provisoire pour scale-up training**, poursuivre tuning Sprint 3/4 (routing, warmup, couches converties, regularisation).

## Tuning rapide (variantes)

- Variante A: `layers=10`, `shared_experts=0`
  - delta loss: +1.39 (moins bon)
  - delta p95: -27.16 ms (mieux)
  - fichier: `bench/moe_training/eval_ab_report_l10_s0.json`

- Variante B: `layers=6,10`, `shared_experts=0`
  - delta loss: +1.08 (moins bon)
  - delta p95: -148.18 ms (mieux)
  - fichier: `bench/moe_training/eval_ab_report_l610_s0.json`

- Variante C: `layers=2,6,10`, `shared_experts=0`, warmup=20
  - delta loss: +3.94 (beaucoup moins bon)
  - delta p95: -359.79 ms (mieux)
  - fichier: `bench/moe_training/eval_ab_report_l2610_s0_w20.json`

Conclusion tuning:
- tendance observee: la latence s'ameliore quand on rend la MoE plus sparse, mais la qualite (proxy loss) se degrade trop.
- prochaine etape necessaire: entrainement cible reelle (SFT/CPT court) avec regularisation router plus stricte pour recuperer la qualite.

## Correctif routing + observation critique

Un correctif de normalisation des poids top-k a ete applique dans `training/moe_model.py`.

Verification importante:
- config `shared_experts=0`, `warmup_steps=0`:
  - delta loss: `0.0` (parite dense atteinte)
  - delta p95: `+66.49 ms` (regression latence)
  - fichier: `bench/moe_training/eval_ab_report_after_gatefix_s0_w0.json`

Interpretation:
- a l'instant T, le bridge MoE est **fonctionnel et reel** (pas simule), avec parite qualite possible a l'initialisation.
- la latence n'est pas encore competitive; l'optimisation runtime/dispatch reste necessaire.

## Optimisation dispatch sparse (etat courant)

Le chemin de calcul MoE a ete optimise pour n'executer que les tokens routés par expert (au lieu de calculer tous les experts sur tous les tokens).

Resultats:

- config `layers=2,6,10`, `shared_experts=0`, `warmup=0`
  - delta loss: `0.0`
  - delta p95: `+12.43 ms`
  - fichier: `bench/moe_training/eval_ab_report_after_dispatch_opt.json`

- config `layers=10`, `shared_experts=0`, `warmup=0`
  - delta loss: `0.0`
  - delta p95: `-27.33 ms`
  - fichier: `bench/moe_training/eval_ab_report_after_dispatch_opt_l10.json`

Conclusion:
- gate qualite: atteint (parite)
- gate latence: atteint sur conversion partielle 1 couche (amelioration p95)
- cette configuration devient la base recommandee pour le sprint suivant (scale-up checkpoint plus grand).

## Optimisation top-k (fast path top_k=1)

Le routeur MoE utilise maintenant un chemin rapide quand `top_k=1`:
- selection directe par `argmax(logits)` (sans `topk` complet)
- poids de route fixes a 1.0 sur le chemin top-1

Resultat A/B (MPS, adapter v2):
- config `layers=10`, `shared_experts=0`, `warmup=0`
  - delta loss: `-0.012`
  - delta p95: `-581.60 ms`
  - fichier: `bench/moe_training/eval_ab_report_adapter_v2_mps_fasttop1.json`

Impact:
- qualite conservee (leger mieux sur ce set)
- reduction latence supplementaire significative
- chemin retenu comme base pour le scale-up 7B cost-first
