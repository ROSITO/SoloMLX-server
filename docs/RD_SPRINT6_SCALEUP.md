# Sprint 6 - Scale-up 7B (decision + execution plan)

## Decision

Gate pre-scale-up valide:
- source eval: `bench/moe_training/eval_ab_report_adapter_v2_mps.json`
- decision: `bench/moe_training/scaleup_gate.json`
- resultat: `go_scaleup = true`
- cible: `mistralai/Mistral-7B-Instruct-v0.3` (Apache-2.0)

## Plan execution

1. Faire un bridge partiel MoE sur 1 couche MLP (strategie conservative).
2. Lancer un training adapter court sur GPU Apple (`mps`) avec batch faible.
3. Eval A/B sur set v1 + set etendu.
4. Ajuster top-k/shared/layers si regression latence.

## Risques

- OOM GPU/Unified Memory sur 7B.
- Temps d'iteration plus long.
- Degradation qualite si bridge trop agressif.

## Mitigation

- commencer avec une seule couche convertie
- `top_k=1`, `shared_experts=0`
- steps courts (20-80) avant run plus long
