# Phase 2 - Proto MoE minimal (demarrable maintenant)

Ce document decrit le prototype implemente pour commencer l'option 2 sans attendre un entrainement lourd.

## Ce qui est implante

- Nouveau backend runtime: `moe_stub`
  - classe: `ExperimentalMoEStubBackend`
  - routing deterministe vers experts "specialises" (code/math/reasoning/general)
  - experts partages simulés
  - compatible API chat existante (non-stream + stream)
- Parametres de config:
  - `MLXSERVE_RUNTIME_BACKEND=moe_stub`
  - `MLXSERVE_MOE_NUM_EXPERTS` (defaut: 4)
  - `MLXSERVE_MOE_TOP_K` (defaut: 2)
  - `MLXSERVE_MOE_NUM_SHARED_EXPERTS` (defaut: 1)

## Pourquoi ce proto

- Valider integration A/B et instrumentation avant investissement ML réel.
- Tester "pipeline MoE" (routing, fallback, benchmark) avec risque faible.
- Preparer l'insertion future d'un backend MoE entraine.

## Commandes utiles

Demarrer serveur en proto MoE:

```bash
MLXSERVE_RUNTIME_BACKEND=moe_stub .venv/bin/mlxserve serve
```

Benchmark mock backend courant:

```bash
.venv/bin/python scripts/bench_chat.py --mock
```

Comparaison A/B `stub` vs `moe_stub`:

```bash
.venv/bin/python scripts/bench_ab.py
```

## Artefacts attendus

- `bench/results_stub.json`, `bench/results_stub.csv`
- `bench/results_moe_stub.json`, `bench/results_moe_stub.csv`
- `bench/ab_report.json`

## Limites connues

- Ce backend n'est pas un vrai modele MoE entraine.
- Qualite linguistique non representative.
- L'objectif est la validation d'architecture et de process.

## Prochaine sous-phase (quand tu valides)

1. Brancher un backend MoE "real model" (petit) derriere la meme interface.
2. Garder `scripts/bench_ab.py` comme gate automatique.
3. Ajouter un gate qualite (evaluation humaine/A-B sur prompts fixes).
