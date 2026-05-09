# Phase 1 - Benchmark baseline (execution guide)

Ce guide execute la phase 1 du plan `docs/MOE_MLX_ROADMAP.md`: etablir une baseline fiable avant tout prototype MoE.

## Fichiers utilises

- `bench/prompts.json`: cas de test de reference (chat general, code, raisonnement)
- `bench/gates.yaml`: seuils Go/No-Go
- `scripts/bench_chat.py`: runner benchmark non-stream
- sorties:
  - `bench/results.json`
  - `bench/results.csv`

## Etapes d'execution

1. Démarrer MLXServe (sur `http://127.0.0.1:8080` par defaut).
2. Lancer le benchmark:

```bash
.venv/bin/python scripts/bench_chat.py
```

Option locale sans serveur externe (mode mock avec TestClient):

```bash
.venv/bin/python scripts/bench_chat.py --mock
```

3. Lire le verdict:
   - code retour `0`: tous les gates passent
   - code retour `2`: au moins un gate echoue
4. Inspecter les resultats dans `bench/results.json` (resume + gates) et `bench/results.csv` (detail par cas).
5. Archiver la baseline (copie resumee) dans `bench/baselines/`.

## Personnaliser

- URL serveur:

```bash
.venv/bin/python scripts/bench_chat.py --base-url http://127.0.0.1:8081
```

- Fichiers d'entree/sortie:

```bash
.venv/bin/python scripts/bench_chat.py \
  --prompts bench/prompts.json \
  --gates bench/gates.yaml \
  --out-json bench/results.json \
  --out-csv bench/results.csv
```

- Exemple serveur reel en backend `stub` (utile pour valider le pipeline sans telechargement modele):

```bash
MLXSERVE_PORT=8082 MLXSERVE_RUNTIME_BACKEND=stub .venv/bin/mlxserve serve
.venv/bin/python scripts/bench_chat.py --base-url http://127.0.0.1:8082 \
  --out-json bench/results_http_8082_stub.json \
  --out-csv bench/results_http_8082_stub.csv
```

## Validation et tests

- Test unitaires du runner:

```bash
.venv/bin/python -m pytest tests/test_bench_chat_script.py -q
```

- Validation complete projet:

```bash
.venv/bin/python -m pytest tests/ -q
```

## Interpretation rapide

- `error_rate`: fiabilite endpoint chat
- `latency_p95_ms`: latence worst-case sur set courant
- `tokens_per_second_avg`: debit generation moyen
- `memory_denials`: refus 503 memoire observes

Ces metriques servent de baseline pour comparer un futur proto MoE.
