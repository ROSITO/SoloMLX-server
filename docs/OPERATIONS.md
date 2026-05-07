# OPERATIONS — MLXServe

## Démarrage standard

```bash
./scripts/start_server.sh
```

## Vérifications rapides

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/metrics
```

## Test chat local

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "messages":[{"role":"user","content":"Bonjour"}],
    "max_tokens":96
  }'
```

## Debug incident rapide

1. Vérifier `/health` et zone mémoire.
2. Lire `/metrics` pour `mlxserve_errors_total` et `mlxserve_rate_limited_total`.
3. Réduire `max_tokens` et/ou choisir un modèle plus léger.
4. Si pression mémoire persistante, unload/restart du service.
