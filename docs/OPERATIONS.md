# OPERATIONS — MLXServe

Guide court pour exploiter MLXServe en local ou derrière un proxy. Pour l’installation et l’API complète, voir le [`README.md`](../README.md).

---

## Démarrage standard

```bash
cd /chemin/vers/MLXserve
./scripts/start_server.sh
```

Ou, avec le venv déjà créé :

```bash
.venv/bin/mlxserve serve
```

Contrôle des variables (exemple) :

```bash
MLXSERVE_PORT=8080 \
MLXSERVE_RUNTIME_BACKEND=mlx \
.venv/bin/mlxserve serve
```

---

## Vérifications rapides

### Santé et zone mémoire

```bash
curl -s http://127.0.0.1:8080/health
```

Réponse attendue (exemple) :

```json
{"status":"ok","memory_zone":"green"}
```

### Métriques (Prometheus texte)

```bash
curl -s http://127.0.0.1:8080/metrics | head -40
```

### Modèle annoncé par l’API

```bash
curl -s http://127.0.0.1:8080/v1/models
```

Si une clé API est configurée (`MLXSERVE_API_KEY`) :

```bash
curl -s http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer VOTRE_CLE"
```

---

## Test chat (non stream)

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Bonjour"}],
    "max_tokens": 96
  }'
```

---

## Interface web

L’application sert une UI sur la racine :

```text
http://127.0.0.1:8080/
```

Captures d’écran à jour : répertoire [`screenshots/`](screenshots/) (voir aussi [`docs/README.md`](README.md)).

---

## Débogage express

1. **`HTTP 503` sur `/v1/chat/completions`** — Souvent zone mémoire **`red`** : voir `/health`, réduire `max_tokens`, modèle plus petit, ou assouplir `MLXSERVE_MAX_MEMORY_GB` / `MLXSERVE_HARD_MEMORY_GB` en test.
2. **`HTTP 401`** — `MLXSERVE_API_KEY` défini : ajouter `Authorization: Bearer …` (UI : champ « Clé API »).
3. **`HTTP 429`** — Rate limit : attendre ou augmenter `MLXSERVE_RATE_LIMIT_PER_MINUTE`.
4. **Téléchargement HF bloqué** — Vérifier réseau ; variables Hugging Face (`HF_*`) si besoin.

---

## Liens utiles

- [README.md](../README.md) — référence principale
- [REVERSE_PROXY.md](REVERSE_PROXY.md) — exposition sécurisée
- [CHAT_TRANSCRIPT_OUTPUT.md](CHAT_TRANSCRIPT_OUTPUT.md) — comportement du chat
