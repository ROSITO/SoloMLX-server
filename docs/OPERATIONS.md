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

## LaunchAgent macOS (service en arrière-plan)

Le terminal n’est plus bloqué : un **LaunchAgent** utilisateur relance MLXServe à l’ouverture de session (avec logs).

Depuis la racine du clone, après `pip install -e ".[mlx]"` :

```bash
./scripts/install_launchagent_macos.sh install
```

Désinstaller :

```bash
./scripts/install_launchagent_macos.sh uninstall
```

Le dépôt par défaut est `~/Documents/MLXServe/SoloMLX-server` ; sinon :  
`SOLOMLX_ROOT=/chemin/vers/SoloMLX-server ./scripts/install_launchagent_macos.sh install`  
Les variables `MLXSERVE_*` peuvent être exportées **avant** la commande pour surcharger les défauts du script.

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

## Service `launchd` (démarrage au login)

Exemple de plist **à adapter** (chemins utilisateur, clé API via fichier env non versionné) :

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.example.mlxserve</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/VOTRE_USER/MLXserve/.venv/bin/mlxserve</string>
    <string>serve</string>
    <string>--host</string>
    <string>127.0.0.1</string>
    <string>--port</string>
    <string>8080</string>
  </array>
  <key>WorkingDirectory</key>
  <string>/Users/VOTRE_USER/MLXserve</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/Users/VOTRE_USER/Library/Logs/mlxserve.out.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/VOTRE_USER/Library/Logs/mlxserve.err.log</string>
</dict>
</plist>
```

Installation :

```bash
cp com.example.mlxserve.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.example.mlxserve.plist
```

---

## Liens utiles

- [README.md](../README.md) — référence principale
- [REVERSE_PROXY.md](REVERSE_PROXY.md) — exposition sécurisée
- [CHAT_TRANSCRIPT_OUTPUT.md](CHAT_TRANSCRIPT_OUTPUT.md) — comportement du chat
- [GRAFANA.md](GRAFANA.md) — scrape Prometheus / Grafana
- [INSTALL_HOMEBREW.md](INSTALL_HOMEBREW.md) — esquisse de formule Homebrew
