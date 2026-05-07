# Documentation MLXServe

Index des guides et ressources hors [`README.md`](../README.md) racine.

## Guides

| Document | Description |
|----------|-------------|
| [OPERATIONS.md](OPERATIONS.md) | Démarrage, vérifications `curl`, dépannage rapide |
| [REVERSE_PROXY.md](REVERSE_PROXY.md) | Exposition derrière un reverse proxy (Caddy, etc.) |
| [CHAT_TRANSCRIPT_OUTPUT.md](CHAT_TRANSCRIPT_OUTPUT.md) | Format des prompts chat, sorties type transcript, blocs code |

## Captures d’écran

Fichiers dans [`screenshots/`](screenshots/) :

| Fichier | Contenu |
|---------|---------|
| `ui-desktop.png` | Vue principale du chat (pilule modèle, compositeur) |
| `ui-settings-catalog.png` | Panneau réglages et listes de modèles |

Ces images sont référencées depuis le README principal pour GitHub / forge.

## API OpenAI-compatible (rappel)

- `GET /v1/models` — modèle par défaut / chargé (identifiant lisible `org/repo` lorsque c’est un chemin de cache HF)
- `GET /v1/models/recommended` — catalogue calibré RAM
- `GET /v1/models/local` — inventory locale
- `POST /v1/chat/completions` — chat JSON ou SSE
- `GET /health`, `GET /metrics`

Détails et exemples : section **API** du [`README.md`](../README.md).
