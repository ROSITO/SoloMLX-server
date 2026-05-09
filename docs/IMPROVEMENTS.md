# Backlog d’amélioration — MLXServe

Liste priorisée (stabilité → mémoire → perf → compat API → simplicité). Cocher au fil des PRs.

## Mémoire et admission

- [x] Métrique Prometheus : refus chat pour pression mémoire (`mlxserve_memory_chat_denied_total`)
- [x] Jauge pression macOS (`memory_pressure -Q` exposée comme `mlxserve_macos_memory_pressure`)
- [x] Estimation pré-admission affinée (heuristique tokens `chars/4` + `max_tokens`, plafonnée — voir `MLXSERVE_MEMORY_ADMISSION_*`)
- [x] Politique d’unload idle : `MLXSERVE_IDLE_UNLOAD_ENABLED` (désactive la décharge automatique)

## API et compatibilité OpenAI

- [x] `usage.prompt_tokens` / `completion_tokens` via tokenizer quand le backend le permet (non-stream)
- [x] `stream_options.include_usage` + chunk final SSE (`usage`, `finish_reason`)
- [x] Séquences d’arrêt (`stop` string ou liste), post-traitement aligné OpenAI (stop exclu du texte)
- [x] Enveloppe d’erreur type OpenAI sur `/v1/*` : `{"error":{"message","type","code"}}`

## Observabilité

- [x] Corrélation légère : `X-Request-ID` + logs (alternative légère à OpenTelemetry complet)
- [x] Recette Grafana / scrape : [GRAFANA.md](GRAFANA.md)
- [x] Compteurs labellisés : `mlxserve_chat_completions_labeled_total{memory_zone=…}`, `mlxserve_chat_completions_by_model_total{model=…}` (désactivables)

## Sécurité

- [x] JWT HS256 optionnel (`MLXSERVE_JWT_HS256_SECRET`, audience optionnelle) — installer `pip install -e ".[security]"` ou PyJWT
- [ ] RBAC / rôles fins (hors périmètre court terme, voir `AGENTS.md`)
- [x] Rate limit par jeton Bearer (hash SHA-256 de la valeur) vs IP anonyme

## UI

- [x] Pull modèle HF + suppression d’une entrée locale (panneau réglages)
- [x] Markdown : liens `[texte](url)`, tableaux GFM simples

## DevOps

- [x] Esquisse Homebrew : [INSTALL_HOMEBREW.md](INSTALL_HOMEBREW.md)
- [x] Exemple `launchd` : [OPERATIONS.md](OPERATIONS.md)

## Qualité

- [x] Test d’intégration minimal MLX (`tests/test_integration_mlx.py`, skip si `mlx-lm` absent)
- [x] Tests streaming (`finish_reason`, `usage`, enveloppe erreur 503)

## Documentation produit

- [x] `MEMORY.md` ignoré par git + modèle [MEMORY.md.example](../MEMORY.md.example) (notes locales sans secrets)

---

*Dernière mise à jour : couverture large du backlog (JWT, stop, stream usage, UI, doc, métriques labellisées).*
