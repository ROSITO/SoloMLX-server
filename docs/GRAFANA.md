# Grafana / Prometheus — MLXServe

MLXServe expose des métriques texte sur `GET /metrics` (format Prometheus).

## Scrape minimal (`prometheus.yml`)

```yaml
scrape_configs:
  - job_name: mlxserve
    scrape_interval: 15s
    static_configs:
      - targets: ["127.0.0.1:8080"]
```

Si une clé API est activée, Prometheus ne peut pas s’authentifier seul : exposez `/metrics` uniquement sur un réseau de confiance, ou placez un reverse proxy qui injecte l’en-tête `Authorization`, ou désactivez la clé pour le scrape interne.

## Panneau suggéré

- **Série** : `mlxserve_memory_zone` (0 vert, 1 jaune, 2 rouge) et `mlxserve_macos_memory_pressure`.
- **Charge** : `rate(mlxserve_chat_requests_total[5m])` (ajustez le nom si vous utilisez uniquement les compteurs labellisés).
- **Refus mémoire** : `mlxserve_memory_chat_denied_total` et `mlxserve_memory_chat_denied_by_reason_total{reason="…"}` (codes : `macos_memory_pressure_critical`, `projected_over_hard_limit_gb`, etc.).

Les compteurs `mlxserve_chat_completions_by_model_total` peuvent exploser en cardinalité si les clients envoient des `model` arbitraires : désactivez-les avec `MLXSERVE_METRICS_LABEL_CHAT_BY_MODEL=0` si besoin.
