# REVERSE PROXY — MLXServe

Référence complète du serveur : [`README.md`](../README.md). Index des guides : [`docs/README.md`](README.md).

## Caddy (LAN / HTTPS local)

Exemple `Caddyfile`:

```caddyfile
mlxserve.local {
  reverse_proxy 127.0.0.1:8080
}
```

## Cloudflare Tunnel (exposition distante)

Configurer le tunnel vers `http://127.0.0.1:8080` puis limiter l'accès avec API key MLXServe.

## Tailscale Funnel

Publier le service local et conserver `MLXSERVE_API_KEY` actif.

## Bonnes pratiques

- Toujours activer API key hors localhost.
- Limiter les origines CORS (`MLXSERVE_CORS_ALLOW_ORIGINS`).
- Conserver le rate limit actif en frontal et côté application.
