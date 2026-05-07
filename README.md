# SoloMLX-server

Serveur local MLX pour Apple Silicon avec une API simple et compatible OpenAI.

## Objectifs

- Stabilite
- Gestion memoire prudente
- Performance inference sur Apple Silicon
- Simplicite d'utilisation locale

## Statut

MVP backend en place et valide en tests.

## Etat actuel (MVP implemente)

- API FastAPI operationnelle
- Endpoints exposes:
  - `GET /health`
  - `GET /v1/models`
  - `POST /v1/chat/completions` (stream SSE et non-stream)
- Validation des requetes via Pydantic
- API key supportee en mode Bearer (configurable)
- Memory guardian actif (zones green/yellow/red)
- Politique d'unload idle modele
- CLI de lancement `mlxserve`

## Tests

- Suite `pytest` locale
- Couverture MVP:
  - health
  - models
  - chat non-stream
  - chat stream
  - securite API key
  - logique memory guardian

## Lancer rapidement

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/mlxserve
```

## Notes

Le moteur d'inference actuel est un backend MVP (stub) pour stabiliser le contrat API et l'observabilite de base. La prochaine etape est l'integration complete `mlx-lm` avec selection de modeles calibres machine.
