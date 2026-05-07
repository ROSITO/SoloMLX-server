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
  - `GET /v1/models/recommended`
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

Le runtime supporte maintenant un backend selectionnable:
- `MLXSERVE_RUNTIME_BACKEND=auto` (defaut, `mlx-lm` si dispo sinon fallback stub)
- `MLXSERVE_RUNTIME_BACKEND=mlx` (force `mlx-lm`)
- `MLXSERVE_RUNTIME_BACKEND=stub` (mode simulation)

La liste `/v1/models/recommended` propose des modeles Hugging Face calibres en fonction de la RAM machine.
