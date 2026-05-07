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
  - `GET /metrics`
  - `GET /v1/models`
  - `GET /v1/models/local`
  - `GET /v1/models/recommended`
  - `POST /v1/models/pull`
  - `DELETE /v1/models/{model_alias}`
  - `POST /v1/chat/completions` (stream SSE et non-stream)
- Validation des requetes via Pydantic
- API key supportee en mode Bearer (configurable)
- Memory guardian actif (zones green/yellow/red)
- Politique d'unload idle modele
- Security middleware:
  - rate limiting
  - headers securite
  - CORS configurable
- CLI:
  - `mlxserve` / `mlxserve serve`
  - `mlxserve autotune`
  - `mlxserve models-list`
  - `mlxserve models-pull --model ...`
  - `mlxserve models-rm --model ...`

## Installation

### 1) Setup environnement

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

### 2) Installer MLX runtime

```bash
.venv/bin/python -m pip install -e ".[mlx]"
```

## Lancement serveur

```bash
.venv/bin/mlxserve serve
# ou
./scripts/start_server.sh
```

Variables utiles:

- `MLXSERVE_HOST` (defaut `127.0.0.1`)
- `MLXSERVE_PORT` (defaut `8080`)
- `MLXSERVE_API_KEY`
- `MLXSERVE_RUNTIME_BACKEND` (`auto`, `mlx`, `stub`)
- `MLXSERVE_DEFAULT_MODEL`
- `MLXSERVE_CORS_ALLOW_ORIGINS` (comma-separated, ex: `http://localhost:3000,http://127.0.0.1:3000`)
- `MLXSERVE_RATE_LIMIT_PER_MINUTE`
- `MLXSERVE_PREFILL_STEP_SIZE`
- `MLXSERVE_KV_BITS`
- `MLXSERVE_KV_GROUP_SIZE`
- `MLXSERVE_QUANTIZED_KV_START`

## Optimisations runtime (token/s)

Optimisations implementees:

- streaming natif temps reel (plus de buffer complet avant emission)
- parametres generation propages (`temperature`, `top_p`)
- support tuning MLX:
  - `MLXSERVE_PREFILL_STEP_SIZE`
  - `MLXSERVE_KV_BITS`
  - `MLXSERVE_KV_GROUP_SIZE`
  - `MLXSERVE_QUANTIZED_KV_START`

Valeurs actuelles par defaut:

- `MLXSERVE_PREFILL_STEP_SIZE=2048`
- `MLXSERVE_KV_BITS=4`
- `MLXSERVE_KV_GROUP_SIZE=64`
- `MLXSERVE_QUANTIZED_KV_START=32`

## Autotune performance

Commande:

```bash
.venv/bin/mlxserve autotune --model "mlx-community/Mistral-7B-Instruct-v0.3-4bit" --max-tokens 96
```

Ce que fait l'autotune:

- charge le modele
- teste plusieurs combinaisons (`prefill_step_size`, `kv_bits`, `kv_group_size`, `quantized_kv_start`)
- classe les runs par `generation_tps` puis `ttft`
- retourne un JSON avec:
  - meilleur profil
  - tous les runs
  - variables d'environnement recommandees

## Model Manager (local inventory)

Gestion locale des modeles dans `~/.mlxserve/models/registry.json`.

Commandes:

```bash
# lister modeles locaux suivis par MLXServe
.venv/bin/mlxserve models-list

# pull un modele HF (telechargement + enregistrement inventory)
.venv/bin/mlxserve models-pull --model "mlx-community/Mistral-7B-Instruct-v0.3-4bit"

# supprimer de l'inventory + tentative de cleanup cache HF
.venv/bin/mlxserve models-rm --model "mistral-7b-instruct-v0.3-4bit"
```

API:

- `GET /v1/models/local`
- `POST /v1/models/pull` avec `{"model":"<hf_repo>"}`
- `DELETE /v1/models/{model_alias}`

## Security et observabilite

Security active:

- rate limiting par client (fenetre 60s)
- headers: `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Cache-Control`
- CORS configurable

Observabilite:

- endpoint `GET /metrics` (format texte Prometheus)
- compteurs:
  - requests total
  - chat requests total
  - errors total
  - rate-limited total
  - generated tokens total (estime)
  - latency moyenne
  - latency p95 recente
  - memory used/swap used
  - memory zone
  - generation tps observe

## Commandes utiles

```bash
# lancer le serveur (backend auto)
.venv/bin/mlxserve

# forcer backend MLX
MLXSERVE_RUNTIME_BACKEND=mlx .venv/bin/mlxserve serve

# tests
.venv/bin/python -m pytest -q

# benchmark rapide natif mlx-lm (exemple)
.venv/bin/python - <<'PY'
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
import time
m,t = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
sampler = make_sampler(temp=0.2, top_p=0.95)
s=time.time(); last=None
for r in stream_generate(m,t,prompt="bench",max_tokens=96,sampler=sampler,prefill_step_size=2048,kv_bits=4,kv_group_size=64,quantized_kv_start=32):
    last=r
print("generation_tps:", round(last.generation_tps,2), "elapsed:", round(time.time()-s,3))
PY
```

## Progression roadmap (etat actuel)

- Phase 0 Foundation: **done**
- Phase 1 MVP API/Inference: **done**
- Phase 2 Memory Guardian: **done (RAM/swap/pressure + hysteresis + anti-swap guardrails)**
- Phase 3 Model Manager: **done (inventory pull/list/rm + metadata size/quantization)**
- Phase 4 Security baseline: **done (rate limit + headers + CORS)**
- Phase 5 Observability: **done (metrics endpoint + p95 + memory + ops docs + startup script)**

## Runbook / Ops docs

- `docs/OPERATIONS.md`
- `docs/REVERSE_PROXY.md`

## Tests

- Suite `pytest` locale
- Couverture MVP:
  - health
  - models
  - chat non-stream
  - chat stream
  - securite API key
  - logique memory guardian
  - parsing CLI

## Notes

Le runtime supporte maintenant un backend selectionnable:
- `MLXSERVE_RUNTIME_BACKEND=auto` (defaut, `mlx-lm` si dispo sinon fallback stub)
- `MLXSERVE_RUNTIME_BACKEND=mlx` (force `mlx-lm`)
- `MLXSERVE_RUNTIME_BACKEND=stub` (mode simulation)

La liste `/v1/models/recommended` propose des modeles Hugging Face calibres en fonction de la RAM machine.
