# Analytics Module

Ce dossier contient le pipeline analytics local SafePerform.

## Objectif

Produire des features fiables (daily/weekly), alertes et rapports a partir des donnees SQL SafePerform, puis exposer ces resultats a l'API.
Sources actuellement branchees:
- wellness (fatigue/stress via `sport_est_critere` + `critere`),
- GPS (table equipe, par defaut `GPS_18`).

## Structure

- `config/`: configuration runtime et logging.
- `src/`: code source ETL (extract/transform/load), alertes, rapports, utilitaires.
- `jobs/`: points d'entree executables (batch quotidien, rebuild cible).
- `tests/`: tests unitaires et integration.
- `scripts/`: scripts shell de lancement local.

## Demarrage rapide

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r analytics/requirements.txt
python analytics/jobs/run_daily_pipeline.py --dry-run
```

## RAG local (retrieval)

```bash
python analytics/jobs/build_rag_index.py --days 90
python analytics/jobs/query_rag.py --question "Quels joueurs ont une charge GPS élevée avec fatigue ?" --top-k 5
```

## Q/A avec MLXServe (Docker / hôte)

En stack `deploy-docker/docker-compose*.yml`, le conteneur **`analytics-qa`** enchaîne automatiquement **pipeline → index RAG → uvicorn** au démarrage (volume `safeperform_rag`). Si `ANALYTICS_AUTO_REFRESH=1`, une boucle relance **le même enchaînement** (données + index à jour) tous les `ANALYTICS_REFRESH_SECONDS` (défaut **86400** = 24 h ; réglable, ex. 21600 pour 6 h).

Prérequis : **MLXServe** joignable (API OpenAI-compatible) :

- `MLXSERVE_BASE_URL` — ex. `http://host.docker.internal:8088` depuis un conteneur vers le Mac hôte, ou `http://127.0.0.1:8088` en local.
- `MLXSERVE_MODEL` — ex. `mlx-community/Qwen2.5-7B-Instruct-4bit` (défaut aligné sur MLXServe).
- `MLXSERVE_API_KEY` — optionnel, si le serveur impose une clé (`Authorization: Bearer …`).

`MLXSERVE_TIMEOUT_S` remplace l’ancien `OLLAMA_TIMEOUT_S` (toujours lu en secours pour compat).

Debug : `GET http://localhost:8090/health`, `POST http://localhost:8090/qa`.

Hors Docker (dev) : construire l’index manuellement puis `query_rag.py` comme ci-dessus.

## Avec Docker (MySQL sur l’hôte)

Quand la stack tourne avec `SAFEPERFORM_DB_PORT=3307` :

- `DB_HOST=127.0.0.1`, `DB_PORT=3307`, `DB_USER` / `DB_PASS` / `DB_NAME` comme dans `.env.docker`.
- Appliquer le schéma si volume déjà existant :  
  `mysql -h 127.0.0.1 -P 3307 -u root -p ... < deploy-docker/initdb/10_analytics_schema.sql`

Optionnel : `ANALYTICS_CRITERE_EQUIPE=18` pour filtrer les critères wellness (colonne `critere.equipe`), aligné sur certaines APIs PHP historiques.
Optionnel : `ANALYTICS_GPS_TABLE=GPS_18` et `ANALYTICS_EQUIPE_ID=18` pour la source GPS.
