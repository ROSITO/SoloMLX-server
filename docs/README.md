# Documentation MLXServe

Index des guides et ressources hors [`README.md`](../README.md) racine.

## Guides

| Document | Description |
|----------|-------------|
| [OPERATIONS.md](OPERATIONS.md) | Démarrage, vérifications `curl`, dépannage rapide |
| [REVERSE_PROXY.md](REVERSE_PROXY.md) | Exposition derrière un reverse proxy (Caddy, etc.) |
| [CHAT_TRANSCRIPT_OUTPUT.md](CHAT_TRANSCRIPT_OUTPUT.md) | Format des prompts chat, sorties type transcript, blocs code |
| [GRAFANA.md](GRAFANA.md) | Prometheus / Grafana, cardinalité des labels |
| [INSTALL_HOMEBREW.md](INSTALL_HOMEBREW.md) | Modèle de formule Homebrew |
| [IMPROVEMENTS.md](IMPROVEMENTS.md) | Backlog produit |
| [BENCH_PHASE1.md](BENCH_PHASE1.md) | Exécution/validation de la baseline benchmark |
| [MOE_PHASE2_PROTO.md](MOE_PHASE2_PROTO.md) | Proto MoE minimal et protocole A/B |
| [RD_DENSE_TO_MOE_PLAN.md](RD_DENSE_TO_MOE_PLAN.md) | Plan R&D ambitieux Dense -> MoE -> MLX |
| [RD_DECISIONS.md](RD_DECISIONS.md) | Décisions Sprint 1 (cost-first) |
| [RD_SPRINTS_STATUS.md](RD_SPRINTS_STATUS.md) | Statut des premiers sprints R&D |
| [RD_CONVERSION_RUNBOOK.md](RD_CONVERSION_RUNBOOK.md) | Procédure de conversion Dense->MoE sur checkpoint réel |
| [RD_SPRINT3_STABILIZATION.md](RD_SPRINT3_STABILIZATION.md) | Stabilisation MoE (smoke training torch) |
| [RD_SPRINT4_EVAL.md](RD_SPRINT4_EVAL.md) | Eval A/B dense vs MoE bridge |
| [RD_SPRINT5_TARGET_TRAINING.md](RD_SPRINT5_TARGET_TRAINING.md) | Entrainement cible court (adapter MoE) |
| [RD_SPRINT6_SCALEUP.md](RD_SPRINT6_SCALEUP.md) | Décision et plan de scale-up 7B |
| [RD_MOE_MLX_SERVE.md](RD_MOE_MLX_SERVE.md) | MoE natif mlx-lm (Mixtral) + bench MLX, alignement cost-first |
| [RD_SCALEUP_20_30B.md](RD_SCALEUP_20_30B.md) | Scale-up MoE custom ~24B / ~32B (Mistral Small, Qwen2.5) |
| [RD_PIPELINE_MLX_INFERENCE_16GB.md](RD_PIPELINE_MLX_INFERENCE_16GB.md) | Pipeline MoE → MLX 4-bit, bench coût, plafond **16 GiB** |
| [RD_INFERENCE_OPTIMIZATION_ROADMAP.md](RD_INFERENCE_OPTIMIZATION_ROADMAP.md) | Roadmap P0–P3 (mémoire, KV, MoE, speculative) vs recherche externe |
| [MOE_OFFLOAD.md](MOE_OFFLOAD.md) | Offload experts MLX au load : comportement, limites, réglages |

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
