# Roadmap MOE + MLX (Go/No-Go)

Objectif: valider rapidement si une approche "MoE + MLX" apporte un gain reel pour MLXServe (qualite, latence, memoire), sans lancer un projet de recherche trop lourd trop tot.

---

## Vue d'ensemble

- Phase 1 (2 semaines): baseline solide + benchmark
- Phase 2 (4 semaines): proto MoE minimal et mesure
- Phase 3 (4-8 semaines): industrialisation si ROI confirme

Principe: pas de decision "a l'intuition". Chaque phase se termine par un gate Go/No-Go avec criteres chiffres.

---

## Phase 1 - Baseline & protocole (2 semaines)

### Livrables

- Bench reproducible pour MLXServe actuel:
  - latence p50/p95
  - tokens/s
  - RAM/swap
  - stabilite (timeouts, erreurs, refus memoire)
- Set de prompts d'evaluation fixe:
  - chat general
  - code
  - raisonnement multi-etapes court
- Script unique de benchmark:
  - entrees JSON
  - sortie CSV/JSON
  - execution locale Apple Silicon

### Definition of done

- On peut rerun le bench et retrouver des resultats coherents (+/- 5-10% max)
- Les KPI de reference sont captures et versionnes dans `docs/`

### Gate Go/No-Go (fin phase 1)

Go si:
- baseline stable
- protocole benchmark fiable

No-Go si:
- mesures non reproductibles (bruit trop eleve)
- environnement local non stable

---

## Phase 2 - Prototype MoE minimal (4 semaines)

But: tester l'hypothese MoE avec risque controle, sans full pretraining.

### Strategie technique recommandee

- Eviter "retrain from scratch" au debut.
- Faire un proto petit:
  - 2 a 8 experts
  - top-k routing simple
  - petite taille de modele
- Prioriser adaptation legere:
  - SFT/finetune court
  - eventuellement LoRA

### Integrations minimales

- Branch experimentale dediee (isolee de `main`)
- Wrapper d'inference compatible interface MLXServe
- Feature flags:
  - backend standard
  - backend proto MoE

### Mesures obligatoires

- Qualite:
  - score relatif sur set de prompts fixe
  - evaluation humaine simple (A/B aveugle)
- Performance:
  - tokens/s
  - latence p95
  - time-to-first-token
- Memoire:
  - RAM pic
  - swap
  - nombre de refus memoire

### Gate Go/No-Go (fin phase 2)

Go si tous les points sont vrais:
- qualite >= baseline +3% (ou meilleure preference humaine nette)
- p95 latence <= +15% de regression max
- RAM pic <= +20%
- pas d'instabilite majeure (crash, OOM, drift)

No-Go si:
- gain qualite marginal mais cout perf/memoire important
- exploitation locale degradee

---

## Phase 3 - Industrialisation (4-8 semaines, seulement si Go)

### Chantiers

- Optimisations runtime MLX (hot paths)
- Caching et routing plus robustes
- Observabilite dediee MoE:
  - distribution d'experts actives
  - load balance
  - tokens/s par expert
- Durcissement API et fallback automatique vers backend dense en cas d'anomalie

### Securite/ops

- garde-fous memoire renforces
- runbooks incidents MoE
- tests de charge prolonges

### Gate final

Go production si:
- SLO respectes sur 7 jours de tests (latence, taux erreur, memoire)
- gain qualite/perf defendable
- exploitation simple pour un setup Apple Silicon local

---

## Budget et realisme

### Budget temps (ordre de grandeur)

- POC utile: 6 a 10 semaines
- Version "prod-ready": 3 mois+

### Budget compute (ordre de grandeur)

- Proto petit: faisable en budget limite
- Entrainement MoE ambitieux: couteux (rapidement hors budget solo local)

Conclusion pratique:
- le pari gagnant est un proto MoE petit + benchmark strict
- pas un full re-entrainement "mythos-scale"

---

## Risques majeurs et mitigations

- Router collapse (experts inutilises)
  - ajouter regularisation/load balancing
- Gains non transferables vers MLX runtime
  - mesurer tot sur hardware cible (Apple Silicon)
- Complexite architecture trop elevee
  - feature flags + fallback dense + scope strict

---

## Plan d'execution immediat (7 jours)

1. Geler set de prompts benchmark (`bench/prompts.json`)
2. Ecrire script benchmark unifie (`scripts/bench_chat.py`)
3. Capturer baseline actuelle MLXServe
4. Definir criteres de gate dans un fichier machine-readable (`bench/gates.yaml`)
5. Ouvrir branche experimentale `exp/moe-proto`

Si tu veux, prochaine etape: je peux te generer directement les fichiers `bench/prompts.json`, `bench/gates.yaml` et `scripts/bench_chat.py` pour lancer la phase 1 aujourd'hui.

