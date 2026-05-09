# Plan R&D - Dense -> MoE -> MLX (Cost-First)

**Mise à jour trajectoire** : la **mesure et la livraison d’inférence** se font en **MLX 4-bit** (bench + MLXServe), avec un objectif **≤ 16 GiB** de pic mémoire côté MLX pour le scénario produit — voir `docs/RD_PIPELINE_MLX_INFERENCE_16GB.md`. PyTorch, s’il est encore utilisé, sert au plus à **préparer** des poids avant export MLX, pas comme runtime de prod.

Objectif: convertir un modele dense open-source en architecture MoE exploitable dans l'ecosysteme MLXServe, avec un axe prioritaire explicite: **reduire au maximum le cout d'inference local** a qualite egale ou meilleure.

Ce plan assume une trajectoire R&D (plus risquee, plus ambitieuse) et non une simple adoption d'un MoE deja pret. Il est aligne sur une logique "efficacite de calcul d'abord" (parametres actifs minimaux, memoire maitrisee, latence stable).

---

## 0) Hypothese de travail (cost-first)

- Un modele dense de base peut gagner en qualite/cout inferentiel apres conversion MoE + re-entraînement cible.
- Les gains doivent rester mesurables sur hardware Apple Silicon (meme si l'entrainement initial est hors MLX).
- La metrique centrale n'est pas le nombre total de parametres, mais le **cout actif par token**.

Success criteria globaux:
- qualite >= baseline dense sur eval cible
- baisse du cout inferentiel (latence/tps/memoire/energie) dans limites d'exploitation MLXServe
- pas de collapse router, pas d'instabilite d'entrainement majeure

---

## 0.b) North-star metrics (obligatoires)

- `active_params_per_token` (ou proxy robuste si indisponible)
- `latency_p95_ms`
- `tokens_per_second`
- `ram_peak_gb` et `swap_used_gb`
- `energy_per_1k_tokens` (si mesurable; sinon proxy CPU/GPU time)
- qualite relative (score benchmark + preference humaine A/B)

Principe de decision:
- **No-Go** si la qualite monte mais que le cout inferentiel local explose.
- **Go** prioritaire si cout baisse a qualite stable.

---

## 1) Scope de depart (Sprint 1, 1-2 semaines)

### Decision de base model (dense)

Choisir un modele dense "petit/moyen" pour iterer vite:
- famille 7B/8B instruct
- tokenizer stable et conversion outillage mature
- licence claire pour fine-tuning
- profil d'inference local compatible (memoire, kv-cache, quantization)

### Livrables

- `docs/RD_DECISIONS.md` (choix modele, licence, contraintes)
- baseline dense capturee avec `bench/` (qualite + perf)
- protocole eval fixe (dataset + prompts + scoring)
- budget compute explicite par run (temps/energie/cout)

### Go/No-Go

Go si baseline reproduisible (variance acceptable), budget training defini et metriques cost-first capturees.

---

## 2) Conversion architecturale Dense -> MoE (Sprint 2, 2-3 semaines)

### Strategie technique

Transformer FFN dense en MoE:
- remplacer blocs MLP/FFN par:
  - `N` experts routables
  - `K` experts actifs par token (top-k)
  - experts partages optionnels (toujours actifs)
- router learned (softmax logits + top-k)
- conversion **partielle** d'abord (certaines couches) pour limiter la dette compute

### Warm-start (critique)

Eviter training from scratch:
- cloner/projeter les poids FFN denses dans les experts
- initialiser router de maniere quasi-uniforme
- conserver un chemin shared proche du FFN original
- imposer un budget d'activation (`K`, capacity factor) des cette etape

### Livrables

- module conversion: `tools/convert_dense_to_moe.py`
- nouvelle archi entrainable (`training/moe_model.py`)
- tests unitaires conversion:
  - shapes
  - equivalence grossiere de sortie avant fine-tuning

### Go/No-Go

Go si:
- conversion deterministe
- forward/backward stables
- perte initiale raisonnable (pas d'explosion immediate)
- cout actif/token non explosif vs dense baseline

---

## 3) Stabilisation training MoE (Sprint 3, 2-3 semaines)

### Problemes attendus

- router collapse (2-3 experts monopolises)
- instabilite gradient
- forte variance de loss

### Contremesures

- load-balancing loss (auxiliary)
- capacity factor explicite
- z-loss / regularisation router (si utile)
- gradient clipping + LR conservative warmup
- penalite compute-aware dans la loss router (garder un top-k minimal utile)

### Livrables

- script training phase stabilisation: `training/moe_stabilize.py`
- dashboards minimaux:
  - utilisation experts
  - entropy router
  - overflow/capacity drops

### Go/No-Go

Go si:
- utilisation experts non-collapsee
- loss stable sur fenetre longue
- pas de NaN/OOM recurrent
- budget inference proxy respecte (active params/token ou equivalent)

---

## 4) Re-entraînement cible (Sprint 4, 2-4 semaines)

### Regime recommande

- etape A: SFT cible domaine (qualite immediate)
- etape B (option): continued pretraining court sur corpus specialise

### Livrables

- checkpoints MoE v1
- eval comparative dense vs MoE:
  - exact match / win-rate / note humaine
  - latence, tps, memoire, cout actif/token

### Go/No-Go

Go si:
- gain qualite net ou meme qualite avec meilleur cout inferentiel
- regression latence/memoire tolerable selon gate produit
- trajectoire "moins de compute pour meme puissance" verifiee

---

## 5) Integration runtime MLXServe (Sprint 5, 2 semaines)

### Cible

- brancher un backend "moe_real" dans la meme interface runtime que les autres backends
- conserver fallback dense/stub en securite

### Livrables

- backend runtime experimental `moe_real` (feature-flag)
- A/B bench automate (`scripts/bench_ab.py` etendu)
- doc exploitation + rollback rapide
- metriques runtime MoE:
  - repartition experts actifs
  - active params/token (ou proxy)
  - cout par requete

### Go/No-Go

Go si:
- passage tests regressions API
- SLO minimaux atteints sur scenario charge
- cout inferentiel local mieux ou egal a qualite comparable

---

## 6) Gates quantifies proposes (cost-first)

- Qualite:
  - +3% min sur score cible ou win-rate humain > 55%
- Latence:
  - p95 <= +20% regression max (ou mieux)
- Memoire:
  - RAM pic <= +25%
  - 0 freeze / 0 swap runaway
- Fiabilite:
  - error rate API <= baseline + 1%
- Efficacite compute:
  - active_params_per_token <= baseline equivalente dense (proxy accepte)
  - energy_per_1k_tokens <= baseline + 10% max (ou baisse nette)

---

## 7) Risques majeurs et plan B

- **Collapse router persistant**
  - Plan B: augmenter shared experts, baisser top-k, renforcer balancing loss
- **Gains non transferables en inference MLX**
  - Plan B: maintenir backend alternatif non-MLX a court terme
- **Cout training trop eleve**
  - Plan B: reduire scale (moins experts, moins couches converties), objectif qualite plus cible
- **Cout inferentiel ne baisse pas**
  - Plan B: MoE partiel plus agressif (moins de couches converties), top-k plus bas, shared experts legers

---

## 8) Ordonnancement concret (10 semaines)

- S1-S2: baseline + choix modele + protocole eval
- S3-S4: conversion Dense -> MoE + tests conversion
- S5-S6: stabilisation router/training
- S7-S8: re-entrainement cible + eval A/B
- S9-S10: integration runtime MLXServe + hardening

Milestone additionnel obligatoire:
- fin S4: revue "compute economics" (go/no-go avant d'investir S5+)

---

## 9) Definition of done finale

Le projet est "done" quand:
- un checkpoint MoE converti depuis dense est entraine et documente
- la comparaison dense vs MoE est reproducible
- l'integration MLXServe est activable via flag et rollbackable
- les gates qualite/perf/memoire/compute sont respectes

