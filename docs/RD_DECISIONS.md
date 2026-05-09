# RD Decisions - Sprint 1 (Cost-First)

Ce document verrouille les decisions de depart pour le plan `docs/RD_DENSE_TO_MOE_PLAN.md`.

## 1) Objectif principal

- **Priorite**: reduire le cout d'inference local (Apple Silicon) a qualite equivalente.
- Objectif cible initial:
  - `active_params_per_token`: -30% vs baseline dense (proxy accepte)
  - `latency_p95_ms`: pas de regression > +15%
  - `ram_peak_gb`: pas de regression > +20%

## 2) Base model dense (decision)

- Famille cible: **7B/8B instruct dense** (iterable local/cloud)
- Criteres obligatoires:
  - licence permissive pour fine-tuning
  - tokenizer stable
  - conversion outillage mature
  - disponibilite de checkpoint standard Hugging Face

Decision bootstrap Sprint 1/2:
- modele de travail: `HuggingFaceTB/SmolLM2-135M-Instruct`
- licence: Apache-2.0
- rationale: single-file safetensors, rapide pour iterer, outillage stable

Etat actuel:
- [x] modele exact selectionne (bootstrap)
- [x] licence validee
- [x] hash/checkpoint reference note (voir `bench/moe_conversion/.../conversion_report.json`)
- [x] candidat principal scale-up (7B/8B) verrouille: `mistralai/Mistral-7B-Instruct-v0.3` (Apache-2.0)

## 3) Design conversion Dense -> MoE (premiere passe)

- Scope initial: **conversion partielle FFN** (pas toutes les couches)
- Parametres init:
  - `num_experts`: 4
  - `top_k`: 1 (objectif cost-first)
  - `shared_experts`: 1
- Warm-start:
  - duplication/projection poids FFN dense vers experts
  - init router quasi uniforme

Etat actuel:
- [x] spec de conversion ecrite
- [x] outil bootstrap cree (`tools/convert_dense_to_moe.py`)
- [x] mapping exact des noms de couches du modele choisi (famille SmolLM/Llama-like)
- [x] conversion partielle executee sur checkpoint reel (layers 2,6,10)
- [x] config partielle recommandee v1 validee: `layers=10`, `top_k=1`, `shared_experts=0`

## 4) Protocoles benchmark et gates

- Inputs: `bench/prompts.json`
- Gates: `bench/gates.yaml`
- Runner:
  - benchmark simple: `scripts/bench_chat.py`
  - A/B: `scripts/bench_ab.py`

Metriques obligatoires sprint 1:
- `latency_p95_ms`
- `tokens_per_second`
- `ram_peak_gb` (ou proxy)
- `memory_denials`

Etat actuel:
- [x] benchmark phase 1 en place
- [x] baseline datee capturee
- [x] instrumentation proxy `active_params_per_token` ajoutee (`tools/estimate_active_params_proxy.py`)
- [ ] instrumentation energie/token ajoutee (ou proxy)
- [x] pack eval qualite v1 cree (`bench/eval_quality_v1.json`)

## 5) Budget et limites sprint 1

- Timebox: 1-2 semaines
- Compute: experiments courts, pas de full training long
- No-go immediate:
  - instability systeme locale
  - absence de baseline reproductible

## 6) Deliverables sprint 1

- [x] Plan R&D cost-first mis a jour
- [x] Decisions initiales documentees (ce fichier)
- [x] Bootstrap conversion Dense->MoE (outil + tests)
- [x] modele dense exact choisi (bootstrap)
- [x] package eval qualite v1 (set initial)
- [ ] candidat dense 7B/8B final pour phase de scale-up

---

Prochaine revision: fin sprint 1 (go/no-go sprint 2 conversion reelle).
