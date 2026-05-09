# Roadmap optimisation inférence (MLXServe)

Référence croisée : `LLM_Inference_Optimization_Research.md` (sources externes) et l’état du dépôt. Priorité produit : stabilité mémoire Apple Silicon, puis qualité / throughput.

---

## Déjà en place (référence code)

| Domaine | Emplacement / artefacts |
|--------|-------------------------|
| Runtime MLX, streaming, KV quant | `src/mlxserve/runtime/backends.py` (`MLXLMBackend`, `kv_bits`, `quantized_kv_start`), `engine.py`, `config.py` (`MLXSERVE_KV_*`) |
| Garde-fous RAM, admission, unload | `MemoryGuardian`, `MEMORY.md`, settings `max_memory_gb` / `hard_memory_gb` / idle unload |
| MoE MLX (résidence au load, fastpath) | `src/mlxserve/runtime/moe_offload.py`, env `MLXSERVE_MOE_*` |
| Bench MoE / gate 16 Go | `training/mlx_moe_bench.py`, `docs/RD_PIPELINE_MLX_INFERENCE_16GB.md` |
| MoE PyTorch : train, entropie routeur, shrink, eval tri (split MPS) | `training/moe_target_train.py`, `training/moe_model.py`, `training/moe_eval_ab.py`, `training/merge_moe_eval_ab_tri.py`, `scripts/run_moe_eval_ab_tri_split.sh`, `scripts/run_moe_expert_ticket_*.sh` |
| Autotune KV / prefill | `src/mlxserve/runtime/autotune.py` |

---

## Phase P0 — Mesurer et cadrer

- Grille reproductible : modèle × bits × `moe_resident_experts` × `kv_bits` / `quantized_kv_start` × longueur de contexte (prolonger l’usage de `mlx_moe_bench` + rapports JSON versionnés).
- Mémoire : pression macOS + estimation KV grossière par architecture (voir limites dans `MEMORY.md`).
- Garder le eval **tri-split** comme référence pour gros modèles MPS (évite OOM entre deux chargements 7B+).

---

## Phase P1 — Mémoire unifiée et KV

- Exploiter systématiquement les variables `MLXSERVE_KV_BITS`, `KV_GROUP_SIZE`, `QUANTIZED_KV_START` avec validation qualité sur prompts fixes.
- Affiner admission / hysteresis / métriques de rejet (aligné `MEMORY.md` « prochaines actions »).
- Suivre les capacités **mlx_lm** pour toute compression KV plus agressive (KIVI / TurboQuant dans le doc de recherche = cibles de veille, pas d’engagement tant que le runtime ne l’expose pas).

---

## Phase P2 — MoE « utile » sur Apple Silicon

- Poursuivre **expert ticket** : entraînement / shrink mesurés vs qualité (`moe_target_train`, `shrink_moe_state_dict`, eval fusionné).
- Réduire l’écart **poids exportés MLX** : automatiser ou documenter la chaîne checkpoint → bundle `mlx_lm` (Mixtral / Qwen2 full-MoE : `tools/convert_dense_to_*_moe_bundle.py`).
- Documenter les limites de `moe_offload` : résidence **statique** au load (pas prefetch disque entre tokens). Explorer de petits ajustements compatibles MLX sans dupliquer NPUMoE / MoE-SpeQ tant que non intégrables.

---

## Phase P3 — Latence perçue et throughput (long terme)

- **Speculative decoding** : principal trou vs le doc de recherche ; à traiter après P1–P2 stables (dépend fortement de mlx_lm / modèles draft).
- Multi-client / batching : seulement si le produit sort du scénario « poste local » ; sinon rester simple (`AGENTS.md`).

---

## Veille (hors chemin critique court terme)

- Surveys : arXiv `2506.21901`, Efficient LLMs survey, low-bit survey `2409.16694`.
- MoE Apple / papers : NPUMoE, MoE-SpeQ, awesome-moe-inference (liens dans `LLM_Inference_Optimization_Research.md`).
- Quant poids : GPTQ, AWQ, AQLM — en pratique MLXServe s’appuie sur **checkpoints déjà quantifiés** ; pipeline maison seulement si besoin produit explicite.

---

## Ordre de lecture

1. `MEMORY.md` — contraintes 16 Go et backlog mémoire.  
2. `docs/RD_PIPELINE_MLX_INFERENCE_16GB.md` — pipeline MLX cible.  
3. `LLM_Inference_Optimization_Research.md` — bibliographie et pile « idéale ».  
4. Ce fichier — traduction en phases livrables dans MLXServe.

---

## Implémenté (code + ops)

- [x] **P0** Pré-admission avec **estimation KV** optionnelle (`mlxserve/memory/estimate.py`, variables `MLXSERVE_MEMORY_ADMISSION_KV_*`, `README.md`).
- [x] **P0** Grille bench MLX : `scripts/run_mlx_inference_grid.sh` + `training/mlx_moe_bench.py` (`--kv-bits`, `--prefill-step-size`, …).
- [x] **P1** Métriques **refus chat par raison** : `classify_detail` + `mlxserve_memory_chat_denied_by_reason_total` (`observability.py`, `app.py`).
- [x] **P2** Doc limites **MoE offload** : `docs/MOE_OFFLOAD.md`.
- [ ] **P3** Speculative decoding — volontairement **non implémenté** (dépendance `mlx_lm` / modèles draft ; voir `LLM_Inference_Optimization_Research.md`).

### Gate 16 GiB — 32B (validation machine)

- Modèle : `mlx-community/Qwen2.5-Coder-32B-Instruct-3bit`
- Commande : `python -m training.mlx_moe_bench --model-id … --max-tokens 24 --ram-budget-gib 16 --kv-bits 4 --quantized-kv-start 32 --prefill-step-size 512`
- Résultat : `within_ram_budget=true`, pic ~**13.5 GiB** — rapport `bench/moe_training/mlx_bench_qwen25_coder_32b_3bit_16g_gate.json`

### Gate 16 GiB — 24B Mistral instruct (cible généraliste)

- Modèle : `mlx-community/Mistral-Small-24B-Instruct-2501-4bit`
- Commande : `python -m training.mlx_moe_bench --model-id mlx-community/Mistral-Small-24B-Instruct-2501-4bit --max-tokens 48 --ram-budget-gib 16 --kv-bits 4 --quantized-kv-start 32 --prefill-step-size 512`
- Résultat : `within_ram_budget=true`, pic ~**12.4 GiB** — rapport `bench/moe_training/mlx_bench_mistral_small_24b_4bit_16g_gate.json`
