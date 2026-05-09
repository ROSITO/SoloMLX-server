# R&D Sprints Status - Dense -> MoE -> MLX

## Sprint 1 - Cadrage cost-first (en cours)

### Done
- [x] Plan cost-first (`docs/RD_DENSE_TO_MOE_PLAN.md`)
- [x] Decisions initiales (`docs/RD_DECISIONS.md`)
- [x] Bench baseline et protocole (`bench/`, `scripts/bench_chat.py`)

### In progress
- [x] Selection checkpoint dense bootstrap (SmolLM2-135M)
- [x] Instrumentation `active_params_per_token` (proxy v1)
- [x] Pack eval qualite v1 (A/B humain + scoring)
- [x] Candidat scale-up 7B verrouille (Mistral-7B-Instruct-v0.3)

### Gate Sprint 1
- baseline reproductible: **OK**
- budget experimentation defini: **OK (bootstrap)**
- modele source verrouille: **OK (bootstrap + scale-up)**

---

## Sprint 2 - Conversion Dense -> MoE bootstrap (demarre)

### Done
- [x] Outil conversion initial: `tools/convert_dense_to_moe.py`
- [x] Tests conversion bootstrap: `tests/test_dense_to_moe_conversion.py`

### In progress
- [x] Mapping couches reel selon modele source (SmolLM/Llama-like)
- [x] Conversion partielle FFN sur un checkpoint reel
- [x] Forward/backward smoke sur modele converti (training stack reel)

### Gate Sprint 2 (cible)
- conversion deterministe: **OK**
- forward/backward stables sur vrai modele: **OK (bridge smoke)**
- cout actif/token non explosif: **PARTIEL** (proxy calcule, mesure runtime fine restante)

---

## Sprint 3 - Stabilisation training MoE (demarre)

### Done
- [x] Bloc MoE entrainable `TopKMoEFFN` (torch)
- [x] Script stabilisation `training/moe_stabilize.py`
- [x] Test smoke `tests/test_moe_stabilize.py`
- [x] Runbook `docs/RD_SPRINT3_STABILIZATION.md`

### In progress
- [x] Bridger le bloc stabilise aux couches converties reelles
- [x] Capturer un premier rapport `stabilize_report.json` long run (>=200 steps)
- [x] Capturer un rapport bridge checkpoint reel (`bridge_smoke_report.json`)

### Gate Sprint 3 (cible)
- loss stable sans NaN/OOM: **OK (smoke)**
- expert usage non-collapsee: **OK (smoke)**
- budget compute proxy acceptable: **PARTIEL** (proxy conversion OK, profilage runtime complet restant)

---

## Sprint 4 - Eval A/B dense vs MoE bridge (demarre)

### Done
- [x] Script eval A/B `training/moe_eval_ab.py`
- [x] Runbook `docs/RD_SPRINT4_EVAL.md`

### In progress
- [x] Executer un rapport complet `eval_ab_report.json`
- [x] Interpréter gate qualite/cout avec set v1
- [x] Itérer tuning bridge MoE (objectif revenir au niveau dense)
- [ ] Lancer entrainement cible reelle pour recuperation qualite (SFT/CPT court)
- [x] Optimiser latence runtime MoE (dispatch sparse v1)
- [x] Optimiser latence runtime MoE (fast path top_k=1)

### Gate Sprint 4 (cible)
- comparaison dense vs moe_bridge reproducible: **OK**
- decision go/no-go pour entrainement cible: **GO** (qualite parity atteinte, latence amelioree sur config partielle)

---

## Sprint 5 - Entrainement cible court (demarre)

### Done
- [x] Script training adapter MoE `training/moe_target_train.py`
- [x] Support chargement adapter dans eval `training/moe_eval_ab.py`
- [x] Runbook `docs/RD_SPRINT5_TARGET_TRAINING.md`
- [x] Expert ticket: `--router-entropy-weight`, `--shrink-to-experts`, `shrink_moe_state_dict`, eval `--adapter-path-shrunk`
- [x] Scripts `scripts/run_moe_expert_ticket_smoke.sh`, `scripts/run_moe_expert_ticket_mistral7b_mps.sh`

### In progress
- [x] Executer training adapter v1
- [x] Eval A/B avec adapter v1
- [x] Executer training adapter v2 sur GPU Apple (MPS)
- [x] Eval A/B adapter v2 sur GPU Apple (MPS)
- [x] Decision gate pre-scale-up (SmolLM -> 7B)
- [x] Eval tri Mistral-7B MPS via split (`scripts/run_moe_eval_ab_tri_split.sh` -> `moe_expert_ticket_mistral7b_mps_eval_ab.json`)
- [ ] Run Mistral-7B expert-ticket long (>=120 steps) + rapport train fige (machine locale)

### Gate Sprint 5 (cible)
- qualite >= dense (sur set v1): **OK** (v1 et v2)
- latence <= dense ou regression acceptable: **OK** (v2: p95 meilleur)
- adapter exportable/rechargeable: **OK**

Decision Sprint 5:
- **GO scale-up** vers `mistralai/Mistral-7B-Instruct-v0.3` (voir `bench/moe_training/scaleup_gate.json`)

---

## Sprint 6 - Scale-up 7B (demarre)

### Done
- [x] Plan de scale-up 7B (`docs/RD_SPRINT6_SCALEUP.md`)
- [x] Config de base retenue (`layers=10`, `top_k=1`, `shared_experts=0`)

### In progress
- [x] Support conversion checkpoints sharded (`tools/run_sprint2_conversion.py`)
- [x] Conversion partielle 7B dense->MoE (couche 10) terminee
- [x] Premier training adapter court sur 7B (MPS)
- [x] Eval A/B 7B dense vs MoE adapter (smoke)

Resultats smoke 7B:
- conversion: `bench/moe_conversion/mistral7b/conversion_report.json` (`converted_keys=3`, `layers=[10]`)
- training: `bench/moe_training/mistral7b_moe_target_train_smoke_report.json` (`loss_delta=-0.873`)
- eval: `bench/moe_training/mistral7b_eval_ab_report_adapter_smoke.json`
  - delta loss: `-0.291`
  - delta p95: `-17497.40 ms`
- memoire inference (run unique MPS): `max RSS=15.50 GB`, `peak footprint=21.60 GB`, `swaps=0`

Resultats run long 7B (adapter v1 long):
- training: `bench/moe_training/mistral7b_moe_target_train_report_v1_long.json` (`steps=60`, `loss_delta=-4.251`)
- eval: `bench/moe_training/mistral7b_eval_ab_report_adapter_v1_long.json`
  - delta loss: `-0.320`
  - delta p95: `+10638.51 ms` (regression latence vs dense)

Resultats run intermediaire 7B (adapter v2 balanced):
- training: `bench/moe_training/mistral7b_moe_target_train_report_v2_balanced.json` (`steps=20`, `lr=1e-5`, `loss_delta=-1.594`)
- eval: `bench/moe_training/mistral7b_eval_ab_report_adapter_v2_balanced.json`
  - delta loss: `-0.220`
  - delta p95: `-3978.60 ms` (amelioration latence vs dense)

### Gate Sprint 6 (cible)
- qualite >= dense (set v1): **OK (smoke)**
- latence p95 <= dense (ou delta acceptable): **OK (smoke)**
- memoire stable sans swap: **OK (smoke, 0 swap)**

---

## Sprint 7 - MoE natif MLX (mlx-lm) — prod cost-first

### Done
- [x] Cartographie: MoE **natif** dans mlx-lm (`mixtral.MixtralSparseMoeBlock`, `SwitchGLU`)
- [x] Doc trajectoire prod + liens efficacite (`docs/RD_MOE_MLX_SERVE.md`)
- [x] Bench reproductible RAM/temps/token/s (`training/mlx_moe_bench.py`)
- [x] Smoke bench dense 4-bit: `bench/moe_training/mlx_moe_bench_smoke_dense.json`

### In progress
- [ ] Bench **Mixtral 4-bit** vs **Mistral-7B 4-bit** (meme `max_tokens`, meme machine) pour courbe RAM/latence
- [ ] (R&D) Export poids MoE **custom** (torch) vers bundle mlx-lm compatible

### Gate Sprint 7
- MoE MLX quantifie servable via MLXServe **sans** stack PyTorch en inference: **OK** (deja le cas pour repos `mlx-community/*Mixtral*-4bit`)
- Metrique `peak_memory_gib` + `tokens_per_s` sur au moins une paire dense/MoE MLX: **A completer** (commandes dans `docs/RD_MOE_MLX_SERVE.md`)

---

## Sprint 8 - Scale-up MoE custom ~20B / ~30B

### Cible
- **~24B** : `mistralai/Mistral-Small-24B-Instruct-2501` (stack la plus proche du 7B deja valide)
- **~32B** : `Qwen/Qwen2.5-32B-Instruct` ou `Qwen/Qwen2.5-Coder-32B` (extension bridge FFN / Qwen2)

### Done
- [x] Plan operatoire + gates (`docs/RD_SCALEUP_20_30B.md`)

### In progress
- [ ] Adapter smoke **24B** PyTorch : **bloqué RAM** sur machine de ref (exit **137** après chargement poids) — pré-requis **~48+ GiB** unifiés bf16 ou poste cloud ; voir `docs/RD_SCALEUP_20_30B.md`
- [ ] Eval A/B **24B** (idem, poste large mémoire)
- [x] Gate chaîne **7B** : `moe_target_train` + `moe_eval_ab` avec `--dtype bf16 --low-cpu-mem-usage --offload-unused-experts` — rapports `bench/moe_training/mistral7b_moe_scaleup_gate_smoke.json`, `mistral7b_eval_ab_scaleup_gate_smoke.json`
- [x] Conversion safetensors **Dense→MoE** **24B** OK : `bench/moe_conversion/mistral_small_24b/conversion_report.json` (couche **20**, 3 clés FFN → MoE)
- [x] Conversion safetensors **32B** OK : `bench/moe_conversion/qwen25_32b/conversion_report.json` (couche **32**, 3 clés FFN → MoE ; script `scripts/run_moe_convert_qwen32b_only.sh`)
- [x] Paquet **mlx-community** 4-bit 24B pour `mlx_moe_bench` + gate 16 GiB (`bench/moe_training/mlx_moe_bench_mistral_small_24b_4bit_16g_gate.json`)
- [x] Premiers benches **32B** MLX : 4-bit Instruct **hors** 16 GiB ; **3-bit** Coder **dans** 16 GiB (`mlx_moe_bench_qwen25_32b_*`, `*_coder_32b_instruct_3bit_*`)
- [x] Spec **conversion seule** Qwen2 : clés `model.layers.*.mlp.{gate,up,down}_proj` (meme pattern que Llama pour `convert_dense_to_moe`)

### Gate Sprint 8
- Chaine 24B **sans OOM destructif** sur machine de reference: **OK (inférence MLX 4-bit micro-bench, pic ~12.4 GiB)** + **bootstrap safetensors MoE couche 20**
- Qualite MoE vs dense (set v1 ou set code leger): **A valider**
- Trajectoire **32B** documentee + **bootstrap safetensors MoE couche 32** : **OK**

---

## Sprint 9 - MLX 4-bit inference-only + budget **16 GiB**

### Principe
- Pas de **PyTorch** sur le chemin d’**inférence** ; MoE → **MLX 4-bit** → `mlx_moe_bench` (+ gate RAM).
- Voir `docs/RD_PIPELINE_MLX_INFERENCE_16GB.md`.

### Done
- [x] Doc pipeline + interprétation budget RAM
- [x] `training/mlx_moe_bench.py` : `--ram-budget-gib`, `peak_memory_gib_worst`, `within_ram_budget`, exit **2** si dépassement

### In progress
- [x] Export MoE **custom** vers bundle **mlx-lm** chargeable + **q_bits=4** sur **7B full-MoE** (`tools/convert_dense_to_mixtral_moe_bundle.py` -> `mlx_lm.load` OK -> `mlx_lm convert --q-bits 4` OK ; bench `mlx_moe_bench_mistral7b_full_moe_e2_mlx4bit_16g.json`)
- [x] Bench comparatif **dense 4-bit vs MoE 4-bit** (classe ~7B) avec `--ram-budget-gib 16` : dense (`mlx_moe_bench_mistral7b_dense_mlx4bit_16g_cmp.json`) vs custom MoE (`mlx_moe_bench_mistral7b_full_moe_e2_mlx4bit_16g.json`)
- [x] Benches **24B / 32B** dense MLX + gate 16 GiB (voir `docs/RD_PIPELINE_MLX_INFERENCE_16GB.md`) : **24B 4-bit OK** ; **32B 4-bit KO** ; **32B 3-bit OK**
- [x] Extension **14B** custom full-MoE (`qwen2_moe`, 2 experts) : export MLX OK ; **4-bit KO RAM** (~18.5 GiB) ; **3-bit OK RAM** (~14.4 GiB) ; baseline dense 14B (4-bit/3-bit) mesurée
- [x] Feature clé **délestage experts MLX runtime** (`src/mlxserve/runtime/moe_offload.py`, settings `MLXSERVE_MOE_RESIDENT_EXPERTS/STRATEGY`) testée sur custom MoE 7B et 14B (qualité: OK en 7B avec keep=1, dégradation en 14B avec keep=1; recommandation 14B keep=2)


### Gate Sprint 9
- `within_ram_budget=true` sur protocole fixe (prompt + `max_tokens`): **PARTIEL** — OK pour **24B 4-bit** et **32B 3-bit (Coder)** ; **non** pour **32B 4-bit Instruct** sur la machine de référence
- `tokens_per_s` + qualité vs baseline dense 4-bit: **A valider**
