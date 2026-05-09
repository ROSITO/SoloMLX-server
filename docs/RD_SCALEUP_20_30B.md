# Scale-up MoE custom — bande **~20B** et **~30B**

> **Priorité actuelle** : enchaîner **scale-up 24B / 32B** sur le chemin **MLX quantifié** (idéalement **4-bit**), mesures avec `mlx_moe_bench` et gate **`--ram-budget-gib 16`** — voir **`docs/RD_PIPELINE_MLX_INFERENCE_16GB.md`** pour les **résultats mesurés** (24B 4-bit OK ; 32B 4-bit au‑delà de 16 GiB sur protocole court ; **32B ~13.5 GiB** avec variante **3-bit** Coder).

Objectif historique: répliquer le playbook **7B** sur des modèles **proches de 20B ou 30B**, en restant **cost-first** (métriques: FLOPs/token proxy, `peak_memory`, tokens/s, qualité).

## Candidats recommandés

| Bande | Rôle | ID Hugging Face (dense, R&D torch) | Remarque |
|-------|------|--------------------------------------|----------|
| ~24B | Proche 20B, Apache-2.0, Llama-like | `mistralai/Mistral-Small-24B-Instruct-2501` | 40 couches, contexte 32k ; même famille d’outils que Mistral-7B |
| ~32B | Cible ~30B, usage général + code | `Qwen/Qwen2.5-32B-Instruct` | Vérifier **mapping FFN** (Qwen2 ≠ Llama strict) avant conversion |
| ~32B | Code | `Qwen/Qwen2.5-Coder-32B` | Idem ; priorité si la cible produit est le codegen |

**MLX 4-bit (prod / bench)** — à valider sur le hub `mlx-community` au moment du run (noms peuvent évoluer) :

- Ex. `mlx-community/Mistral-Small-24B-Instruct-2501-4bit` ou équivalent publié.
- Ex. `mlx-community/Qwen2.5-32B-Instruct-4bit` ou `...-Coder-32B-4bit` si disponible.

## Délestage des experts non utilisés (bridge PyTorch / R&D)

**Constat** : avec `top_k=1`, le forward ne **calcule** que les experts sélectionnés, mais **tous** les experts restent **résidents** en mémoire vive tant que leurs paramètres sont sur le device d’inférence — ce qui ne réduit pas l’empreinte **pic RAM**.

**Technique implémentée** (`training/moe_model.py`, `TopKMoELlamaMLP`) : **`offload_unused_experts=True`** (ou flags CLI ci‑dessous).

- Début de forward : seuls les experts **sélectionnés** sur ce pas sont montés sur le **device** (MPS/CUDA) ; les autres sont poussés en **`cpu`**.
- Fin de forward : en mode **`eval()`** uniquement, **tous** les experts sont re‑parkés sur **CPU** pour libérer le device avant la couche suivante. En **`train()`**, on **ne** fait **pas** ce release final (sinon **autograd** cassé).

**Après conversion safetensors** (`*.moe-bootstrap.safetensors`) : les poids MoE bootstrap ne sont **pas** chargés tels quels par `transformers` ; le délestage s’applique au **bridge PyTorch** (checkpoint HF dense + `replace_llama_mlp_with_moe` + adaptateur entraîné). Commandes typiques :

```bash
# Profilage MPS avec délestage
.venv/bin/python -m training.profile_mps_inference \
  --model-id mistralai/Mistral-7B-Instruct-v0.3 \
  --layers 10 --offload-unused-experts \
  --adapter-path bench/moe_training/ton_adapter.pt

# Eval A/B avec délestage sur le chemin MoE
.venv/bin/python -m training.moe_eval_ab \
  --model-id mistralai/Mistral-7B-Instruct-v0.3 \
  --layers 10 --offload-unused-experts \
  --adapter-path bench/moe_training/ton_adapter.pt \
  --device mps

# Entraînement adaptateur : flag supporté (experts inactifs restent sur CPU pendant le forward)
.venv/bin/python -m training.moe_target_train \
  --model-id mistralai/Mistral-Small-24B-Instruct-2501 \
  --layers 20 --offload-unused-experts --device mps ...
```

**Coût** : transferts device↔CPU ; sur **MPS**, peut coûter cher en latence — comparer avec / sans via `profile_mps_inference`.

**MLX / mlx-lm** : MoE **natif** charge en général **tous** les experts ; un délestage **inférence** propre = **couche MoE MLX dédiée** (cache LRU / shards lazy) — hors `mlx_lm.load()` vanilla.

## Mesures MLX gate **16 GiB** (dense catalog, **mai 2026**)

Reproductibles via `training/mlx_moe_bench.py` ; détail et table dans **`docs/RD_PIPELINE_MLX_INFERENCE_16GB.md`**. Synthèse :

- **24B** (`mlx-community/Mistral-Small-24B-Instruct-2501-4bit`) : pic **~12.4 GiB** — **rentre** dans 16 GiB (prompt court, `max_tokens=16`).
- **32B Instruct** (`.../Qwen2.5-32B-Instruct-4bit`) : pic **~17.3 GiB** — **ne rentre pas** dans 16 GiB au même protocole.
- **32B Coder** (`.../Qwen2.5-Coder-32B-Instruct-3bit`) : pic **~13.5 GiB** — **rentre** dans 16 GiB (quant **3-bit**, pas 4-bit).

**MoE** : ces runs sont des modèles **denses** quantifiés sur le hub. Dès que l’**export MoE custom → mlx-lm** fonctionne pour **24B / 32B**, rejouer les mêmes commandes en remplaçant `--model-id` par le bundle MLX obtenu (attendre un pic **≥ dense** si tous les experts sont résidents, sauf architecture type offload / partage non encore cadrée).

## Contraintes matérielles (macOS, mémoire unifiée)

Ordres de grandeur indicatifs :

- **24B / 32B en bf16/fp16** sur PyTorch+MPS : souvent **au-delà** d’un Mac **36–64 GiB** confortable pour training + généralisation ; attendre **OOM**, **swap**, ou itérations **très lentes**.
- **Stratégie réaliste** :
  1. **Conversion safetensors** (I/O disque) : faisable sur SSD externe si espace suffisant (~50–100+ GiB selon shards).
  2. **Training adapter** : commencer **`max_length` bas** (64–128), **`steps` courts**, **1 couche** MoE ; surveiller Activity Monitor ; sinon **CPU** (lent mais stable) ou machine avec **≥ 64 GiB**.
  3. **Inférence cible** : **MLX 4-bit** (objectif RAM maîtrisée), pas le dense flottant.

## Compatibilité code actuel

- `tools/convert_dense_to_moe.py` : suffixes FFN type **Llama/Mistral** (`mlp.gate_proj`, `up_proj`, `down_proj`). **Mistral-Small-24B** est le candidat le plus **direct** après 7B.
- `training/moe_model.replace_llama_mlp_with_moe` : attend un `layer.mlp` avec **`gate_proj` / `up_proj` / `down_proj`** (Llama, **Qwen2**, Mistral). La **conversion safetensors** Qwen2.5 utilise les mêmes clés ; le bridge torch doit être validé par un **smoke forward** sur `Qwen2ForCausalLM` une fois le 32B chargé (RAM élevée).

## Protocole (même squelette que le 7B)

### 1) Conversion dense → MoE (1 couche pour commencer)

**Fichiers safetensors** : si le dépôt HF expose à la fois `consolidated.safetensors` et des shards `model-*-of-*`, `tools.convert_dense_to_moe` **n’utilise que les shards** (évite de charger ~2× le modèle entier en RAM). Supprimer localement un `consolidated.safetensors` téléchargé par erreur libère beaucoup d’espace disque.

Choisir `layers` conservateur (ex. couche **milieu** : `20` sur 40 pour 24B — à ajuster après inspection `config.num_hidden_layers`).

```bash
.venv/bin/python -m tools.run_sprint2_conversion \
  --repo-id mistralai/Mistral-Small-24B-Instruct-2501 \
  --workdir bench/moe_conversion/mistral_small_24b \
  --layers 20 \
  --num-experts 4 \
  --top-k 1 \
  --shared-experts 0
```

### 2) Training adapter (MPS si possible)

**RAM** : un **24B bf16** complet en PyTorch tient typiquement **~45–50+ GiB** de mémoire unifiée (poids seuls) avant MoE ; sur une machine plus petite le processus est **tué (exit 137)**. En pratique : valider la chaîne **`moe_target_train` → `moe_eval_ab`** sur **Mistral-7B** (mêmes flags), puis reproduire sur **24B** uniquement sur poste **≥ 64 GiB** ou avec stratégie quant / offload non encore intégrée ici.

Flags utiles (7B smoke reproductible) :

```bash
.venv/bin/python -m training.moe_target_train \
  --model-id mistralai/Mistral-7B-Instruct-v0.3 --layers 10 \
  --num-experts 4 --top-k 1 --shared-experts 0 \
  --steps 4 --max-length 96 --lr 1e-5 \
  --device mps --dtype bf16 --low-cpu-mem-usage --offload-unused-experts \
  --out-adapter bench/moe_training/mistral7b_moe_scaleup_gate_smoke.pt \
  --out-report bench/moe_training/mistral7b_moe_scaleup_gate_smoke.json

.venv/bin/python -m training.moe_eval_ab \
  --model-id mistralai/Mistral-7B-Instruct-v0.3 --layers 10 \
  --num-experts 4 --top-k 1 --shared-experts 0 \
  --warmup-steps 3 --max-new-tokens 24 \
  --device mps --dtype bf16 --low-cpu-mem-usage --offload-unused-experts \
  --adapter-path bench/moe_training/mistral7b_moe_scaleup_gate_smoke.pt \
  --out bench/moe_training/mistral7b_eval_ab_scaleup_gate_smoke.json
```

`moe_eval_ab` charge **dense puis MoE séquentiellement** (une copie à la fois sur le device) et le **warmup** n’optimise **que** les paramètres **MoE** (plus d’`AdamW` sur tout le modèle).

```bash
.venv/bin/python -m training.moe_target_train \
  --model-id mistralai/Mistral-Small-24B-Instruct-2501 \
  --corpus bench/train_corpus_v1.jsonl \
  --layers 20 \
  --num-experts 4 \
  --top-k 1 \
  --shared-experts 0 \
  --steps 12 \
  --lr 1e-5 \
  --max-length 64 \
  --device mps \
  --out-adapter bench/moe_training/mistral24b_moe_adapter_smoke.pt \
  --out-report bench/moe_training/mistral24b_moe_target_train_smoke.json
```

### 3) Eval A/B

```bash
.venv/bin/python -m training.moe_eval_ab \
  --model-id mistralai/Mistral-Small-24B-Instruct-2501 \
  --prompts bench/eval_quality_v1.json \
  --layers 20 \
  --shared-experts 0 \
  --warmup-steps 0 \
  --device mps \
  --adapter-path bench/moe_training/mistral24b_moe_adapter_smoke.pt \
  --out bench/moe_training/mistral24b_eval_ab_smoke.json
```

### 4) Bench MLX 4-bit (prod)

Quand un paquet **mlx-community** existe pour ce modèle :

```bash
.venv/bin/python -m training.mlx_moe_bench \
  --model-id mlx-community/<ID-4bit-Mistral-24B> \
  --max-tokens 32 \
  --out bench/moe_training/mlx_moe_bench_mistral24b_4bit.json
```

## Gates Go / No-Go (20–30B)

- **Mémoire** : pas de **swap runaway** ; si zone rouge répétée, réduire `max_length`, couches converties, ou passer en **MLX-only** plus tôt.
- **Qualité** : même logique que 7B — Δ loss / eval humaine dans des bornes acceptables.
- **Compute** : proxy **active experts / token** + latence p95 ; comparer **dense vs MoE** **à nombre de couches converties fixé**.
- **Export MLX** : No-Go si impossible de servir un **4-bit** raisonnable sans surcoût mémoire vs baseline dense 4-bit.

## Ordre de passage conseillé

1. **Mistral-Small-24B** (même stack que 7B) → valider toute la chaîne.
2. **Qwen2.5-32B** (ou Coder) → étendre bridge / clés FFN → répéter.

Document vivant : mettre à jour `docs/RD_SPRINTS_STATUS.md` (Sprint 8) avec les rapports `bench/moe_training/*24b*` / `*32b*` au fur et à mesure.
