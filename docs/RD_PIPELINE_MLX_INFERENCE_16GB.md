# Pipeline cible : **MoE → MLX 4-bit → mesure coût d’inférence** (pas de PyTorch en prod)

## Principe

1. **Convertir** le checkpoint dense en **MoE** (FFN → experts + routeur), idéalement en **Safetensors** encore lisible par l’écosystème HF / mlx-lm.
2. **Ne pas servir l’inférence en PyTorch** : l’outil torch reste au plus une **étape transitoire** de R&D si indispensable pour produire des poids ; la **ligne de mesure et de prod** est **MLX + 4-bit**.
3. **Quantifier en 4-bit** pour MLX via `mlx_lm` (`convert` avec `quantize` / paquets `mlx-community` déjà prêts).
4. **Mesurer** uniquement sur MLX : `training/mlx_moe_bench.py` (pic mémoire MLX, temps, tokens/s) et/ou MLXServe + charge réelle.

## Budget RAM : **≤ 16 GiB** (objectif produit)

Interprétation pratique :

- **16 GiB** = enveloppe **processus / pic MLX** lors d’une génération représentative (**prompt + `max_tokens`** fixés au protocole), pas “toute la machine”.
- Sous **mémoire unifiée**, garder de la marge pour macOS : viser idéalement **pic MLX &lt; ~12–14 GiB** pour tenir le plafond **16 GiB** sans swap agressif.
- Un **MoE custom** charge en général **tous les experts** en RAM même si le **compute** n’en active qu’une partie : le **4-bit** est ce qui permet encore de tenir une **capacité “large”** dans l’enveloppe. Pour la **classe 32B**, le **dense 4-bit** seul peut déjà **dépasser 16 GiB** de pic MLX (poids + runtime) : voir mesures ci‑dessous.

**Candidats pour ≤ 16 GiB (ordre typique)** :

- **7B / 8B** en **4-bit** (+ MoE maîtrisé si export MLX OK) : marge confortable.
- **14B** 4-bit : **à valider** par mesure selon contexte / KV.
- **~24B** en **dense 4-bit MLX** (paquet `mlx-community`) : **mesuré sous le plafond** sur protocole court (voir ci‑dessous) — à re‑valider avec **prompt long** et **`max_tokens` service**.
- **~32B** en **dense 4-bit MLX** : **pic mesuré au‑delà de 16 GiB** sur le même protocole ; pour viser **16 GiB** en **32B**, prévoir **3-bit** (ex. Coder Instruct sur le hub) ou machine plus large / contexte très réduit.

### Mesures `mlx_moe_bench` (gate `--ram-budget-gib 16`, **mai 2026**)

Protocole : prompt par défaut du script (~63 caractères), `max_tokens=16`, température 0. Machine : Mac de référence projet (mémoire unifiée). Fichiers JSON sous `bench/moe_training/`.

| Modèle (mlx-community) | Quant | `peak_memory_gib_worst` | `within_ram_budget` (16) |
|------------------------|-------|-------------------------|---------------------------|
| `Mistral-Small-24B-Instruct-2501-4bit` | 4-bit | **~12.43** | **oui** |
| `Qwen2.5-32B-Instruct-4bit` | 4-bit | **~17.27** | **non** |
| `Qwen2.5-Coder-32B-Instruct-3bit` | 3-bit | **~13.47** | **oui** |

Interprétation : l’objectif **« 32B + 4-bit + ≤ 16 GiB »** est **contraint par la taille des poids** (ordre **~16 GiB** pour 32B en 4-bit, avant marge OS / KV / activations). La voie **32B sous 16 GiB** validée ici passe par **3-bit** (ou MoE **natif** dont l’empreinte chargée est inférieure — à bench séparément). La chaîne **dense → MoE custom → MLX** pour 24/32B reste à **fermer côté export** (même gate que le 7B).

Commandes (rejeu) :

```bash
.venv/bin/python -m training.mlx_moe_bench \
  --model-id mlx-community/Mistral-Small-24B-Instruct-2501-4bit \
  --max-tokens 16 --ram-budget-gib 16 \
  --out bench/moe_training/mlx_moe_bench_mistral_small_24b_4bit_16g_gate.json

.venv/bin/python -m training.mlx_moe_bench \
  --model-id mlx-community/Qwen2.5-32B-Instruct-4bit \
  --max-tokens 16 --ram-budget-gib 16 \
  --out bench/moe_training/mlx_moe_bench_qwen25_32b_instruct_4bit_16g_gate.json

.venv/bin/python -m training.mlx_moe_bench \
  --model-id mlx-community/Qwen2.5-Coder-32B-Instruct-3bit \
  --max-tokens 16 --ram-budget-gib 16 \
  --out bench/moe_training/mlx_moe_bench_qwen25_coder_32b_instruct_3bit_16g_gate.json
```

## Chaîne technique (cible)

```text
Dense HF safetensors
    → MoE safetensors (conversion FFN)
    → bundle compatible mlx-lm (HF ou dossier local)
    → mlx_lm.convert(..., quantize=True, q_bits=4)  [ou équivalent]
    → dossier MLX local ou repo mlx-community privé
    → mlx_moe_bench + gate --ram-budget-gib 16
    → MLXServe (backend mlx)
```

**Blocage R&D actuel (reproduit, mai 2026)** : les bundles locaux préparés depuis `*.moe-bootstrap.safetensors` (24B et 32B) échouent au `mlx_lm.load()` avec des paramètres "extra" de type `model.layers.<L>.moe.experts.<i>.{gate,up,down}_proj.weight` (12 clés pour une couche convertie, `num_experts=4`). Cela confirme que `mlx-lm` n'accepte pas ces clés sans **classe modèle MLX MoE** correspondante (ou mapping explicite vers une archi MoE supportée).

Outil de repro: `python -m tools.prepare_moe_hf_bundle --source-dir ... --base-repo ... --out-dir ... --smoke-load`
- 24B: `bench/moe_conversion/mistral_small_24b_moe_hf/` → `mlx_load_ok=false`
- 32B: `bench/moe_conversion/qwen25_32b_moe_hf/` → `mlx_load_ok=false`

Livrable clé restant : **aligner les clés + config sur une archi MoE réellement supportée par `mlx_lm`** (ou ajouter un module MLX minimal dédié), puis relancer `mlx_lm.convert(..., q_bits=4)` et `mlx_moe_bench`.

Point structurel confirmé : nos artefacts `*.moe-bootstrap.safetensors` actuels sont **hybrides** (une seule couche MoE convertie).
- 24B : couche convertie = `20`
- 32B : couche convertie = `32`

Or les implémentations MoE natives de `mlx-lm` (ex. `mixtral`, `qwen2_moe`) attendent une architecture cohérente au niveau modèle (couches MoE selon leur schéma), pas un patch "1 couche MoE" dans un modèle dense standard. Sans **classe MLX hybride dense+MoE** dédiée, `mlx_lm.load()` ne peut pas servir directement ces bundles.


### Validation end-to-end (custom MoE, 7B) — **mai 2026**

Pipeline exécuté en local de bout en bout :

1. Dense 7B (`mistralai/Mistral-7B-Instruct-v0.3`) -> bundle HF local **full-MoE** style Mixtral (`num_experts=2`, `top_k=1`, couches FFN converties = toutes) via `tools/convert_dense_to_mixtral_moe_bundle.py`.
2. Smoke load : `mlx_lm.load()` **OK** sur `bench/moe_conversion/mistral7b_full_mixtral_moe_e2_v2/`.
3. Quantification : `python -m mlx_lm convert --hf-path ... --mlx-path ... --quantize --q-bits 4` **OK**.
4. Bench inférence : `training/mlx_moe_bench --ram-budget-gib 16` **OK**.

Résultat custom MoE 7B (MLX 4-bit) :
- modèle : `bench/moe_conversion/mistral7b_full_mixtral_moe_e2_v2_mlx_4bit`
- `within_ram_budget=true`
- `peak_memory_gib_worst=6.841`
- `tokens_per_s=12.3588`
- rapport : `bench/moe_training/mlx_moe_bench_mistral7b_full_moe_e2_mlx4bit_16g.json`

Baseline dense 7B (MLX 4-bit, même protocole) :
- modèle : `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
- `peak_memory_gib_worst=3.922`
- `tokens_per_s=23.0088`
- rapport : `bench/moe_training/mlx_moe_bench_mistral7b_dense_mlx4bit_16g_cmp.json`

Interprétation : la chaîne **custom dense -> MoE -> MLX -> 4-bit -> bench** est validée sur 7B ; le profil mesuré ici privilégie la capacité MoE (2 experts résidents) au détriment de RAM/latence vs dense 7B 4-bit.

### Extension 14B (Qwen2.5) — **mai 2026**

Pipeline exécuté sur `Qwen/Qwen2.5-14B-Instruct` avec export **qwen2_moe** custom (2 experts, top_k=1, toutes couches FFN) :

- Outil : `tools/convert_dense_to_qwen2_moe_bundle.py`
- Bundle local : `bench/moe_conversion/qwen25_14b_full_qwen2_moe_e2_v3/`
- `mlx_lm.load(..., lazy=True)` : **OK**
- Quantification 4-bit : `bench/moe_conversion/qwen25_14b_full_qwen2_moe_e2_v3_mlx_4bit/` **OK**

Mesure RAM/compute (prompt default, `max_tokens=32`, gate 16 GiB):

| Modèle | Quant | `peak_memory_gib_worst` | `tokens_per_s` | `within_ram_budget` | Rapport |
|--------|-------|-------------------------|----------------|---------------------|---------|
| custom 14B full-MoE (`qwen2_moe`, 2 experts) | 4-bit | **18.494** | **0.5207** | **non** | `mlx_moe_bench_qwen25_14b_full_moe_e2_mlx4bit_16g.json` |
| custom 14B full-MoE (`qwen2_moe`, 2 experts) | 3-bit | **14.409** | **4.4028** | **oui** | `mlx_moe_bench_qwen25_14b_full_moe_e2_mlx3bit_16g.json` |
| dense 14B baseline (`mlx-community/Qwen2.5-14B-Instruct`) | 4-bit | **7.841** | **11.4647** | **oui** | `mlx_moe_bench_qwen25_14b_dense_4bit_16g_cmp.json` |
| dense 14B baseline (`mlx-community/Qwen2.5-14B-Instruct`) | 3-bit | **6.133** | **13.6862** | **oui** | `mlx_moe_bench_qwen25_14b_dense_3bit_16g_cmp.json` |

Lecture : sur cette implémentation full-MoE (experts tous résidents), la voie **14B custom MoE 4-bit** ne tient pas le budget 16 GiB ; la voie **3-bit** passe en RAM mais reste nettement plus lente que le dense 14B quantifié.


### Délestage experts MLX runtime (feature clé) — **mai 2026**

Implémentation livrée : `src/mlxserve/runtime/moe_offload.py` + intégration backend `MLXLMBackend` via settings:
- `MLXSERVE_MOE_RESIDENT_EXPERTS` (0 = désactivé)
- `MLXSERVE_MOE_RESIDENT_STRATEGY` (`l2` ou `first`)

Principe : pour `mixtral` et `qwen2_moe`, on conserve seulement `K` experts résidents par couche (slicing des matrices experts + gate), puis on ajuste `num_experts_per_tok` si nécessaire.

Validation bench (`training/mlx_moe_bench --moe-resident-experts 1`):

| Modèle custom MoE | Quant | Offload | `peak_memory_gib_worst` | `active_memory_gib_end` | `tokens_per_s` | Gate 16 GiB |
|-------------------|-------|---------|-------------------------|-------------------------|----------------|-------------|
| 7B full-MoE e2 | 4-bit | non | 6.841 | 6.751 | 12.3588 | OK |
| 7B full-MoE e2 | 4-bit | keep=1 | 6.950 | 3.797 | 7.6528 | OK |
| 14B full-MoE e2 | 4-bit | non | 18.494 | 18.418 | 0.5207 | KO |
| 14B full-MoE e2 | 4-bit | keep=1 | 18.595 | 13.078 | 1.6177 | KO (pic load) |
| 14B full-MoE e2 | 3-bit | keep=1 | 14.510 | 10.172 | 6.5233 | OK |

Lecture : le délestage réduit fortement la mémoire **active en régime** (`active_memory_gib_end`), mais ne baisse pas suffisamment le **pic de chargement** (`peak_memory_gib_worst`) pour faire tenir le 14B en 4-bit sous 16 GiB. En 3-bit, 14B + offload tient le budget.

Pertinence sortie (sanity check manuel, prompt explicatif court):
- 7B custom MoE avec `keep_experts=1`: réponse cohérente, similaire au mode sans offload.
- 14B custom `qwen2_moe` avec `keep_experts=1` (avec/sans fastpath): réponse dégradée / hors consigne observée.

Conclusion produit: activer agressivement `keep_experts=1` n'est pas robuste qualité sur 14B. Pour 14B, conserver `keep_experts=2` (ou offload désactivé) maintient mieux la pertinence ; le gain RAM clé reste la quantification (3-bit/4-bit).

## Mesure du coût d’inférence (MLX uniquement)

- Script : `python -m training.mlx_moe_bench --model-id ... --max-tokens ... --out ...`
- Gate RAM optionnelle : `--ram-budget-gib 16` → champ `within_ram_budget` dans le JSON ; code de sortie **2** si dépassement.
- Métriques à suivre : `tokens_per_s`, `generate_wall_s`, `peak_memory_gib_*`, et à terme **comparatif** dense 4-bit vs MoE 4-bit **même taille “catalogue”**.

## Rapport avec Mixtral mlx-community

Les paquets **Mixtral 4-bit** sont une **preuve** “MoE + MLX + quant” ; ils ne remplacent pas le **MoE custom** dense→MoE, mais valident toolchain **bench + serveur**. Le **gain compute** que tu vises (FFN partiellement activé) doit être démontré sur **ton** checkpoint une fois import MLX.
