# MoE sous MLX — trajectoire prod (cost-first)

**Inférence** : viser **MLX 4-bit** + `mlx_moe_bench` (option `--ram-budget-gib 16`), sans PyTorch en chemin critique — détail **`docs/RD_PIPELINE_MLX_INFERENCE_16GB.md`**.

## Constats techniques

1. **mlx-lm supporte déjà des MoE natifs** (architecture **Mixtral** : `MixtralSparseMoeBlock`, `SwitchGLU`, top‑`k` experts par token). Ce n’est pas du PyTorch : c’est du **MLX** exécutable sur GPU Apple via Metal.
2. **MLXServe** charge tout modèle compatible `mlx_lm.load()` via `MLXLMBackend`. Un checkpoint **`mlx-community/...Mixtral...-4bit`** est donc déjà un **MoE sous MLX** servable par l’API actuelle, avec **quantification** (RAM bien inférieure à un 7B flottant PyTorch+MPS).
3. La branche R&D **dense → MoE custom** (conversion Safetensors + entraînement PyTorch) est **orthogonal** : elle vise un MoE *spécifique*; la livraison *immédiate* “MoE + efficacité” côté prod = **Mixtral (ou autre MoE mlx-community) en 4-bit** jusqu’à export MLX des poids custom.

## Alignement “efficacité algorithmique” (MIT / littérature)

- Mesurer le coût réel d’inférence (prix, latence, **RAM pic**, tokens/s) autant que le score bench — logique proche des travaux sur **baisse du coût d’inférence** et **efficacité algorithmique** ([FutureTech / arXiv](https://arxiv.org/abs/2511.23455), page projet : [The Price of Progress](https://futuretech.mit.edu/publication/the-price-of-progress-algorithmic-efficiency-and-the-falling-cost-of-ai-inference)).
- La **compression / structure parcimonieuse** pendant ou après l’apprentissage (ex. repérer des sous-réseaux efficaces) reste une boussole de recherche ([Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)).
- Les approches **réduction de complexité pendant l’entraînement** (ex. travaux type CompreSSM / MIT News) renforcent l’idée : **efficacité = objectif de conception**, pas uniquement post-hoc.

## Prochaines étapes (ordre recommandé)

| Étape | Action | Gate |
|------|--------|------|
| A | Servir `mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit` (ou variante 4-bit disponible) via MLXServe | Latence + RAM mesurées vs dense 7B 4-bit |
| B | Script `training/mlx_moe_bench.py` : pic mémoire MLX, temps, métadonnées `num_experts_per_tok` / `num_local_experts` | Rapport JSON reproductible |
| C | (R&D) Export poids MoE **custom** (notre conversion PyTorch) vers format **mlx-lm** | Go/No-Go selon parité qualité + charge |

## Commandes utiles

```bash
# Bench MoE MLX (premier run = téléchargement modèle)
.venv/bin/python -m training.mlx_moe_bench \
  --model-id mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit \
  --max-tokens 32 \
  --out bench/moe_training/mlx_moe_bench_mixtral.json

# Baseline dense MLX comparable
.venv/bin/python -m training.mlx_moe_bench \
  --model-id mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --max-tokens 32 \
  --out bench/moe_training/mlx_moe_bench_mistral7b_dense.json
```

## Note sur le compute “considérablement” réduit

- Pour un **Mixtral 8x7B**, le **nombre total de paramètres** est grand, mais les **paramètres activés par token** sont contrôlés par `num_experts_per_tok` (souvent 2) sur `num_local_experts` (souvent 8) dans la couche MoE — c’est là que vit le gain FLOPs/token **dans la partie FFN**, en plus du gain **mémoire** du 4-bit côté MLX.

Pour le passage **~20B / ~30B** (Mistral Small 24B, Qwen2.5 32B / Coder), voir **`docs/RD_SCALEUP_20_30B.md`**.

## Vision produit : **7B classique → MoE custom → MLX 4-bit**

Objectif explicitement **cost-first** :

1. **Partir d’un dense 7B** (ex. instruct / code) déjà maîtrisé.
2. **Remplacer une ou plusieurs couches FFN** par un bloc MoE **custom** (routage top‑`k`, experts issus d’un warm-start depuis le dense), puis **SFT / adapter** pour récupérer la qualité — comme déjà prototypé côté PyTorch.
3. **Exporter** le checkpoint vers un bundle **mlx-lm** + **quantification 4-bit** pour l’inférence dans **MLXServe**.

**Ce que ça réduit vraiment**

- **Compute d’inférence (FLOPs / token)** sur les **couches MoE** : on n’exécute qu’une **fraction** des experts par token (`top_k / num_experts` sur le FFN concerné). C’est le cœur du gain “MIT-style” : **moins d’opérations utiles par token** pour une capacité proche si le routeur + training sont bons.
- **RAM totale** : en général le modèle **garde tous les experts en mémoire** (poids chargés) sauf architecture / runtime spécifique ; le gain RAM **principal** vient surtout du **4-bit MLX**, pas du seul MoE. Le MoE custom aide surtout **latence FFN / énergie** tant que le dispatch reste efficace sur Metal.

**Risque / condition de succès**

- Il faut **fermer la boucle export** : état dict compatible `mlx_lm` (ou modèle custom minimal dans `mlx_lm.models`) + **quants** — c’est l’étape **C** du tableau ci-dessus.

## Piste de montée : **QwenCoder &lt; 20B en MoE MLX 4-bit**

Hypothèse réaliste : une fois le pipeline **dense → MoE → MLX 4-bit** validé sur **7B**, on peut **réutiliser la même mécanique** sur une famille **code** (ex. Qwen * Coder) **sous 20B** : mêmes types de couches FFN, même routage, même protocole de bench (`mlx_moe_bench`, gates qualité / tokens/s / `peak_memory_gib`).

**À valider avant de promettre le gain**

- Taille **totale** du modèle cible vs **RAM unifiée** du Mac (un 20B 4-bit peut tenir, mais marge + KV + pression système doivent être mesurées).
- **Parité qualité** sur tâches code après MoE + 4-bit (régression acceptable bornée).
- **Dispatch MoE** performant en MLX (éviter le chemin “dense sur tous les experts” par erreur d’implémentation).
