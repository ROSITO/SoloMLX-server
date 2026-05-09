# LLM Inference Optimization Research
## Objectif
Rendre possible l’exécution locale d’un modèle de type 30B sur une machine limitée à 16 Go de RAM, notamment via :
- Mixture of Experts (MoE)
- Quantization agressive
- KV-cache compression
- Offloading intelligent
- MLX / Apple Silicon
- Speculative decoding
- Runtime optimisé

---

# 1. Surveys généraux

## A Survey of LLM Inference Systems
Vue globale des systèmes d’inférence :
- batching
- scheduling
- gestion mémoire
- kernels
- KV-cache
- quantization
- serving

Lien :
https://arxiv.org/html/2506.21901v1

---

## Efficient Large Language Models: A Survey

Taxonomie complète :
- model-centric optimization
- framework-centric optimization
- efficient serving

Lien :
https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey

---

## A Survey of Low-bit Large Language Models

Le survey probablement le plus important pour la quantization.

Couvre :
- INT8
- INT4
- INT2
- PTQ
- QAT
- low-bit inference

Lien :
https://arxiv.org/html/2409.16694v3

---

## A Systematic Evaluation of On-Device LLMs

Très pertinent pour Apple Silicon et petites machines.

Lien :
https://arxiv.org/html/2505.15030v5

---

# 2. MoE et Sparse Inference

## NPUMoE – Efficient Mixture-of-Experts LLM Inference with Apple Silicon NPUs

Sujet central :
- utilisation des NPUs Apple
- exécution sparse
- acceleration MoE sur Apple Silicon

Très proche d’un runtime MLX spécialisé.

Lien :
https://arxiv.org/abs/2604.18788

---

## MoE-SpeQ

Combine :
- speculative decoding
- expert offloading
- quantization

Très intéressant pour machines limitées.

Lien :
https://arxiv.org/html/2511.14102v1

---

## Mixture of Lookup Experts

Explique un problème majeur :
même si peu d’experts sont activés, les experts doivent souvent rester chargés en mémoire.

Lien :
https://openreview.net/forum?id=wUEp13rqXP&noteId=IRpmp05Y3X

---

## Awesome MoE Inference

Bibliographie MoE extrêmement utile.

Lien :
https://github.com/MoE-Inf/awesome-moe-inference/

---

# 3. Quantization

## GPTQ

Quantization post-training très populaire pour INT4.

Lien :
https://arxiv.org/abs/2210.17323

---

## AWQ

Activation-aware Weight Quantization.

Excellent compromis qualité/performance.

Lien :
https://arxiv.org/abs/2306.00978

---

## SmoothQuant

Facilite la quantization INT8 des activations.

Lien :
https://arxiv.org/abs/2211.10438

---

## SpQR

Sparse + quantization.

Lien :
https://arxiv.org/abs/2306.03078

---

## SqueezeLLM

Compression low-bit avancée.

Lien :
https://arxiv.org/abs/2306.07629

---

## AQLM

Additive Quantization for LLMs.

Lien :
https://arxiv.org/abs/2401.06118

---

## EfficientQAT

Quantization-aware training pour préserver la qualité.

Lien :
https://aclanthology.org/2025.acl-long.498.pdf

---

# 4. KV Cache Compression

## KIVI

Quantization asymétrique 2-bit du KV-cache.

Très important pour les longues context windows.

Lien :
https://proceedings.mlr.press/v235/liu24bz.html

---

## KVQuant

Compression KV-cache pour très longs contextes.

Lien :
https://openreview.net/forum?id=0LXotew9Du

---

## MiniCache

Compression inter-layer du KV-cache.

Lien :
https://proceedings.neurips.cc/paper_files/paper/2024/file/fd0705710bf01b88a60a3d479ea341d9-Paper-Conference.pdf

---

## TurboQuant

Compression KV-cache 3-bit.

Lien :
https://www.tomshardware.com/tech-industry/artificial-intelligence/googles-turboquant-compresses-llm-kv-caches-to-3-bits-with-no-accuracy-loss

---

# 5. Runtime et Attention

## vLLM / PagedAttention

Référence majeure pour :
- gestion mémoire
- paging
- throughput élevé

Lien :
https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-192.pdf

---

## vAttention

Alternative à PagedAttention.

Lien :
https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/vattention_arxiv24.pdf

---

## FlashAttention

Kernel attention ultra optimisé.

Lien :
https://arxiv.org/abs/2205.14135

---

## FlashAttention-2

Lien :
https://arxiv.org/abs/2307.08691

---

## FlashAttention-3

Lien :
https://tridao.me/blog/2024/flash3/

---

## FlashInfer

Moteur attention haute performance.

Lien :
https://arxiv.org/pdf/2501.01005

---

# 6. Speculative Decoding

## Fast Inference from Transformers via Speculative Decoding

Papier fondateur.

Lien :
https://arxiv.org/abs/2211.17192

---

## Medusa

Décodage parallèle multi-head.

Lien :
https://arxiv.org/abs/2401.10774

---

## EAGLE

Accélération speculative decoding.

Lien :
https://arxiv.org/abs/2401.15077

---

## EAGLE-3

Version améliorée.

Lien :
https://arxiv.org/html/2503.01840v1

---

## Mirror Speculative Decoding (Apple)

Optimisé pour Apple Silicon.

Lien :
https://machinelearning.apple.com/research/mirror

---

# 7. MLX / Apple Silicon

## MLX Official Documentation

Documentation officielle.

Lien :
https://ml-explore.github.io/mlx/

---

## mlx-lm

Inference et fine-tuning MLX.

Lien :
https://github.com/ml-explore/mlx-lm

---

## Exploring LLMs with MLX and Neural Accelerators in Apple Silicon

Article Apple Research.

Lien :
https://machinelearning.apple.com/research/exploring-llms-mlx-m5

---

## Production-Grade Local LLM Inference on Apple Silicon

Comparaison MLX / Ollama / llama.cpp / MLC.

Lien :
https://arxiv.org/abs/2511.05502

---

# Architecture cible recommandée

## Pipeline recommandé

30B MoE total
↓
3B à 6B actifs par token
↓
Experts quantifiés 3-bit ou 4-bit
↓
Experts froids offloadés
↓
Experts chauds gardés en mémoire unifiée
↓
KV-cache quantifié 2-bit / 3-bit
↓
Runtime MLX spécialisé
↓
Speculative decoding pour masquer les temps de chargement

---

# Sujet de recherche réellement critique

Le problème principal n’est PAS :
« faire rentrer les poids ».

Le vrai problème est :
- la latence de chargement des experts
- la prédiction des experts nécessaires
- la residency des experts
- la gestion du KV-cache
- la fragmentation mémoire
- l’ordonnancement dynamique

---

# Stack potentiellement la plus prometteuse

- MLX
- MLX-LM
- NPUMoE
- KIVI
- TurboQuant
- MoE-SpeQ
- speculative decoding
- expert prefetching
- KV-cache paging

---

# Hypothèse de projet extrêmement intéressante

Créer un runtime type Ollama/vLLM :
- entièrement MLX-native
- spécialisé MoE
- conçu pour Apple Silicon
- avec expert residency intelligente
- support du streaming d’experts
- quantization agressive
- speculative decoding
- KV-cache ultra compressé

Objectif :
faire tourner localement des modèles de classe 30B à 70B “effectifs”
sur des machines 16 Go Apple Silicon.

---

## MLXServe — plan d’exécution interne

Traduction en phases (P0–P3), liens vers le code et la doc existante :  
`docs/RD_INFERENCE_OPTIMIZATION_ROADMAP.md`
