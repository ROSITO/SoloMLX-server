# MoE expert offload (MLXServe / `mlx_lm`)

## Comportement

Le module `src/mlxserve/runtime/moe_offload.py` applique une **réduction statique du nombre d’experts résidents** au **chargement** du modèle MLX (`apply_moe_expert_offload`), pour les architectures supportées (ex. Mixtral, Qwen2-MoE selon implémentation).

- Les poids des experts non retenus sont **retirés des tenseurs chargés** (slice), pas seulement masqués.
- La stratégie `l2` classe les experts par norme L2 des poids de routeur ; `first` garde les premiers indices.
- Option **single-expert fastpath** : si un seul expert reste, un chemin forward simplifié peut être utilisé.

## Limites (important)

1. **Pas de chargement dynamique** entre tokens : il n’y a pas de prefetch disque, pas de swap « chaud/froid » à la volée comme dans certains papers (MoE-SpeQ, NPUMoE, etc.).
2. **Réduction du pic au load** : partiellement ; le pic peut rester dominé par le chargement initial ou d’autres buffers.
3. **Qualité** : réduire agressivement le nombre d’experts peut dégrader les sorties ; valider avec bench qualité (`moe_eval_ab`, prompts fixes).
4. **Couverture** : seuls les types de blocs reconnus dans `moe_offload.py` sont modifiés ; un nouveau layout MoE dans `mlx_lm` peut nécessiter une extension.

## Réglages serveur

Variables d’environnement (préfixe `MLXSERVE_`) : voir `src/mlxserve/config.py` — `MOE_RESIDENT_EXPERTS`, `MOE_RESIDENT_STRATEGY`, `MOE_SINGLE_EXPERT_FASTPATH`.

Bench : `python -m training.mlx_moe_bench --moe-resident-experts …`
