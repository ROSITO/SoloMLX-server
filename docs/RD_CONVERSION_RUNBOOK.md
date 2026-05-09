# RD Conversion Runbook (Sprint 2)

Runbook pour conversion Dense -> MoE bootstrap sur checkpoint reel.

## Modele de reference courant

- `HuggingFaceTB/SmolLM2-135M-Instruct`
- Licence: Apache-2.0
- Fichier poids: `model.safetensors` (single-file)

## Commande de conversion executee

```bash
.venv/bin/python -m tools.run_sprint2_conversion \
  --repo-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --layers 2,6,10 \
  --workdir bench/moe_conversion/smollm2_135m
```

## Artefacts generes

- `bench/moe_conversion/smollm2_135m/model.safetensors`
- `bench/moe_conversion/smollm2_135m/model.moe-bootstrap.safetensors`
- `bench/moe_conversion/smollm2_135m/conversion_report.json`
- `bench/moe_conversion/smollm2_135m/active_params_proxy.json`

## Rapport attendu (resume)

- `ffn_keys_detected`: 90
- `ffn_keys_selected`: 9
- `converted_keys`: 9
- `layers`: [2, 6, 10]

## Notes techniques

- Le convertisseur duplique les tensors FFN vers:
  - `moe.experts.{i}`
  - `moe.shared_experts.{j}`
- Les tensors sont clones pour eviter les erreurs de partage memoire safetensors.
- Cette etape ne fait pas encore d'entrainement; elle valide la conversion structurelle.
- Le proxy de cout actif est calcule via:

```bash
.venv/bin/python -m tools.estimate_active_params_proxy \
  --report bench/moe_conversion/smollm2_135m/conversion_report.json \
  --out bench/moe_conversion/smollm2_135m/active_params_proxy.json
```
