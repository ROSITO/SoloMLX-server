#!/bin/zsh
# Six successive OpenCode tasks against Qwen3-Next served with SSD expert paging, logged by scripts/peek_proxy.py.
# usage: OUT=dir CHECKPOINT=/path/to/snapshot PACKED_DIR=/path/to/packed scripts/opencode_ssd_session.sh LABEL
S=${OUT:-$PWD/artifacts/opencode_session}; mkdir -p $S
R=$(cd "$(dirname "$0")/.." && pwd)
L=${1:-ssd}
pkill -f peek_proxy.py; pkill -f "mlxserve.cli serve"; sleep 2
rm -rf $S/oc_multi; mkdir -p $S/oc_multi; cd $S/oc_multi; git init -q; uv venv -q -p 3.12 .venv; uv pip install -q --python .venv/bin/python pytest
echo ".venv/" > .gitignore; echo "# Inventaire" > README.md; git add -A; git -c user.email=t@t -c user.name=t commit -qm init
export HF_HUB_OFFLINE=1 MLXSERVE_DEFAULT_MODEL="$CHECKPOINT" \
  MLXSERVE_CONTEXT_TOKENS=24576 MLXSERVE_MTP=false MLXSERVE_SSD_EXPERTS_ENABLED=true \
  MLXSERVE_SSD_EXPERTS_PACKED_DIR="$PACKED_DIR" MLXSERVE_SSD_EXPERTS_CACHE_GB=12 \
  MLXSERVE_SSD_EXPERTS_CACHE_POLICY=lfu MLXSERVE_SSD_EXPERTS_FREQUENCY_CAP=1 MLXSERVE_PREFILL_STEP_SIZE=1024 MLXSERVE_PROMPT_CACHE_GB=1
cd $R && MLXSERVE_PORT=8099 nohup .venv/bin/python -m mlxserve.cli serve > $S/server_$L.log 2>&1 &
for i in $(seq 60); do curl -s localhost:8099/health >/dev/null && break; sleep 1; done
: > $S/peek_$L.log; nohup $R/.venv/bin/python -I $R/scripts/peek_proxy.py $S/peek_$L.log > $S/proxy_$L.err 2>&1 &
sleep 2; cd $S/oc_multi
TASKS=(
 "Crée inventory.py avec une classe Inventory : add(name, qty, unit_price), remove(name, qty) qui lève ValueError si stock insuffisant, total_value(). Écris test_inventory.py avec pytest et lance '.venv/bin/python -m pytest -q'."
 "Ajoute à Inventory to_csv(path) et une classmethod from_csv(path) (colonnes name,qty,unit_price). Ajoute les tests correspondants (aller-retour) et relance les tests."
 "Ajoute un système de remises : apply_discount(name, percent) qui baisse le unit_price, refuse percent hors ]0,90] avec ValueError. Tests + relance."
 "Ajoute cli.py (argparse) avec les sous-commandes add, remove, total qui lisent/écrivent un fichier CSV passé en --file. Ajoute test_cli.py qui appelle main([...]) et relance tous les tests."
 "Ajoute un historique : chaque add/remove/apply_discount est enregistré dans self.history (liste de tuples (action, name, valeur)). Tests + relance."
 "Sans relire les fichiers, rappelle-moi : quelles méthodes as-tu ajoutées à Inventory depuis le début de la session, et quelle règle exacte s'applique au pourcentage de remise ? Ensuite vérifie en lançant les tests."
)
n=0; T00=$(date +%s)
for t in $TASKS; do
  n=$((n+1)); T0=$(date +%s); F=(); [ $n -gt 1 ] && F=(--continue)
  MLXSERVE_PORT=8098 timeout 1500 $R/.venv/bin/mlxserve opencode run $F "$t" < /dev/null > $S/task_${L}_$n.out 2>&1
  echo "TASK $n exit=$? $(( $(date +%s)-T0 ))s tests: $(.venv/bin/python -m pytest -q 2>&1 | tail -1)"
done
echo "SESSION wall=$(( $(date +%s)-T00 ))s"
pkill -f peek_proxy.py; pkill -f "mlxserve.cli serve"
