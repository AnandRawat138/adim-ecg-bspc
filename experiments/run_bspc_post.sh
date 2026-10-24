#!/usr/bin/env bash
# Runs after the main queue: masking analyses and federated simulations.
cd "$(dirname "$0")/.."
PY=./venv312/Scripts/python.exe
LOG=experiments/runs/bspc/queue.log
for s in 42 43 44; do for cfg in adim2 tattn; do
  [ -f experiments/runs/bspc/$cfg/seed$s/results.json ] || continue
  echo "$(date '+%F %T') START faithfulness $cfg seed$s" >> $LOG
  $PY -W ignore -m experiments.run_bspc_faithfulness --config $cfg --seed $s > experiments/runs/bspc/faith_${cfg}_${s}.out 2>&1
  echo "$(date '+%F %T') END   faithfulness $cfg seed$s rc=$?" >> $LOG
done; done
for args in "--partition iid" "--partition dirichlet --alpha 0.5" "--partition dirichlet --alpha 0.5 --mu 0.01"; do
  echo "$(date '+%F %T') START federated $args" >> $LOG
  $PY -W ignore -m experiments.run_bspc_federated $args --clients 5 --seed 42 > /dev/null 2>&1
  echo "$(date '+%F %T') END   federated $args rc=$?" >> $LOG
done
echo "$(date '+%F %T') POST COMPLETE" >> $LOG
