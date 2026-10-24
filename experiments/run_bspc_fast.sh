#!/usr/bin/env bash
# Time-boxed run: masking/inspection analyses on existing checkpoints, then one FedAvg (IID) simulation capped at 30 rounds.
cd "$(dirname "$0")/.."
PY=./venv312/Scripts/python.exe
LOG=experiments/runs/bspc/queue.log
echo "$(date '+%F %T') LAUNCH fast (pid $$)" >> $LOG
for job in "adim2 42" "adim2 43" "adim2 44" "tattn 42"; do set -- $job
  echo "$(date '+%F %T') START faithfulness $1 seed$2" >> $LOG
  $PY -W ignore -m experiments.run_bspc_faithfulness --config $1 --seed $2 > experiments/runs/bspc/faith_$1_$2.out 2>&1
  echo "$(date '+%F %T') END   faithfulness $1 seed$2 rc=$?" >> $LOG
done
echo "$(date '+%F %T') START federated iid" >> $LOG
$PY -W ignore -m experiments.run_bspc_federated --partition iid --clients 5 --seed 42 --rounds 30 --patience 8 > experiments/runs/bspc/fed_iid.out 2>&1
echo "$(date '+%F %T') END   federated iid rc=$?" >> $LOG
echo "$(date '+%F %T') FAST COMPLETE" >> $LOG
