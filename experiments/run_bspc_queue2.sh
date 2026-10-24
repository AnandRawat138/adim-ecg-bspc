#!/usr/bin/env bash
# Revised queue (ADIM-v2 is the proposed method). Resumable: completed runs are skipped.
cd "$(dirname "$0")/.."
PY=./venv312/Scripts/python.exe
LOG=experiments/runs/bspc/queue.log
run() { echo "$(date '+%F %T') START $1 seed$2" >> $LOG
        $PY -W ignore -m experiments.run_bspc_study --config $1 --seed $2 > /dev/null 2>&1
        echo "$(date '+%F %T') END   $1 seed$2 rc=$?" >> $LOG; }
# wait for any study run started by the previous queue
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | ? { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'run_bspc' }) { exit 0 } else { exit 1 }"; do sleep 60; done
for cfg in adim2 it_tr_bl tattn se it it_tr; do for s in 42 43 44; do run $cfg $s; done; done
for s in 42 43 44; do for cfg in adim2_wo_lgig adim2_wo_ain adim2_wo_ice adim2_wo_res cbam; do run $cfg $s; done; done
echo "$(date '+%F %T') QUEUE COMPLETE" >> $LOG
