#!/usr/bin/env bash
cd "$(dirname "$0")/.."
echo "$(date '+%F %T') LAUNCH queue2 (pid $$)" >> experiments/runs/bspc/queue.log
bash experiments/run_bspc_queue2.sh
bash experiments/run_bspc_post.sh
