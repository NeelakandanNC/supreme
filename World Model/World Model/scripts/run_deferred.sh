#!/usr/bin/env bash
# Deferred stages, run after the priority queue: SequenceRecall (capacity axis),
# continual learning, the lambda=1 ablation, and the corridor sweep figure.
set -uo pipefail
cd "$(dirname "$0")/.."

# Activate the project virtualenv. Runner scripts must not depend on the caller
# having done this: a bare `python` here resolves to the system interpreter,
# which has no torch, and with `set -e` absent every stage then "succeeds"
# instantly while doing nothing. That failure mode cost a full queue run.
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
python - <<'PYCHECK' || { echo "FATAL: project venv not active (torch missing)"; exit 1; }
import torch, sys           # noqa: F401
PYCHECK

# Refuse to start a second copy: two queues on one GPU thrash and interleave
# their logs into something unreadable.
LOCK="$(pwd)/.$(basename "$0").lock"
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "FATAL: $(basename "$0") already running as PID $(cat "$LOCK")"; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# Abort the queue on a failed stage rather than reporting DONE for a no-op.
run() { echo "\$ $*" >> "$LOG"; "$@" >> "$LOG" 2>&1 || { echo "FATAL: stage failed: $*" | tee -a "$LOG"; exit 1; }; }
LOG=runs_deferred.log
say() { echo "=== $(date '+%H:%M:%S')  $*" | tee -a "$LOG"; }

while pgrep -f "run_priority.sh" > /dev/null; do sleep 60; done
say "priority queue finished; starting deferred stages"

say "SequenceRecall (capacity axis)"
run python -u wm.py study -c configs/recall.yaml --backbones lstm,gru,attention,supreme \
    -o out_dir=runs_recall 
run python -u wm.py report runs_recall 

say "continual learning"
run python -u scripts/run_continual.py 

say "lambda = 1 ablation at the operating point"
run python -u wm.py study -c configs/tmaze_hard.yaml --backbones lstm,gru,attention,supreme \
    -o out_dir=runs_w1 -o memory.critical_loss_weight=1 -o memory.epochs=150 \
    --no-control 
run python -u wm.py report runs_w1 

say "corridor sweep"
run python -u scripts/sweep_horizon.py --backbones lstm,gru,supreme \
    --corridors 6,16,32,48 --rollouts 1200 --epochs 150 --out runs_sweep 

say "DEFERRED STAGES COMPLETE"
